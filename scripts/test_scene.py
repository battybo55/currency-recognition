"""背景鲁棒性测试：把同一张纸币与硬币贴到不同背景上，检查检测是否稳定。

思路：目标本身不变、位置与尺度可控、背景完全替换，
这样背景就是唯一自变量，能直接看出前景提取是否依赖某种特定桌面。

用法：
    python scripts/test_scene.py
    python scripts/test_scene.py --save     # 同时保存合成图到 data/raw/_evidence/scene_test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import imread_unicode, imwrite_unicode  # noqa: E402
from src.preprocess import resize_long_side  # noqa: E402
from src.scene import SceneConfig, extract_objects  # noqa: E402

SELF = ROOT / "data/raw/self"
EVID = ROOT / "data/raw/_evidence/scene_test"
W, H = 1000, 750


def make_backgrounds(wood_src: np.ndarray) -> dict[str, np.ndarray]:
    """四种背景：纯白、深色、渐变、木纹（来自实拍裁片）。"""
    white = np.full((H, W, 3), 245, np.uint8)

    dark = np.full((H, W, 3), 32, np.uint8)
    noise = np.random.default_rng(0).normal(0, 4, dark.shape)
    dark = np.clip(dark.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    grad = np.tile(np.linspace(40, 220, W, dtype=np.float32), (H, 1))
    gradient = cv2.cvtColor(grad.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    tile = cv2.resize(wood_src, (W, H), interpolation=cv2.INTER_LINEAR)
    return {"white": white, "dark": dark, "gradient": gradient, "wood": tile}


def cut_object(img: np.ndarray, obj: dict) -> tuple[np.ndarray, np.ndarray]:
    x, y, w, h = obj["bbox"]
    return img[y:y + h, x:x + w].copy(), obj["mask"][y:y + h, x:x + w].copy()


def paste(bg: np.ndarray, crop: np.ndarray, mask: np.ndarray, x: int, y: int, scale: float):
    ch, cw = int(crop.shape[0] * scale), int(crop.shape[1] * scale)
    c = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
    x2, y2 = min(x + cw, bg.shape[1]), min(y + ch, bg.shape[0])
    cw, ch = x2 - x, y2 - y
    roi = bg[y:y2, x:x2]
    roi[m[:ch, :cw] > 0] = c[:ch, :cw][m[:ch, :cw] > 0]
    return (x, y, cw, ch)


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    src = resize_long_side(imread_unicode(SELF / "s02.jpg"), 1000)
    objs, _, _ = extract_objects(src, SceneConfig())
    notes = [o for o in objs if o["kind"] == "note"]
    coins = [o for o in objs if o["kind"] == "coin"]
    if not notes or len(coins) < 2:
        raise SystemExit("源图 s02 未提取到足够的纸币/硬币，先检查 scene 模块")

    note_crop, note_mask = cut_object(src, max(notes, key=lambda o: o["area"]))
    coin_crops = [cut_object(src, c) for c in coins[:3]]

    wood_src = resize_long_side(imread_unicode(SELF / "s01.jpg"), 1000)[:400, :600]
    backgrounds = make_backgrounds(wood_src)

    cfg = SceneConfig()
    print(f"{'背景':<10}{'缩放':>6}{'纸币命中':>10}{'IoU':>8}{'硬币检出':>10}")
    print("-" * 46)
    hit_note = hit_coin = total = 0
    for bname, bg in backgrounds.items():
        for scale in (0.55, 0.75, 0.95):
            canvas = bg.copy()
            nbox = paste(canvas, note_crop, note_mask, 120, 120, scale)
            cboxes = []
            for i, (cc, cm) in enumerate(coin_crops):
                cboxes.append(paste(canvas, cc, cm, 640 + (i % 2) * 130, 420 + (i // 2) * 140, scale))

            res, _, _ = extract_objects(canvas, cfg)
            det_notes = [o for o in res if o["kind"] == "note"]
            det_coins = [o for o in res if o["kind"] == "coin"]

            best_iou = max((iou(o["bbox"], nbox) for o in det_notes), default=0.0)
            note_ok = best_iou >= 0.3
            coin_hits = sum(any(iou(o["bbox"], cb) > 0.2 for o in det_coins) for cb in cboxes)
            coin_ok = coin_hits >= 2

            hit_note += note_ok
            hit_coin += coin_ok
            total += 1
            print(f"{bname:<10}{scale:>6.2f}{'是' if note_ok else '否':>10}{best_iou:>8.2f}{coin_hits:>7}/3")

            if args.save:
                imwrite_unicode(EVID / f"{bname}_{int(scale*100)}.png", canvas)

    print("-" * 46)
    print(f"纸币命中 {hit_note}/{total}，硬币命中 {hit_coin}/{total}（判定：至少 2/3 枚找到）")


if __name__ == "__main__":
    main()
