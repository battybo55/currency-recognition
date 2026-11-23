"""单张纸币识别演示：输出与 MATLAB 参考流程一致的中间结果。

    python scripts/single_note_demo.py --image data/raw/_evidence/single_crops/s06_0.jpg
    python scripts/single_note_demo.py --image xxx.jpg --save out.png

阶段：原图 → 定位 → 透视校正 → 灰度 → 二值 → 面额数字区域 → 识别结果
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import imread_unicode, imwrite_unicode  # noqa: E402
from src.preprocess import resize_long_side  # noqa: E402
from src.rmb import RMBConfig, classify_rmb  # noqa: E402
from src.single_note import binarize, locate_note, numeral_region, warp_note  # noqa: E402

PANEL = (480, 240)


def panel(img: np.ndarray, title: str) -> np.ndarray:
    out = np.full((PANEL[1] + 34, PANEL[0], 3), 255, np.uint8)
    h, w = img.shape[:2]
    s = min(PANEL[0] / w, PANEL[1] / h)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    y0 = 30 + (PANEL[1] - small.shape[0]) // 2
    x0 = (PANEL[0] - small.shape[1]) // 2
    out[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small if small.ndim == 3 else cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--save", default="")
    ap.add_argument("--json-out", default="", help="把结果 JSON 写到该文件（供 GUI 读取）")
    args = ap.parse_args()

    img = resize_long_side(imread_unicode(args.image), 1000)
    if img is None:
        raise SystemExit(f"无法读取 {args.image}")

    q = locate_note(img)
    stage: dict[str, np.ndarray] = {"原图": img}
    result = {"image": Path(args.image).name, "located": q is not None}

    if q is None:
        # 兜底：定位失败时按整图判定。面额判据用的是颜色，与方向/位置无关，
        # 所以只要票面占画面主体，仍能给出面额（附低置信提示）。
        warped = cv2.resize(img, (900, 439), interpolation=cv2.INTER_AREA)
        r = classify_rmb(warped, RMBConfig())
        result.update({
            "located": False,
            "denomination": r["value"],
            "label": r["label"],
            "confidence": round(r["confidence"] * 0.8, 3),   # 定位失败，置信打八折
            "candidates": r["candidates"],
            "fallback": True,
        })
        label_img = warped.copy()
        cv2.putText(label_img, f"{r['label'] or '未识别'}（定位失败，按整图判定）",
                    (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2, cv2.LINE_AA)
        stage["定位失败（按整图判定）"] = label_img
    else:
        warped = warp_note(img, q)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        binimg = binarize(warped)
        nr = numeral_region(warped)

        marked = img.copy()
        cv2.polylines(marked, [q.quad.astype(np.int32)], True, (0, 200, 0), 3)
        stage.update({"定位": marked, "透视校正": warped, "灰度": gray, "二值": binimg})
        if nr is not None:
            crop, box = nr
            stage["面额数字区域"] = crop
            vis = warped.copy()
            cv2.rectangle(vis, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 3)
            stage["数字区域位置"] = vis
        else:
            stage["面额数字区域"] = warped

        r = classify_rmb(warped, RMBConfig())
        result.update({
            "denomination": r["value"],
            "label": r["label"],
            "confidence": r["confidence"],
            "candidates": r["candidates"],
            "features": r.get("features"),
            "quad_area": round(q.area_ratio, 3),
            "quad_aspect": round(q.aspect, 2),
            "contrast": round(q.contrast, 1),
        })

        label_img = warped.copy()
        text = f"{r['label']}" if r["label"] else "未识别"
        cv2.rectangle(label_img, (0, 0), (label_img.shape[1], 70), (255, 255, 255), -1)
        cv2.putText(label_img, text, (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 200), 3, cv2.LINE_AA)
        stage["识别结果"] = label_img

    cols = 4
    items = list(stage.items())
    tiles = [panel(v, k) for k, v in items]
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    montage = np.vstack(rows)

    out = Path(args.save) if args.save else ROOT / "data/raw/_evidence/single_demo.png"
    imwrite_unicode(out if out.is_absolute() else ROOT / out, montage)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"已写出 {out}")


if __name__ == "__main__":
    main()
