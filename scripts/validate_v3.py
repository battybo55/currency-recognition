"""V3 跑通性验证：不调参，只确认管线能稳定产出结果。

用法：
    python scripts/validate_v3.py                 # 全量硬币 + 30 张纸币
    python scripts/validate_v3.py --coin-limit 120  # 快速试跑

产物：
    results/metrics/v3_external.csv        指标体系（可入库）
    data/raw/_evidence/v3/*.png            可视化拼图（不入库）
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import imread_unicode, imwrite_unicode  # noqa: E402
from src.preprocess import resize_long_side  # noqa: E402
from src.segment import (  # noqa: E402
    detect_coins_hough,
    detect_coins_watershed,
    find_banknote_quad,
    warp_quad,
)

COIN_CSV = ROOT / "data/raw/count_coins/coins_count_values.csv"
COIN_ROOT = ROOT / "data/raw/count_coins/coins_images/coins_images"
BN_ZIP = ROOT / "data/raw/indian_currency.zip"
EVID = ROOT / "data/raw/_evidence/v3"
METRICS = ROOT / "results/metrics"


def load_cfg() -> dict:
    with open(ROOT / "configs/baseline.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    tp = float(np.minimum(pred, gt).sum())
    p = tp / pred.sum() if pred.sum() else 0.0
    r = tp / gt.sum() if gt.sum() else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "n": int(len(gt)),
        "mae": float(np.abs(pred - gt).mean()),
        "exact": float((pred == gt).mean()),
        "within1": float((np.abs(pred - gt) <= 1).mean()),
        "precision": p,
        "recall": r,
        "f1": f1,
        "pred_total": int(pred.sum()),
        "gt_total": int(gt.sum()),
    }


def run_coins(cfg: dict, limit: int = 0, seed: int = 0) -> tuple[list[dict], dict, dict]:
    df = pd.read_csv(COIN_CSV)
    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)

    c = cfg["coin"]
    rows = []
    pred_h, pred_w, gts = [], [], []
    for r in df.itertuples():
        img = imread_unicode(COIN_ROOT / r.folder / r.image_name)
        if img is None:
            continue
        img = resize_long_side(img, cfg["image"]["long_side"])
        t0 = time.perf_counter()
        hough = detect_coins_hough(
            img,
            radius_ratio=tuple(c["radius_ratio"]),
            dp=c["hough"]["dp"],
            min_dist_ratio=c["hough"]["min_dist_ratio"],
            param1=c["hough"]["param1"],
            param2=c["hough"]["param2"],
        )
        t1 = time.perf_counter()
        ws = detect_coins_watershed(
            img,
            min_distance_ratio=c["watershed"]["min_distance_ratio"],
            min_area_ratio=c["watershed"]["min_area_ratio"],
        )
        t2 = time.perf_counter()
        rows.append(
            {
                "dataset": "count_coins",
                "image": f"{r.folder}/{r.image_name}",
                "gt": int(r.coins_count),
                "hough": int(len(hough)),
                "watershed": int(len(ws)),
                "w": img.shape[1],
                "h": img.shape[0],
                "hough_ms": round((t1 - t0) * 1000, 1),
                "ws_ms": round((t2 - t1) * 1000, 1),
            }
        )
        pred_h.append(len(hough))
        pred_w.append(len(ws))
        gts.append(int(r.coins_count))

    mh = count_metrics(np.array(pred_h), np.array(gts))
    mw = count_metrics(np.array(pred_w), np.array(gts))
    return rows, mh, mw


def run_banknotes(cfg: dict, per_class: int = 3) -> tuple[list[dict], np.ndarray | None]:
    if not BN_ZIP.exists():
        return [], None
    cfg_bn = cfg["banknote"]
    z = zipfile.ZipFile(BN_ZIP)
    names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]

    by_cls: dict[str, list[str]] = {}
    for n in names:
        parts = n.split("/")
        if len(parts) >= 4:
            by_cls.setdefault(f"{parts[1]}/{parts[2]}", []).append(n)

    picks = []
    for cls in sorted(by_cls):
        picks.extend(sorted(by_cls[cls])[:per_class])

    rows, tiles = [], []
    for name in picks:
        buf = np.frombuffer(z.read(name), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = resize_long_side(img, cfg["image"]["long_side"])
        quad, method, ratio = find_banknote_quad(
            img,
            min_area_ratio=cfg_bn["min_area_ratio"],
            corner_method=cfg_bn["corner_method"],
        )
        ok = quad is not None
        rows.append(
            {
                "dataset": "indian_currency",
                "image": name.replace("Indian-Currency-Dataset/", ""),
                "gt": "",
                "hough": "",
                "watershed": "",
                "w": img.shape[1],
                "h": img.shape[0],
                "method": method,
                "area_ratio": round(float(ratio), 4),
                "ok": int(ok),
            }
        )
        if ok:
            tile = img.copy()
            cv2.polylines(tile, [quad.astype(np.int32)], True, (0, 255, 0), 3)
            warped = warp_quad(img, quad, tuple(cfg_bn["output_size"]))
        else:
            tile = img.copy()
            warped = np.zeros((cfg_bn["output_size"][1], cfg_bn["output_size"][0], 3), np.uint8)
        th = 180
        tile = cv2.resize(tile, (int(tile.shape[1] * th / tile.shape[0]), th))
        warped = cv2.resize(warped, (int(warped.shape[1] * th / warped.shape[0]), th))
        gap = np.full((th, 8, 3), 255, np.uint8)
        tiles.append(np.hstack([tile, gap, warped]))

    sheet = None
    if tiles:
        wmax = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, 0, 0, wmax - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
        cols = 3
        rows_n = (len(tiles) + cols - 1) // cols
        blank = np.full_like(tiles[0], 255)
        while len(tiles) < rows_n * cols:
            tiles.append(blank)
        sheet = np.vstack([np.hstack(tiles[i * cols:(i + 1) * cols]) for i in range(rows_n)])
        EVID.mkdir(parents=True, exist_ok=True)
        imwrite_unicode(EVID / "banknote_quad_sheet.png", sheet)
    return rows, sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin-limit", type=int, default=0)
    ap.add_argument("--bn-per-class", type=int, default=3)
    ap.add_argument("--skip-coins", action="store_true")
    ap.add_argument("--skip-banknote", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg()
    METRICS.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    if not args.skip_coins:
        print("[A] 硬币计数：Hough 与分水岭，默认参数，不调参")
        rows, mh, mw = run_coins(cfg, limit=args.coin_limit)
        all_rows.extend(rows)
        print(f"    样本 {mh['n']} 张")
        for name, m in (("Hough", mh), ("Watershed", mw)):
            print(
                f"    {name:<10} MAE={m['mae']:.2f}  exact={m['exact']:.1%}  "
                f"±1={m['within1']:.1%}  P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  "
                f"pred={m['pred_total']} gt={m['gt_total']}"
            )

    if not args.skip_banknote:
        print("[B] 纸币四角定位：Indian Currency，每类 3 张")
        rows_bn, _ = run_banknotes(cfg, per_class=args.bn_per_class)
        all_rows.extend(rows_bn)
        if rows_bn:
            ok = sum(r["ok"] for r in rows_bn)
            methods = pd.Series([r["method"] for r in rows_bn]).value_counts().to_dict()
            print(f"    样本 {len(rows_bn)} 张，定位成功 {ok}（{ok / len(rows_bn):.1%}）")
            print(f"    方法分布 {methods}")

    out = METRICS / "v3_external.csv"
    if all_rows:
        keys = sorted({k for r in all_rows for k in r})
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
        print(f"已写出 {out.relative_to(ROOT)}，{len(all_rows)} 行")


if __name__ == "__main__":
    main()
