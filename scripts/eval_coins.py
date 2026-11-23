"""M3 硬币检测对比：v1 默认 / 多尺度霍夫 / v1 分水岭 / v2 分水岭。

用法：
    python scripts/eval_coins.py --n 400
    python scripts/eval_coins.py --n 0        # 全量

产物：
    results/metrics/m3_coin_compare.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import imread_unicode  # noqa: E402
from src.preprocess import resize_long_side  # noqa: E402
from src.segment import (  # noqa: E402
    detect_coins_hough,
    detect_coins_multiscale,
    detect_coins_watershed,
    detect_coins_watershed_v2,
)

COIN_CSV = ROOT / "data/raw/count_coins/coins_count_values.csv"
COIN_ROOT = ROOT / "data/raw/count_coins/coins_images/coins_images"
OUT = ROOT / "results/metrics/m3_coin_compare.csv"

METHODS = {
    "hough_v1": lambda img: detect_coins_hough(img),
    "hough_multi": lambda img: detect_coins_multiscale(img),
    "ws_v1": lambda img: detect_coins_watershed(img),
    "ws_v2": lambda img: detect_coins_watershed_v2(img),
}


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = np.asarray(pred, float)
    gt = np.asarray(gt, float)
    tp = float(np.minimum(pred, gt).sum())
    p = tp / pred.sum() if pred.sum() else 0.0
    r = tp / gt.sum() if gt.sum() else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "mae": float(np.abs(pred - gt).mean()),
        "exact": float((pred == gt).mean()),
        "within1": float((np.abs(pred - gt) <= 1).mean()),
        "precision": p,
        "recall": r,
        "f1": f1,
        "pred_total": int(pred.sum()),
        "gt_total": int(gt.sum()),
    }


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df
    bins = [(1, 1), (2, 3), (4, 10), (11, 30), (31, 10**6)]
    parts = []
    for lo, hi in bins:
        sub = df[(df["coins_count"] >= lo) & (df["coins_count"] <= hi)]
        if len(sub):
            take = max(1, round(n * len(sub) / len(df)))
            parts.append(sub.sample(n=min(take, len(sub)), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()

    df = stratified_sample(pd.read_csv(COIN_CSV), args.n)
    rows = []
    t_all = time.perf_counter()

    for i, r in enumerate(df.itertuples(), 1):
        img = imread_unicode(COIN_ROOT / r.folder / r.image_name)
        if img is None:
            continue
        img = resize_long_side(img, 1000)
        rec = {
            "image": f"{r.folder}/{r.image_name}",
            "gt": int(r.coins_count),
            "w": img.shape[1],
            "h": img.shape[0],
        }
        for name, fn in METHODS.items():
            t0 = time.perf_counter()
            pred = fn(img)
            rec[f"{name}_n"] = int(len(pred))
            rec[f"{name}_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        rows.append(rec)
        if i % 100 == 0:
            print(f"  {i}/{len(df)}")

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")

    gt = out["gt"].values
    print(f"\n样本 {len(out)} 张，耗时 {time.perf_counter() - t_all:.1f}s")
    header = f"{'方法':<12}{'MAE':>7}{'完全正确':>9}{'±1':>7}{'P':>7}{'R':>7}{'F1':>7}{'预测/真值':>14}{'ms/张':>9}"
    print(header)
    print("-" * len(header))
    for name in METHODS:
        m = metrics(out[f"{name}_n"].values, gt)
        ms = out[f"{name}_ms"].mean()
        print(
            f"{name:<12}{m['mae']:>7.2f}{m['exact']:>8.1%}{m['within1']:>7.1%}"
            f"{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}"
            f"{m['pred_total']:>7d}/{m['gt_total']:<6d}{ms:>9.1f}"
        )
    print(f"\n已写出 {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
