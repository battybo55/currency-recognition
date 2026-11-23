"""硬币误差切片（E6 部分）：按真值密度与图像尺度分组看最终方法的失效位置。

用法：
    python scripts/slice_coins.py --n 400 --support 0.55

产物：
    results/metrics/e6_coin_slices.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import imread_unicode  # noqa: E402
from src.preprocess import resize_long_side  # noqa: E402
from src.segment import circle_edge_support, detect_coins_multiscale  # noqa: E402

COIN_CSV = ROOT / "data/raw/count_coins/coins_count_values.csv"
COIN_ROOT = ROOT / "data/raw/count_coins/coins_images/coins_images"
OUT = ROOT / "results/metrics/e6_coin_slices.csv"


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = np.asarray(pred, float)
    gt = np.asarray(gt, float)
    tp = float(np.minimum(pred, gt).sum())
    p = tp / pred.sum() if pred.sum() else 0.0
    r = tp / gt.sum() if gt.sum() else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"n": len(gt), "mae": float(np.abs(pred - gt).mean()),
            "exact": float((pred == gt).mean()), "precision": p, "recall": r, "f1": f1}


def stratified(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df.reset_index(drop=True)
    parts = []
    for lo, hi in [(1, 1), (2, 3), (4, 10), (11, 30), (31, 10**6)]:
        sub = df[(df["coins_count"] >= lo) & (df["coins_count"] <= hi)]
        if len(sub):
            take = max(1, round(n * len(sub) / len(df)))
            parts.append(sub.sample(n=min(take, len(sub)), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--support", type=float, default=0.55)
    args = ap.parse_args()

    df = stratified(pd.read_csv(COIN_CSV), args.n)
    rows = []
    for i, r in enumerate(df.itertuples(), 1):
        img = imread_unicode(COIN_ROOT / r.folder / r.image_name)
        if img is None:
            continue
        img = resize_long_side(img, 1000)
        circles = detect_coins_multiscale(img, min_support=args.support)
        rows.append({"gt": int(r.coins_count), "pred": len(circles), "short": min(img.shape[:2])})
        if i % 100 == 0:
            print(f"  {i}/{len(df)}")

    d = pd.DataFrame(rows)
    d["density"] = pd.cut(d["gt"], [0, 1, 3, 10, 30, 10**6],
                          labels=["1", "2–3", "4–10", "11–30", "31+"])
    d["scale"] = pd.cut(d["short"], [0, 300, 600, 10**6],
                        labels=["短边<300", "300–600", ">600"])

    out = []
    for group in ("density", "scale"):
        for name, g in d.groupby(group, observed=True):
            m = metrics(g["pred"].values, g["gt"].values)
            out.append({"slice": group, "bucket": str(name), **m})
    res = pd.DataFrame(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False, encoding="utf-8")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n已写出 {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
