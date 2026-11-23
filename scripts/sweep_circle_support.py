"""圆周边缘支持度的阈值扫描：在 dev 上选阈值，在 held-out 上报结果。

用法：
    python scripts/sweep_circle_support.py --n 400

产物：
    results/metrics/m3_support_sweep.csv   阈值扫描（dev）
    results/metrics/m3_support_eval.csv    held-out 对比
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import imread_unicode  # noqa: E402
from src.preprocess import resize_long_side, to_gray  # noqa: E402
from src.segment import circle_edge_support, detect_coins_multiscale  # noqa: E402

COIN_CSV = ROOT / "data/raw/count_coins/coins_count_values.csv"
COIN_ROOT = ROOT / "data/raw/count_coins/coins_images/coins_images"
METRICS = ROOT / "results/metrics"


def split_part(name: str) -> int:
    return int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16) % 2


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = np.asarray(pred, float)
    gt = np.asarray(gt, float)
    tp = float(np.minimum(pred, gt).sum())
    p = tp / pred.sum() if pred.sum() else 0.0
    r = tp / gt.sum() if gt.sum() else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"mae": float(np.abs(pred - gt).mean()), "exact": float((pred == gt).mean()),
            "precision": p, "recall": r, "f1": f1, "pred": int(pred.sum()), "gt": int(gt.sum())}


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


def collect(df: pd.DataFrame) -> list[tuple[int, np.ndarray]]:
    out = []
    for i, r in enumerate(df.itertuples(), 1):
        img = imread_unicode(COIN_ROOT / r.folder / r.image_name)
        if img is None:
            continue
        img = resize_long_side(img, 1000)
        circles = detect_coins_multiscale(img)
        scores = circle_edge_support(to_gray(img), circles)
        out.append((int(r.coins_count), scores))
        if i % 100 == 0:
            print(f"  {i}/{len(df)}")
    return out


def summarize(data, t: float) -> dict:
    preds = np.array([int((s >= t).sum()) for _, s in data])
    gts = np.array([g for g, _ in data])
    return metrics(preds, gts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()

    df = stratified(pd.read_csv(COIN_CSV), args.n)
    df["image"] = df["folder"] + "/" + df["image_name"]
    df["part"] = df["image"].map(split_part)
    dev, ev = df[df.part == 0], df[df.part == 1]
    print(f"dev {len(dev)} 张 / eval {len(ev)} 张")

    print("[dev] 提取候选与支持度")
    dev_data = collect(dev)
    print("[eval] 提取候选与支持度")
    ev_data = collect(ev)

    thresholds = [round(x, 2) for x in np.arange(0.0, 0.65, 0.05)]
    rows = []
    for t in thresholds:
        m = summarize(dev_data, t)
        rows.append({"threshold": t, **m})
    sweep = pd.DataFrame(rows)
    METRICS.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(METRICS / "m3_support_sweep.csv", index=False, encoding="utf-8")

    best = sweep.loc[sweep["f1"].idxmax()]
    print("\n[dev] 阈值扫描")
    print(sweep[["threshold", "mae", "exact", "precision", "recall", "f1"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"选定阈值 {best['threshold']:.2f}（dev F1 {best['f1']:.3f}）")

    t = float(best["threshold"])
    m0 = summarize(ev_data, 0.0)
    m1 = summarize(ev_data, t)
    report = pd.DataFrame([
        {"setting": "multiscale（无校验）", **m0},
        {"setting": f"multiscale + 边缘支持≥{t:.2f}", **m1},
    ])
    report.to_csv(METRICS / "m3_support_eval.csv", index=False, encoding="utf-8")

    print("\n[held-out] 结果")
    print(report.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n已写出 {METRICS.relative_to(ROOT)}/m3_support_sweep.csv 与 m3_support_eval.csv")


if __name__ == "__main__":
    main()
