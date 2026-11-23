"""实拍图回归测试：在 data/raw/self 上跑端到端，统计定位与计数是否正确。

用法：
    python scripts/test_self.py                 # 不带模型，只测定位与计数
    python scripts/test_self.py --model models/demo_indian.joblib

真值写在 data/raw/self/labels.csv（image,notes,coins），不入库。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402

from src.io_utils import imwrite_unicode  # noqa: E402
from src.pipeline import analyze, load_config  # noqa: E402

SELF = ROOT / "data/raw/self"
EVID = ROOT / "data/raw/_evidence/self"
LABELS = SELF / "labels.csv"


def read_labels() -> dict[str, tuple[int, int]]:
    if not LABELS.exists():
        return {}
    out = {}
    with open(LABELS, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["image"]] = (int(row["notes"]), int(row["coins"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    args = ap.parse_args()

    model = joblib.load(ROOT / args.model) if args.model else None
    cfg = load_config()
    truth = read_labels()
    images = sorted(p for p in SELF.glob("*.jpg"))

    print(f"{'图像':<10}{'检出纸币':>9}{'真值':>6}{'检出硬币':>9}{'真值':>6}{'耗时ms':>9}  面额输出")
    print("-" * 78)
    hit_note = hit_coin = 0
    for p in images:
        result, overlay = analyze(p, cfg, banknote_model=model, draw=True)
        imwrite_unicode(EVID / f"{p.stem}_pred.png", overlay)
        nb, nc = len(result["banknotes"]), len(result["coins"])
        gn, gc = truth.get(p.name, (-1, -1))
        hit_note += int(nb == gn)
        hit_coin += int(nc == gc)
        labels = [b["label"] or "-" for b in result["banknotes"]]
        print(f"{p.name:<10}{nb:>9}{gn if gn >= 0 else '-':>6}{nc:>9}{gc if gc >= 0 else '-':>6}"
              f"{result['debug']['elapsed_ms']:>9.0f}  {','.join(labels)}")

    if truth:
        n = len(images)
        print("-" * 78)
        print(f"纸币张数正确 {hit_note}/{n}，硬币枚数正确 {hit_coin}/{n}")
    print(f"标注图：{EVID.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
