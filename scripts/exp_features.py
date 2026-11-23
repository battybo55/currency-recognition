"""E4 特征消融 + E5 分类器对比（纸币，Indian Currency）。

用法：
    python scripts/exp_features.py --per-class 40

产物：
    results/metrics/e4_features.csv
    results/metrics/e5_classifiers.csv
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classify import build_classifier  # noqa: E402
from src.features import (  # noqa: E402
    color_features,
    frequency_features,
    gabor_features,
    glcm_features,
    lbp_features,
)
from src.preprocess import resize_long_side  # noqa: E402

ZIP = ROOT / "data/raw/indian_currency.zip"
OUT_SIZE = (512, 256)
FAMILIES = ["color", "lbp", "glcm", "gabor", "freq"]


def is_train(name: str) -> bool:
    return int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16) % 10 < 7


def encode(img: np.ndarray) -> dict[str, np.ndarray]:
    small = cv2.resize(img, OUT_SIZE, interpolation=cv2.INTER_AREA)
    return {
        "color": color_features(small),
        "lbp": lbp_features(small),
        "glcm": glcm_features(small),
        "gabor": gabor_features(small),
        "freq": frequency_features(small),
    }


def evaluate(Xtr, ytr, Xte, yte, clf_name="svm") -> tuple[float, float]:
    clf = build_classifier(clf_name)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return accuracy_score(yte, pred), f1_score(yte, pred, average="macro")


def cross_validate(X, y, clf_name="svm", folds: int = 5, seed: int = 42):
    """分层 K 折，返回 accuracy 与 macro-F1 的均值/标准差。"""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for tr_idx, te_idx in skf.split(X, y):
        clf = build_classifier(clf_name)
        clf.fit(X[tr_idx], y[tr_idx])
        pred = clf.predict(X[te_idx])
        accs.append(accuracy_score(y[te_idx], pred))
        f1s.append(f1_score(y[te_idx], pred, average="macro"))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)), float(np.std(f1s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=40)
    args = ap.parse_args()

    z = zipfile.ZipFile(ZIP)
    groups: dict[str, list[str]] = {}
    for n in z.namelist():
        parts = n.split("/")
        if len(parts) == 4 and parts[1] == "Straight" and n.lower().endswith((".jpg", ".jpeg", ".png")):
            groups.setdefault(parts[2], []).append(n)

    rows = []
    t0 = time.perf_counter()
    for cls in sorted(groups):
        for name in sorted(groups[cls])[: args.per_class]:
            buf = np.frombuffer(z.read(name), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = resize_long_side(img, 1000)
            rec = {"cls": cls, "split": "train" if is_train(name) else "test"}
            rec.update(encode(img))
            rows.append(rec)
    df = pd.DataFrame(rows)
    print(f"样本 {len(df)} 张（{len(groups)} 类 × {args.per_class}），特征提取 {time.perf_counter() - t0:.1f}s")

    y = df["cls"].values

    def stack(fams):
        return np.hstack([np.stack(df[f].values) for f in fams])

    # E4：单族 + 累加，5 折交叉验证
    e4 = []
    for fam in FAMILIES:
        X = stack([fam])
        acc, acc_sd, f1, f1_sd = cross_validate(X, y)
        e4.append({"mode": "single", "features": fam, "dim": int(X.shape[1]),
                   "acc": acc, "acc_sd": acc_sd, "macro_f1": f1, "f1_sd": f1_sd})

    cumulative: list[str] = []
    for fam in FAMILIES:
        cumulative.append(fam)
        X = stack(cumulative)
        acc, acc_sd, f1, f1_sd = cross_validate(X, y)
        e4.append({"mode": "cumulative", "features": "+".join(cumulative), "dim": int(X.shape[1]),
                   "acc": acc, "acc_sd": acc_sd, "macro_f1": f1, "f1_sd": f1_sd})

    print("\n[E4] 特征消融（5 折）")
    print(pd.DataFrame(e4)[["mode", "features", "dim", "acc", "acc_sd", "macro_f1"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # E5：分类器对比（全部特征），同样 5 折
    X = stack(FAMILIES)
    e5 = []
    for name in ("svm", "rf", "knn", "lr"):
        acc, acc_sd, f1, f1_sd = cross_validate(X, y, name)
        e5.append({"classifier": name, "dim": int(X.shape[1]),
                   "acc": acc, "acc_sd": acc_sd, "macro_f1": f1, "f1_sd": f1_sd})
        print(f"  {name:4s} acc={acc:.3f}±{acc_sd:.3f}  macroF1={f1:.3f}")

    METRICS = ROOT / "results/metrics"
    METRICS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(e4).to_csv(METRICS / "e4_features.csv", index=False, encoding="utf-8")
    pd.DataFrame(e5).to_csv(METRICS / "e5_classifiers.csv", index=False, encoding="utf-8")
    print(f"\n已写出 {METRICS.relative_to(ROOT)}/e4_features.csv 与 e5_classifiers.csv")


if __name__ == "__main__":
    main()
