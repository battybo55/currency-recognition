"""训练纸币面额分类器（当前用外部数据，等自采数据到位后换成人民币重训）。

用法：
    python scripts/train_banknote.py --per-class 40 --out models/demo_indian.joblib

产物：
    models/*.joblib          分类器（不入库，见 .gitignore）
    models/*.json            类别与特征说明
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classify import build_classifier  # noqa: E402
from src.features import color_features, lbp_features  # noqa: E402
from src.preprocess import resize_long_side  # noqa: E402

ZIP = ROOT / "data/raw/indian_currency.zip"
OUT_SIZE = (512, 256)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--out", default="models/demo_indian.joblib")
    args = ap.parse_args()

    z = zipfile.ZipFile(ZIP)
    groups: dict[str, list[str]] = {}
    for n in z.namelist():
        parts = n.split("/")
        if len(parts) == 4 and parts[1] == "Straight" and n.lower().endswith((".jpg", ".jpeg", ".png")):
            groups.setdefault(parts[2], []).append(n)

    X, y = [], []
    for cls in sorted(groups):
        for name in sorted(groups[cls])[: args.per_class]:
            img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(resize_long_side(img, 1000), OUT_SIZE, interpolation=cv2.INTER_AREA)
            X.append(np.concatenate([color_features(img), lbp_features(img)]))
            y.append(cls)

    X = np.stack(X)
    clf = build_classifier("lr")
    clf.fit(X, y)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out)
    meta = {
        "dataset": "indian_currency",
        "classes": list(clf.classes_),
        "feature": "Lab 3x32 hist + moments + LBP(8,1)(16,2)",
        "dim": int(X.shape[1]),
        "n_samples": len(y),
        "warning": "用外部数据训练的演示模型，不代表人民币性能",
    }
    (out.with_suffix(".json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"训练完成：{len(y)} 样本 / {len(clf.classes_)} 类，维度 {X.shape[1]}")
    print(f"已写出 {out.relative_to(ROOT)} 与同名 .json")


if __name__ == "__main__":
    main()
