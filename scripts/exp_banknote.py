"""E0 / E2 纸币对照实验：整图 vs 四角校正，颜色 vs 颜色+LBP。

用法：
    python scripts/exp_banknote.py --per-class 60 --rot-per-class 25

产物：
    results/metrics/e2_banknote.csv
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classify import build_classifier  # noqa: E402
from src.features import color_features, lbp_features  # noqa: E402
from src.preprocess import resize_long_side  # noqa: E402
from src.segment import find_banknote_quad_v2, warp_quad  # noqa: E402

ZIP = ROOT / "data/raw/indian_currency.zip"
OUT = ROOT / "results/metrics/e2_banknote.csv"
OUT_SIZE = (512, 256)


def is_train(name: str) -> bool:
    return int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16) % 10 < 7


def collect(z: zipfile.ZipFile, prefix: str, per_class: int) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for n in z.namelist():
        parts = n.split("/")
        if len(parts) == 4 and parts[1] == prefix and n.lower().endswith((".jpg", ".jpeg", ".png")):
            groups.setdefault(parts[2], []).append(n)
    return {k: sorted(v)[:per_class] for k, v in sorted(groups.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--rot-per-class", type=int, default=25)
    args = ap.parse_args()

    z = zipfile.ZipFile(ZIP)
    straight = collect(z, "Straight", args.per_class)
    rotated = collect(z, "Rotated", args.rot_per_class)
    classes = sorted(straight)
    print(f"类别 {len(classes)}：{classes}")
    print(f"Straight 每类 {args.per_class}，Rotated 每类 {args.rot_per_class}")

    rows, labels, names = [], [], []
    t0 = time.perf_counter()
    total = sum(len(v) for v in straight.values())

    for cls in classes:
        for name in straight[cls]:
            buf = np.frombuffer(z.read(name), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = resize_long_side(img, 1000)

            whole = cv2.resize(img, OUT_SIZE, interpolation=cv2.INTER_AREA)
            quad, method, ratio, found = find_banknote_quad_v2(img)
            warped = warp_quad(img, quad, OUT_SIZE) if found else whole

            rows.append({
                "image": name.split("/", 2)[2],
                "cls": cls,
                "split": "train" if is_train(name) else "test",
                "found": int(found),
                "method": method,
                "e0_color": color_features(whole),
                "e2_color": color_features(warped),
                "e0_color_lbp": np.concatenate([color_features(whole), lbp_features(whole)]),
                "e2_color_lbp": np.concatenate([color_features(warped), lbp_features(warped)]),
            })
        if len(rows) % 100 < args.per_class:
            print(f"  {len(rows)}/{total}  {time.perf_counter() - t0:.0f}s")

    # 旋转集只用于跨集测试
    rot_rows = []
    for cls in classes:
        for name in rotated[cls]:
            buf = np.frombuffer(z.read(name), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = resize_long_side(img, 1000)
            whole = cv2.resize(img, OUT_SIZE, interpolation=cv2.INTER_AREA)
            quad, method, ratio, found = find_banknote_quad_v2(img)
            warped = warp_quad(img, quad, OUT_SIZE) if found else whole
            rot_rows.append({
                "cls": cls,
                "e0_color": color_features(whole),
                "e2_color": color_features(warped),
                "e0_color_lbp": np.concatenate([color_features(whole), lbp_features(whole)]),
                "e2_color_lbp": np.concatenate([color_features(warped), lbp_features(warped)]),
                "found": int(found),
            })

    df = pd.DataFrame(rows)
    print(f"Straight {len(df)} 张，四角定位成功 {df.found.mean():.1%}；Rotated {len(rot_rows)} 张")

    results = []
    for feat in ("e0_color", "e0_color_lbp", "e2_color", "e2_color_lbp"):
        tr = df[df.split == "train"]
        te = df[df.split == "test"]
        Xtr = np.stack(tr[feat].values)
        Xte = np.stack(te[feat].values)
        ytr, yte = tr["cls"].values, te["cls"].values

        clf = build_classifier("svm")
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro")

        Xr = np.stack([r[feat] for r in rot_rows])
        yr = np.array([r["cls"] for r in rot_rows])
        acc_rot = accuracy_score(yr, clf.predict(Xr))

        results.append({
            "setting": feat.split("_")[0], "features": feat.split("_", 1)[1],
            "n_train": len(tr), "n_test": len(te),
            "acc": acc, "macro_f1": f1, "acc_rotated": acc_rot,
        })

        if feat == "e2_color_lbp":
            cm = confusion_matrix(yte, pred, labels=classes)
            off = [(classes[i], classes[j], int(cm[i, j]))
                   for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0]
            off.sort(key=lambda t: -t[2])
            print("  主要混淆对：", off[:5])

        print(f"  {feat:14s} acc={acc:.3f}  macroF1={f1:.3f}  rotated={acc_rot:.3f}")

    out = pd.DataFrame(results)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\n已写出 {OUT.relative_to(ROOT)}")
    print(f"总耗时 {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
