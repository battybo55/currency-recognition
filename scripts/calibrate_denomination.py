"""从带标签的文件夹标定面额原型。

目录结构：
    <folder>/<面额>/<图片>      例：data/raw/_calib/10/xx.jpg

用法：
    python scripts/calibrate_denomination.py --folder data/raw/_calib_indian \
        --out configs/denomination_indian.yaml

输出 yaml 记录每类的 Lab 色度中位数与样本数，供 src/denomination.py 使用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.denomination import _mean_chroma  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out", default="configs/denomination.yaml")
    ap.add_argument("--note-tol", type=float, default=22.0)
    args = ap.parse_args()

    folder = ROOT / args.folder
    if not folder.is_dir():
        raise SystemExit(f"目录不存在：{folder}")

    notes = []
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        files = [q for q in sub.iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png")]
        chroma = []
        for q in files:
            img = cv2.imdecode(np.fromfile(str(q), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
            chroma.append(_mean_chroma(img, inset=0.10))
        if len(chroma) < 5:
            print(f"  跳过 {sub.name}：样本 {len(chroma)} 张，少于 5")
            continue
        arr = np.asarray(chroma)
        a, b = float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
        spread = float(np.median(np.hypot(arr[:, 0] - a, arr[:, 1] - b)))
        # 类别名允许 500_old 这类后缀：取数值前缀作为面额，保留原名便于追溯
        try:
            value = float(sub.name)
        except ValueError:
            value = float(sub.name.split("_")[0])
        notes.append({"value": value, "name": sub.name,
                      "a": round(a, 1), "b": round(b, 1), "spread": round(spread, 1),
                      "n": len(chroma)})
        print(f"  {sub.name:>8s}: a={a:6.1f} b={b:6.1f} spread={spread:5.1f} n={len(chroma)}")

    if not notes:
        raise SystemExit("没有可用的类别，检查目录结构是否为 <folder>/<面额>/<图片>")

    data = {
        "calibrated": True,
        "note_tol": args.note_tol,
        "min_confidence": 0.45,
        "notes": notes,
        "coin_gold_hue": [12, 38],
        "coin_gold_sat": 70,
        "source": str(folder.relative_to(ROOT)),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"\n已写出 {out.relative_to(ROOT)}，{len(notes)} 类")


if __name__ == "__main__":
    main()
