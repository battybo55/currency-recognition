"""单图分阶段诊断：原图 / 二值 / 距离变换 / 霍夫 / 分水岭。

用法：
    python scripts/diagnose_v3.py --image china_coins/800af58787.jpg
    python scripts/diagnose_v3.py --image all_coins/26fb204f51.jpg --out my.png
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

from src.io_utils import imread_unicode, imwrite_unicode  # noqa: E402
from src.preprocess import resize_long_side, to_gray  # noqa: E402
from src.segment import (  # noqa: E402
    binary_foreground,
    detect_coins_hough,
    detect_coins_watershed,
)

COIN_ROOT = ROOT / "data/raw/count_coins/coins_images/coins_images"
EVID = ROOT / "data/raw/_evidence/v3"


def tile(title: str, img: np.ndarray, h: int = 240) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    scale = h / img.shape[0]
    img = cv2.resize(img, (max(1, int(img.shape[1] * scale)), h))
    canvas = np.full((h + 28, img.shape[1], 3), 255, np.uint8)
    canvas[28:, :] = img
    cv2.putText(canvas, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="形如 china_coins/xxx.jpg")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/baseline.yaml", encoding="utf-8"))
    img = imread_unicode(COIN_ROOT / args.image)
    if img is None:
        raise SystemExit("读不到图像")
    img = resize_long_side(img, cfg["image"]["long_side"])

    gray = cv2.medianBlur(to_gray(img), 5)
    b = binary_foreground(gray)
    b_open = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dist = cv2.distanceTransform(b_open, cv2.DIST_L2, 5)
    dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist_vis = cv2.applyColorMap(dist_vis, cv2.COLORMAP_JET)

    c = cfg["coin"]
    hough = detect_coins_hough(
        img,
        radius_ratio=tuple(c["radius_ratio"]),
        dp=c["hough"]["dp"],
        min_dist_ratio=c["hough"]["min_dist_ratio"],
        param1=c["hough"]["param1"],
        param2=c["hough"]["param2"],
    )
    ws = detect_coins_watershed(
        img,
        min_distance_ratio=c["watershed"]["min_distance_ratio"],
        min_area_ratio=c["watershed"]["min_area_ratio"],
    )

    ov_h = img.copy()
    for x, y, r in hough:
        cv2.circle(ov_h, (int(x), int(y)), int(r), (0, 255, 0), 2)
    ov_w = img.copy()
    for x, y, r in ws:
        cv2.circle(ov_w, (int(x), int(y)), max(2, int(r)), (0, 200, 255), 2)

    row1 = np.hstack([tile("input", img), tile("binary", b), tile("distance", dist_vis)])
    row2 = np.hstack([tile(f"hough n={len(hough)}", ov_h), tile(f"watershed n={len(ws)}", ov_w), tile("mask open", b_open)])
    sheet = np.vstack([row1, row2])

    out = Path(args.out) if args.out else EVID / (Path(args.image).stem + "_diag.png")
    imwrite_unicode(out, sheet)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
