"""按参考文章实现的硬币分割计数（公众号「OpenCV与AI深度学习」）。

原文骨架：
    pyrMeanShiftFiltering(21,51) → 灰度 + Otsu → 距离变换
    → peak_local_max(min_distance=10) → 分水岭 → 计数

与原文的唯一差别：先做**极性校正**。原文假设「亮目标、暗背景」，
本项目的硬币是「暗目标、亮背景」，Otsu 会得到反相结果，距离变换与
分水岭都依赖目标为前景(255)，所以先判断极性再取反。

用法：
    python scripts/coin_count_article.py --glob "data/raw/self/coins/*.jpg"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import imread_unicode, imwrite_unicode  # noqa: E402
from src.preprocess import resize_long_side  # noqa: E402


def coin_count_article(img: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """返回 (计数, 标注图, 二值图)。"""
    shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 极性校正：让目标为 255。若 Otsu 的前景(255)比背景更暗，说明前景是目标却被打成暗，取反。
    fg_mean = gray[thresh == 255].mean() if (thresh == 255).any() else 0.0
    bg_mean = gray[thresh == 0].mean() if (thresh == 0).any() else 255.0
    if fg_mean < bg_mean:
        thresh = 255 - thresh

    # 距离变换 + 峰值
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    coords = peak_local_max(dist, min_distance=10, exclude_border=False, labels=thresh.astype(bool))
    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i
    labels = watershed(-dist, markers, mask=thresh)

    count = int(labels.max())
    vis = img.copy()
    for lab in range(1, int(labels.max()) + 1):
        m = (labels == lab).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)
        cv2.circle(vis, (int(x), int(y)), int(r), (0, 255, 0), 2)
    return count, vis, thresh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/raw/self/coins/*.jpg")
    ap.add_argument("--save", default="data/raw/_evidence/coin_article")
    args = ap.parse_args()

    GT = {"c01": 6, "c02": 5, "c03": 4, "c04": 3, "c05": 2, "c06": 1}
    out_dir = ROOT / args.save
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(ROOT.glob(args.glob))
    ok = 0
    print(f"{'图':<8}{'计数':>6}{'真值':>6}")
    for p in files:
        img = resize_long_side(imread_unicode(p), 1000)
        cnt, vis, thresh = coin_count_article(img)
        gt = GT.get(p.stem, None)
        if gt is not None:
            ok += cnt == gt
        print(f"{p.stem:<8}{cnt:>6}{gt if gt is not None else '-':>6}")
        imwrite_unicode(out_dir / f"{p.stem}_vis.png", np.hstack([cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), vis]))

    if GT:
        print(f"\n完全正确 {ok}/{len([f for f in files if f.stem in GT])}")


if __name__ == "__main__":
    main()
