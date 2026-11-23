"""由同机位多帧构建背景板，并报告可信覆盖率。

用法：
    python scripts/make_background.py --pattern "data/raw/self/s1*.jpg" --out data/raw/_evidence/plate.png
    python scripts/make_background.py --pattern "data/raw/self/s0*.jpg" --min-coverage 0.7

输出：
    背景板 PNG + 覆盖率。覆盖率 = 时间标准差小于阈值的像素占比。
    <0.7 时提示"覆盖不足"，中值会保留目标残影，建议补拍空景或让目标铺满更多区域。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.background import build_plate  # noqa: E402
from src.io_utils import imread_unicode, imwrite_unicode  # noqa: E402
from src.preprocess import resize_long_side  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="data/raw/self/s1*.jpg")
    ap.add_argument("--out", default="data/raw/_evidence/plate.png")
    ap.add_argument("--long-side", type=int, default=1000)
    ap.add_argument("--min-coverage", type=float, default=0.70)
    args = ap.parse_args()

    files = sorted(ROOT.glob(args.pattern))
    if len(files) < 3:
        raise SystemExit(f"帧数不足（{len(files)} 张），至少 3 张同机位照片")

    frames = []
    for p in files:
        img = imread_unicode(p)
        if img is None:
            continue
        frames.append(resize_long_side(img, args.long_side))
    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        raise SystemExit(f"帧尺寸不一致：{shapes}，请先统一分辨率")

    plate, coverage, std_map = build_plate(frames)
    out = ROOT / args.out
    imwrite_unicode(out, plate)
    print(f"用了 {len(frames)} 帧，尺寸 {frames[0].shape[1]}×{frames[0].shape[0]}")
    print(f"背景板可信覆盖率 {coverage:.1%}（std<12 的像素占比）")
    if coverage < args.min_coverage:
        print("警告：覆盖不足。中值背景会保留目标残影。")
        print("     补救：① 每个场次补一张空景照；② 或让目标在画面中分散到更多区域，")
        print("     保证每块区域至少有一半的帧是空的。")
    imwrite_unicode(out.with_name(out.stem + "_std.png"),
                    cv2.applyColorMap(cv2.normalize(std_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                                      cv2.COLORMAP_JET))
    print(f"已写出 {out.relative_to(ROOT)} 与 {out.stem}_std.png")


if __name__ == "__main__":
    main()
