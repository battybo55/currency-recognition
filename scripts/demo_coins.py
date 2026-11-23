"""硬币计数演示：六张纯色背景硬币图，霍夫圆计数，拼图展示。

用法：
    python scripts/demo_coins.py
输出：
    results/figures/demo_coins.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import imread_unicode  # noqa: E402
from src.preprocess import resize_long_side  # noqa: E402
from src.segment import detect_coins_multiscale  # noqa: E402

FIG = ROOT / "results" / "figures"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

GT = {"c01": 6, "c02": 5, "c03": 4, "c04": 3, "c05": 2, "c06": 1}


def main() -> None:
    names = sorted(GT)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    ok = 0
    for ax, name in zip(axes.ravel(), names):
        img = resize_long_side(imread_unicode(ROOT / f"data/raw/self/coins/{name}.jpg"), 1000)
        circles = detect_coins_multiscale(img, param2=45, min_support=0.45)
        vis = img.copy()
        for (x, y, r) in circles:
            cv2.circle(vis, (int(x), int(y)), int(r), (0, 200, 0), 3)
        hit = len(circles) == GT[name]
        ok += hit
        disp = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        ax.imshow(disp)
        ax.set_title(f"{name}：计数 {len(circles)} / 真值 {GT[name]}",
                     color="green" if hit else "red", fontsize=12)
        ax.axis("off")

    fig.suptitle(f"硬币计数演示（霍夫圆 param2=45）：{ok}/6 正确", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "demo_coins.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"硬币计数：{ok}/6 正确 → {FIG / 'demo_coins.png'}")


if __name__ == "__main__":
    main()
