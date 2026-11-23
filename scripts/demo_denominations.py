"""六面额识别演示：生成一张「六个面额全部识别成功」的拼图。

用法：
    python scripts/demo_denominations.py
输出：
    results/figures/demo_six_denominations.png

用于答辩现场：一张图展示 1/5/10/20/50/100 元各被正确识别（含置信度）。
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

from src.rmb import RMBConfig, classify_rmb  # noqa: E402
from src.single_note import locate_note, warp_note  # noqa: E402

FIG = ROOT / "results" / "figures"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _im(p: str) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def sample(name: str, gt: int):
    """返回 (图像, 面额) 的干净样本。"""
    if name == "100":
        img = _im("data/raw/self/rmb_ref.jpg")
        return img[0:55, 0:134], gt            # 渲染 100 元正面
    if name == "5":
        img = _im("data/raw/self/yuan5.png")
        return img[int(img.shape[0] * 0.03):int(img.shape[0] * 0.48), :].copy(), gt
    # 1/10/20/50 用 rmb_set 实拍（左列 1999 版）
    img = _im("data/raw/self/rmb_set.jpg")
    row = {1: 3, 10: 2, 20: 1, 50: 0}[gt]
    return img[row * 200:(row + 1) * 200, 0:400], gt


def main() -> None:
    order = [1, 5, 10, 20, 50, 100]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    ok = 0
    for ax, gt in zip(axes.ravel(), order):
        img, truth = sample(str(gt), gt)
        q = locate_note(img)
        warped = warp_note(img, q) if q else img
        r = classify_rmb(warped, RMBConfig())
        hit = r["value"] == truth
        ok += hit

        disp = cv2.cvtColor(warped if q else img, cv2.COLOR_BGR2RGB)
        ax.imshow(disp)
        label = f"{r['label']}" if r["label"] else "未识别"
        color = "green" if hit else "red"
        ax.set_title(f"真值 {truth}元 → 识别 {label}  ({r['confidence']:.2f})",
                     color=color, fontsize=11)
        ax.axis("off")

    fig.suptitle(f"六面额识别演示：{ok}/6 正确", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "demo_six_denominations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"六面额演示：{ok}/6 正确 → {FIG / 'demo_six_denominations.png'}")


if __name__ == "__main__":
    main()
