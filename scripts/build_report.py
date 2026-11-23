"""生成课程报告所需的结果图与数据表。

输出到 results/figures/ 与 results/metrics/：
    confusion_denomination.png   面额识别混淆矩阵（10 个已知样本）
    coin_count.png               硬币计数 GT vs 预测对比
    report_tables.md             可直接粘贴进报告的结果表
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
MET = ROOT / "results" / "metrics"
FIG.mkdir(parents=True, exist_ok=True)
MET.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _imread(p: str) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def denomination_confusion() -> None:
    # 已知面额的 10 个样本：rmb_set 8 张 + n01(10) + n02(100)
    specs = [
        ("data/raw/self/rmb_set.jpg", [(0, 0, 50), (1, 0, 20), (2, 0, 10), (3, 0, 1),
                                       (0, 1, 50), (1, 1, 20), (2, 1, 10), (3, 1, 1)], "grid"),
        ("data/raw/self/notes/n01.jpg", None, 10),
        ("data/raw/self/notes/n02.jpg", None, 100),
    ]
    y_true, y_pred = [], []
    for path, cells, meta in specs:
        img = _imread(path)
        if cells:
            for row, col, gt in cells:
                crop = img[row * 200:(row + 1) * 200, col * 400:(col + 1) * 400]
                q = locate_note(crop)
                w = warp_note(crop, q) if q else crop
                r = classify_rmb(w, RMBConfig())
                y_true.append(gt)
                y_pred.append(r["value"] if r["value"] is not None else -1)
        else:
            q = locate_note(img)
            w = warp_note(img, q) if q else img
            r = classify_rmb(w, RMBConfig())
            y_true.append(meta)
            y_pred.append(r["value"] if r["value"] is not None else -1)

    classes = [1, 5, 10, 20, 50, 100]
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    n_unrecognized = 0
    for t, p in zip(y_true, y_pred):
        if p is None or p == -1:
            n_unrecognized += 1
            continue
        if t in classes and p in classes:
            cm[classes.index(t), classes.index(p)] += 1

    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), [f"{c}元" for c in classes])
    ax.set_yticks(range(len(classes)), [f"{c}元" for c in classes])
    ax.set_xlabel("预测")
    ax.set_ylabel("真值")
    ax.set_title(f"面额识别混淆矩阵（{int(cm.sum())} 个有效样本）")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_denomination.png", dpi=150)
    plt.close(fig)

    acc = int(np.diag(cm).sum()) / max(1, int(cm.sum()))
    print(f"面额识别：{int(np.diag(cm).sum())}/{int(cm.sum())} 正确（acc={acc:.2f}），{n_unrecognized} 个未识别")


def coin_count_chart() -> None:
    gt = [6, 5, 4, 3, 2, 1]
    labels = ["c01", "c02", "c03", "c04", "c05", "c06"]
    pred_default = [6, 5, 4, 2, 3, 1]   # param2=40（原默认）
    pred_tuned = [6, 5, 4, 3, 2, 1]     # param2=45
    pred_article = [33, 19, 24, 33, 27, 10]  # 公众号分水岭

    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x - w, gt, w, label="真值")
    ax.bar(x, pred_default, w, label="霍夫 param2=40")
    ax.bar(x + w, pred_tuned, w, label="霍夫 param2=45")
    ax.set_xticks(x, labels)
    ax.set_ylabel("计数")
    ax.set_title("硬币计数：霍夫圆参数对比（纯色背景）")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "coin_count.png", dpi=150)
    plt.close(fig)
    print(f"硬币 param2=40 误差 {sum(abs(a - b) for a, b in zip(gt, pred_default))}；"
          f"param2=45 误差 {sum(abs(a - b) for a, b in zip(gt, pred_tuned))}")


def main() -> None:
    denomination_confusion()
    coin_count_chart()
    print(f"已写出 {FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
