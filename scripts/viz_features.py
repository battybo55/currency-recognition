"""特征可分性可视化：PCA 与 t-SNE。

用法：
    python scripts/viz_features.py --per-class 40

产物：
    results/figures/pca_tsne.png
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import color_features, lbp_features  # noqa: E402
from src.preprocess import resize_long_side  # noqa: E402

ZIP = ROOT / "data/raw/indian_currency.zip"
OUT = ROOT / "results/figures/pca_tsne.png"
OUT_SIZE = (512, 256)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=40)
    args = ap.parse_args()

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    z = zipfile.ZipFile(ZIP)
    groups: dict[str, list[str]] = {}
    for n in z.namelist():
        parts = n.split("/")
        if len(parts) == 4 and parts[1] == "Straight" and n.lower().endswith((".jpg", ".jpeg", ".png")):
            groups.setdefault(parts[2], []).append(n)

    feats, labels = [], []
    for cls in sorted(groups):
        for name in sorted(groups[cls])[: args.per_class]:
            img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = resize_long_side(img, 1000)
            small = cv2.resize(img, OUT_SIZE, interpolation=cv2.INTER_AREA)
            feats.append(np.concatenate([color_features(small), lbp_features(small)]))
            labels.append(cls)

    X = np.stack(feats)
    y = np.array(labels)
    classes = sorted(set(labels))

    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(X)
    ratio = pca.explained_variance_ratio_

    ts = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    Xt = ts.fit_transform(X)

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), layout="constrained")
    for i, cls in enumerate(classes):
        m = y == cls
        axes[0].scatter(Xp[m, 0], Xp[m, 1], s=16, alpha=0.85, color=cmap(i), label=cls)
        axes[1].scatter(Xt[m, 0], Xt[m, 1], s=16, alpha=0.85, color=cmap(i), label=cls)

    axes[0].set_title(f"PCA（累计解释方差 {ratio.sum():.1%}）")
    axes[1].set_title("t-SNE（perplexity=30）")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.15)
    axes[1].legend(title="面额", fontsize=8, title_fontsize=9, loc="best", frameon=False, ncol=1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"样本 {len(X)} 条，PCA 前两主成分解释方差 {ratio.sum():.1%}")
    print(f"已写出 {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
