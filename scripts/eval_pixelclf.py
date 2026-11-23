"""用人工标注评估像素级前景分类器（5 折，按图分组）。

指标：
    IoU / F1  —— 预测掩膜与人工掩膜的像素级重合
    实例数误差 —— 连通域个数与人工多边形个数之差
并同时给出「现有颜色模型」在同一批图上的 IoU，作为基线对照。

用法：
    python scripts/eval_pixelclf.py --folds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.background import detect_background, raw_residual  # noqa: E402
from src.io_utils import imread_unicode, imwrite_unicode  # noqa: E402
from src.pixelclf import collect_samples, predict_map  # noqa: E402
from src.preprocess import resize_long_side  # noqa: E402
from src.scene import SceneConfig, extract_objects  # noqa: E402

SELF = ROOT / "data/raw/self"
MASK_DIR = ROOT / "data/raw/annotations/masks"
LABELS = ROOT / "data/raw/annotations/labels.json"


def load_pair(stem: str, long_side: int = 1000):
    img = resize_long_side(imread_unicode(SELF / f"{stem}.jpg"), long_side)
    h, w = img.shape[:2]
    m = cv2.imdecode(np.fromfile(str(MASK_DIR / f"{stem}.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None, None
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return img, (m > 127).astype(np.uint8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union else 1.0


def split_instances(mask: np.ndarray, min_distance_ratio: float = 0.05,
                    threshold_rel: float = 0.08) -> np.ndarray:
    """距离变换 + 峰值分水岭，把粘连的目标拆开。

    只对「厚度足够」的区域拆：硬币与纸币的尺度差很大，用相对短边的
    最小间距只能取一个折中值，参数在 15 张标注图上选定（小样本，需谨慎外推）。
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    if mask.sum() == 0:
        return mask
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if float(dist.max()) < 3:
        return mask
    md = max(3, int(min(mask.shape[:2]) * min_distance_ratio))
    coords = peak_local_max(dist, min_distance=md, threshold_rel=threshold_rel, exclude_border=False)
    if len(coords) <= 1:
        return mask
    markers = np.zeros_like(mask, dtype=np.int32)
    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i
    labels = watershed(-dist, markers, mask=mask)
    out = np.zeros_like(mask)
    for lab in range(1, int(labels.max()) + 1):
        part = (labels == lab).astype(np.uint8)
        if part.sum() / mask.size > 0.0015:
            out = np.maximum(out, part)
    return out


def color_split(mask: np.ndarray, img: np.ndarray, min_each: float = 0.22,
                min_de: float = 14.0, max_area: float = 0.45) -> np.ndarray:
    """对「面积偏大」的连通域做 Lab 二聚类，色差显著时按主色切开。

    针对「10 元叠 100 元」这类跨面额粘连——它们没有几何颈部，距离变换分不开，
    但主色差异很大。同面额叠放仍然无法拆开，这是方法的边界。
    """
    from sklearn.cluster import KMeans

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        comp = labels == i
        area = int(comp.sum())
        if area / mask.size < 0.002:
            continue
        if area / mask.size > max_area:
            out = np.maximum(out, comp.astype(np.uint8))
            continue
        ys, xs = np.nonzero(comp)
        pts = lab[ys, xs]
        if len(pts) < 200:
            out[ys, xs] = 1
            continue
        km = KMeans(n_clusters=2, n_init=5, random_state=0).fit(pts)
        de = float(np.linalg.norm(km.cluster_centers_[0] - km.cluster_centers_[1]))
        parts = []
        for c in range(2):
            sel = km.labels_ == c
            m = np.zeros_like(mask)
            m[ys[sel], xs[sel]] = 1
            parts.append(m)
        if de >= min_de and min(p.sum() for p in parts) >= min_each * area:
            out = np.maximum(out, parts[0])
            out = np.maximum(out, parts[1])
        else:
            out[ys, xs] = 1
    return out


def postprocess(prob: np.ndarray, thr: float = 0.5, close: int = 5, split: bool = False,
                min_dist: float = 0.05, img: np.ndarray | None = None,
                csplit: bool = False) -> np.ndarray:
    m = (prob > thr * 255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if close > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    if split:
        m = split_instances(m, min_dist)
    if csplit and img is not None:
        m = color_split(m, img)
    return m


def count_instances(mask: np.ndarray, min_area: float = 0.002) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    return int(sum(1 for i in range(1, n) if stats[i, 4] / mask.size > min_area))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-each", type=int, default=6000)
    ap.add_argument("--split", action="store_true", help="对粘连区域做分水岭拆分")
    ap.add_argument("--min-dist", type=float, default=0.05)
    ap.add_argument("--color-split", action="store_true", help="按 Lab 主色拆分大团块")
    args = ap.parse_args()

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    stems = sorted(Path(k).stem for k in labels)
    print(f"{len(stems)} 张标注图，{args.folds} 折")

    data = {}
    for s in stems:
        img, gt = load_pair(s)
        if img is None:
            print(f"  跳过 {s}：缺少图像或掩膜")
            continue
        data[s] = (img, gt)
    stems = [s for s in stems if s in data]

    rows = []
    for k in range(args.folds):
        test = [s for i, s in enumerate(stems) if i % args.folds == k]
        train = [s for s in stems if s not in test]
        Xs, ys = [], []
        for s in train:
            img, gt = data[s]
            pos = cv2.erode(gt, np.ones((5, 5), np.uint8))
            far = cv2.dilate(gt, np.ones((31, 31), np.uint8)) == 0
            model = detect_background(img)
            neg = (far & (raw_residual(img, model) < 15)).astype(np.uint8)
            X, y = collect_samples(img, pos, neg, model, max_each=args.max_each, seed=k)
            Xs.append(X)
            ys.append(y)
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        clf = RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=20,
                                     n_jobs=-1, random_state=42, class_weight="balanced")
        clf.fit(X, y)

        for s in test:
            img, gt = data[s]
            prob = predict_map(clf, img, detect_background(img))
            pred = postprocess(prob, split=args.split, min_dist=args.min_dist,
                               img=img, csplit=args.color_split)
            base_objs, base_mask, _ = extract_objects(img, SceneConfig())
            base = (base_mask > 127).astype(np.uint8)

            rows.append({
                "image": s,
                "iou_clf": iou(pred, gt),
                "iou_base": iou(base, gt),
                "inst_gt": len(labels[f"{s}.jpg"]),
                "inst_clf": count_instances(pred),
                "inst_base": len(base_objs),
            })
        print(f"  fold {k + 1}/{args.folds} 完成（测试 {len(test)} 张）")

    import pandas as pd
    df = pd.DataFrame(rows).sort_values("image")
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n平均 IoU：像素分类器 {df.iou_clf.mean():.3f}，现有颜色模型 {df.iou_base.mean():.3f}")
    print(f"实例数平均绝对误差：分类器 {np.abs(df.inst_clf - df.inst_gt).mean():.2f}，"
          f"颜色模型 {np.abs(df.inst_base - df.inst_gt).mean():.2f}")

    out = ROOT / "results/metrics/eval_pixelclf.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"已写出 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
