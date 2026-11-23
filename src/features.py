"""特征提取：颜色、纹理（LBP / GLCM）、几何。

约定：所有特征在函数内部归一化，输出一维 float32 向量，
便于直接拼接后送入浅层分类器。
"""

from __future__ import annotations

import cv2
import numpy as np


def color_features(bgr: np.ndarray, bins: int = 32, space: str = "lab") -> np.ndarray:
    """三通道直方图（各 bins 个） + 一二阶矩（均值/标准差）。"""
    if space == "lab":
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    elif space == "hsv":
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    else:
        img = bgr

    feats = []
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [bins], [0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-8)
        feats.append(hist)
        feats.append(np.array([img[..., i].mean() / 255.0, img[..., i].std() / 255.0], np.float32))
    return np.concatenate(feats).astype(np.float32)


def lbp_features(bgr: np.ndarray, settings=((8, 1), (16, 2))) -> np.ndarray:
    """均匀模式 LBP 直方图，多尺度拼接。"""
    from skimage.feature import local_binary_pattern

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    feats = []
    for p, r in settings:
        lbp = local_binary_pattern(gray, p, r, method="uniform")
        hist, _ = np.histogram(lbp, bins=p + 2, range=(0, p + 2))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-8
        feats.append(hist)
    return np.concatenate(feats).astype(np.float32)


def glcm_features(bgr: np.ndarray, distances=(1, 2, 4), angles=(0, 45, 90, 135)) -> np.ndarray:
    """灰度共生矩阵的四个标量，方向取平均。"""
    from skimage.feature import graycomatrix, graycoprops

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    q = (gray // 8).astype(np.uint8)  # 量化到 32 级，控制矩阵大小
    rads = [int(a * np.pi / 180) for a in angles]
    m = graycomatrix(q, distances=list(distances), angles=rads, levels=32, symmetric=True, normed=True)
    out = []
    for prop in ("contrast", "correlation", "energy", "homogeneity"):
        v = graycoprops(m, prop)
        out.extend([float(v.mean()), float(v.std())])
    return np.asarray(out, dtype=np.float32)


def shape_features(quad: np.ndarray | None, image_shape: tuple[int, int]) -> np.ndarray:
    """四边形的几何描述：长宽比、面积占比、角度偏差。缺失时全 0。"""
    if quad is None:
        return np.zeros(4, dtype=np.float32)
    q = np.asarray(quad, np.float32)
    sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    long_side, short_side = max(sides), min(sides)
    aspect = long_side / max(short_side, 1e-6)
    area = abs(cv2.contourArea(q)) / (image_shape[0] * image_shape[1])
    # 四个内角与 90° 的平均偏差，衡量是否近似矩形
    dev = 0.0
    for i in range(4):
        a = q[i] - q[(i - 1) % 4]
        b = q[(i + 1) % 4] - q[i]
        cosv = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        dev += abs(np.degrees(np.arccos(np.clip(cosv, -1, 1))) - 90.0)
    return np.array([aspect / 3.0, area, dev / 4.0 / 90.0, 1.0], dtype=np.float32)
