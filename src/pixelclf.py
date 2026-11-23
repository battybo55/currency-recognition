"""像素级前景分类器：用局部颜色 + 纹理特征，替代单一色差判据。

动机：
    木纹与纸币/硬币的颜色可能很接近，但**纹理差别明显**——木纹高频、纸面平缓。
    单一 Lab 色差无法利用这一点，像素级分类器可以。

特征（每像素 9 维，全部局部、不用绝对坐标，便于跨图泛化）：
    L, a, b                 —— 当前像素颜色
    局部均值 L/a/b（15×15）  —— 所在区域的平均色
    局部标准差（15×15）      —— 纹理粗糙度，木纹高、纸面低
    梯度幅值（Sobel）        —— 边缘强度
    到背景色距离             —— 与背景模型的色差
"""

from __future__ import annotations

import cv2
import numpy as np

from .background import BackgroundModel, detect_background

FEATURE_NAMES = [
    "L", "a", "b",
    "mean_L", "mean_a", "mean_b",
    "local_std", "grad", "dist_bg",
]


def extract_features(bgr: np.ndarray, model: BackgroundModel | None = None, ksize: int = 15) -> np.ndarray:
    """返回 (H, W, 9) 的 float32 特征图。"""
    model = model or detect_background(bgr)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    mean = cv2.blur(lab, (ksize, ksize))
    sq = cv2.blur(lab * lab, (ksize, ksize))
    var = np.clip(sq - mean * mean, 0, None)
    local_std = np.sqrt(var).mean(axis=2)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)

    d_bg = np.min(np.linalg.norm(lab[:, :, None, :] - model.colors[None, None, :, :], axis=3), axis=2)

    feats = np.dstack([
        lab[..., 0], lab[..., 1] - 128.0, lab[..., 2] - 128.0,
        mean[..., 0], mean[..., 1] - 128.0, mean[..., 2] - 128.0,
        local_std, grad, d_bg,
    ]).astype(np.float32)
    return feats


def collect_samples(
    bgr: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    model: BackgroundModel | None = None,
    max_each: int = 6000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """按掩膜采样像素特征，返回 (X, y)。"""
    feats = extract_features(bgr, model)
    rng = np.random.default_rng(seed)

    def take(mask: np.ndarray, label: int) -> tuple[np.ndarray, np.ndarray]:
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return np.empty((0, feats.shape[2]), np.float32), np.empty((0,), np.int8)
        idx = rng.choice(len(ys), size=min(max_each, len(ys)), replace=False)
        return feats[ys[idx], xs[idx]], np.full(len(idx), label, np.int8)

    Xp, yp = take(positives, 1)
    Xn, yn = take(negatives, 0)
    return np.vstack([Xp, Xn]), np.concatenate([yp, yn])


def predict_map(model_rf, bgr: np.ndarray, bg_model: BackgroundModel | None = None) -> np.ndarray:
    """输出 0–255 的前景概率图。"""
    feats = extract_features(bgr, bg_model)
    h, w, f = feats.shape
    proba = model_rf.predict_proba(feats.reshape(-1, f))[:, 1]
    return (proba.reshape(h, w) * 255).astype(np.uint8)
