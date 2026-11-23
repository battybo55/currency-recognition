"""预处理：缩放、色彩空间、白平衡、去噪、光照校正。

约定：所有以像素为单位的参数都由图像短边推导，配置里只写比例。
"""

from __future__ import annotations

import cv2
import numpy as np


def resize_long_side(image: np.ndarray, long_side: int = 1000) -> np.ndarray:
    """按长边等比缩放；只缩小不放大。"""
    h, w = image.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1.0:
        return image.copy()
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray, method: str = "median", ksize: int = 5) -> np.ndarray:
    """ksize 会强制为奇数。"""
    k = ksize if ksize % 2 == 1 else ksize + 1
    if method == "median":
        return cv2.medianBlur(image, k)
    if method == "gaussian":
        return cv2.GaussianBlur(image, (k, k), 0)
    if method == "bilateral":
        return cv2.bilateralFilter(image, k, 75, 75)
    return image


def gray_world(image: np.ndarray) -> np.ndarray:
    """灰度世界白平衡：按通道均值归一化。"""
    if image.ndim != 3:
        return image
    img = image.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    means[means == 0] = 1.0
    gain = means.mean() / means
    out = np.clip(img * gain, 0, 255).astype(np.uint8)
    return out


def clahe_lab(image: np.ndarray, clip_limit: float = 2.0, tile_grid=(8, 8)) -> np.ndarray:
    """仅对 Lab 的 L 通道做 CLAHE，避免色偏。"""
    if image.ndim == 2:
        return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid)).apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def flatten_illumination(gray: np.ndarray, ksize_ratio: float = 0.1) -> np.ndarray:
    """用大核闭运算估计背景照度并做除法归一化。"""
    h, w = gray.shape[:2]
    k = max(3, int(min(h, w) * ksize_ratio))
    k = k if k % 2 == 1 else k + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    bg = np.where(bg == 0, 1, bg).astype(np.float32)
    out = np.clip(gray.astype(np.float32) / bg * 200.0, 0, 255)
    return out.astype(np.uint8)
