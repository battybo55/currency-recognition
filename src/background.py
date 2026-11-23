"""背景色检测与剔除。

流程：
    1. 对降采样像素做 Lab 空间 k-means（k=3），得到候选色簇；
    2. 用**边框占比**判定哪些簇属于背景——背景通常占据画面外圈，
       这一步比"取最大簇"稳，因为目标可能占掉大半画面；
    3. 若没有簇通过边框判据，退回最大簇，保证总有可用背景色；
    4. 剔除时对每个像素取到最近背景色的距离，并按**分块噪声**归一化，
       从而兼容背景色不均与光照渐变：局部噪声大的区域要求更大的色差才算前景。

输出带有背景类型（uniform / textured / gradient）与均匀度，供上层决定策略。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class BackgroundModel:
    colors: np.ndarray                 # (k,3) Lab 背景色
    weights: np.ndarray                # (k,) 各色占比
    kind: str = "uniform"              # uniform | textured | gradient
    uniformity: float = 0.0            # 0–1，越高越均匀
    tile_sigma: np.ndarray | None = None   # (gy,gx) 分块噪声
    grid: tuple = (4, 3)
    border_fraction: np.ndarray | None = None


def _odd(v: float, lo: int = 3) -> int:
    k = max(lo, int(round(v)))
    return k if k % 2 == 1 else k + 1


def detect_background(
    bgr: np.ndarray,
    k: int = 3,
    grid: tuple = (4, 3),
    sample_step: int = 4,
    border_ratio: float = 0.06,
) -> BackgroundModel:
    """检测背景色并给出类型。"""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    band = max(3, int(min(h, w) * border_ratio))

    # 采样像素用于聚类
    sub = lab[::sample_step, ::sample_step].reshape(-1, 3)
    if len(sub) < k * 10:
        sub = lab.reshape(-1, 3)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(sub, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()

    counts = np.bincount(labels, minlength=k).astype(np.float32)
    weights = counts / counts.sum()

    # 边框像素落在哪个簇 → 该簇的背景可能性
    border = np.concatenate([
        lab[:band, :, :].reshape(-1, 3), lab[-band:, :, :].reshape(-1, 3),
        lab[:, :band, :].reshape(-1, 3), lab[:, -band:, :].reshape(-1, 3),
    ])
    d = np.linalg.norm(border[:, None, :] - centers[None, :, :], axis=2)
    border_lab = np.argmin(d, axis=1)
    border_frac = np.bincount(border_lab, minlength=k).astype(np.float32) / max(1, len(border_lab))

    keep = [i for i in range(k) if border_frac[i] >= 0.15]
    if not keep:
        keep = [int(np.argmax(weights))]

    colors = centers[keep].astype(np.float32)
    weights_kept = weights[keep]

    # 均匀度：背景色在边框上的离散程度（越小越均匀）
    spread = float(np.median(np.linalg.norm(border - colors[np.argmin(
        np.linalg.norm(border[:, None, :] - colors[None, :, :], axis=2), axis=1)], axis=1)))
    uniformity = float(np.clip(1.0 - spread / 40.0, 0.0, 1.0))

    # 背景类型：看边框沿水平/垂直方向的缓慢变化
    top = lab[:band, :, :].mean(axis=(0, 2))
    bottom = lab[-band:, :, :].mean(axis=(0, 2))
    left = lab[:, :band, :].mean(axis=(0, 2))
    right = lab[:, -band:, :].mean(axis=(0, 2))
    lr_gap = float(np.linalg.norm(left - right))
    tb_gap = float(np.linalg.norm(top - bottom))
    kind = "uniform"
    if max(lr_gap, tb_gap) > 18:
        kind = "gradient"
    elif uniformity < 0.55:
        kind = "textured"

    return BackgroundModel(
        colors=colors, weights=weights_kept, kind=kind,
        uniformity=uniformity, grid=grid, border_fraction=border_frac[keep],
    )


def tile_sigma(bgr: np.ndarray, model: BackgroundModel, grid: tuple | None = None) -> np.ndarray:
    """分块估计背景噪声：每块内残差的 MAD，作为局部阈值基准。"""
    grid = grid or model.grid
    res = raw_residual(bgr, model)
    h, w = res.shape
    gy, gx = grid
    out = np.zeros((gy, gx), np.float32)
    for i in range(gy):
        for j in range(gx):
            ys, ye = int(h * i / gy), int(h * (i + 1) / gy)
            xs, xe = int(w * j / gx), int(w * (j + 1) / gx)
            blk = res[ys:ye, xs:xe]
            if blk.size == 0:
                continue
            med = np.median(blk)
            mad = np.median(np.abs(blk - med)) * 1.4826
            out[i, j] = max(3.0, float(mad))
    return out


def raw_residual(bgr: np.ndarray, model: BackgroundModel) -> np.ndarray:
    """到最近背景色的 Lab 距离（未归一化）。"""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = np.min(np.linalg.norm(lab[:, :, None, :] - model.colors[None, None, :, :], axis=3), axis=2)
    return d.astype(np.float32)


def normalized_residual(bgr: np.ndarray, model: BackgroundModel) -> np.ndarray:
    """按分块噪声归一化：局部噪声越大，同样的色差越不被当作前景。"""
    res = raw_residual(bgr, model)
    if model.tile_sigma is None:
        model.tile_sigma = tile_sigma(bgr, model)
    h, w = res.shape
    gy, gx = model.tile_sigma.shape
    sig = cv2.resize(model.tile_sigma, (w, h), interpolation=cv2.INTER_LINEAR)
    return res / np.maximum(sig, 1e-3)


# ---------------------------------------------------------------------------
# 多帧背景板：同机位拍摄时的统计背景
# ---------------------------------------------------------------------------


def _normalize_lab(lab: np.ndarray, own: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """把一帧的 Lab 归一化到参考背景色：L 用增益，a/b 用偏移。"""
    out = lab.astype(np.float32).copy()
    out[..., 0] *= float(ref[0]) / max(float(own[0]), 1e-3)
    out[..., 1] += float(ref[1]) - float(own[1])
    out[..., 2] += float(ref[2]) - float(own[2])
    return np.clip(out, 0, 255)


def build_plate(frames: list[np.ndarray], coverage_std: float = 12.0):
    """由同机位多帧构建背景板。

    返回 (plate_bgr, coverage, std_map)：
      plate     —— 逐像素中值（已做光度归一化）
      coverage  —— 时间标准差小于阈值的像素占比，代表背景板可信程度
      std_map   —— 每像素的时间标准差，可用于掩掉不可信区域

    注意：**必须先把各帧曝光/白平衡对齐**，否则手机自动曝光会让中值全是重影；
    另外每块区域至少要有一半以上的帧是空的，覆盖不足时中值会保留目标。
    """
    if not frames:
        raise ValueError("no frames")

    labs, ref = [], None
    for f in frames:
        model = detect_background(f)
        own = model.colors[0]
        if ref is None:
            ref = own
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
        labs.append(_normalize_lab(lab, own, ref))

    stack = np.stack(labs)
    med = np.median(stack, axis=0)
    std_map = stack.std(axis=0).mean(axis=2)
    coverage = float((std_map < coverage_std).mean())
    plate = cv2.cvtColor(np.clip(med, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return plate, coverage, std_map


def residual_against_plate(bgr: np.ndarray, plate: np.ndarray) -> np.ndarray:
    """与背景板的 Lab 距离；两者先按各自背景色对齐，抵消曝光差异。"""
    m_img = detect_background(bgr)
    m_plate = detect_background(plate)
    lab_i = _normalize_lab(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32),
                           m_img.colors[0], m_plate.colors[0])
    lab_p = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = np.linalg.norm(lab_i - lab_p, axis=2)
    return d.astype(np.float32)
