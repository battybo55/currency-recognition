"""场景自适应前景提取：从单张图自身估计背景，不依赖固定模板。

思路：
    1. **边界颜色模型**：目标通常位于画面中部，外圈以背景为主。分别取上下左右
       四条边框带的中位色（Lab），得到 4 个背景参考色，能容忍左右光照不均；
    2. **色差响应**：每个像素到最近参考色的 Lab 距离，归一化后作为「与背景有多不像」；
    3. **自适应阈值 + 形态学**：Otsu 分割（背景峰集中在 0 附近），开闭运算去噪、填孔；
    4. **粘连拆分**：距离变换 + 峰值标记 + 分水岭，把「纸币压着硬币」这类连通块拆开；
    5. **形状分类**：按长宽比、矩形度、圆度分成 note / coin。

所有参数都是相对画面短边的比例，换分辨率、换背景不需要改。
已知前提：目标不应占据四条边框的全部，否则背景参考色会被污染。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SceneConfig:
    """场景自适应参数，全部为相对量。"""

    border_ratio: float = 0.06      # 边框取样带宽 / 短边
    use_edge_cue: bool = True       # 是否融合边缘线索
    residual_gain: float = 1.6      # 色差对比度拉伸
    min_area_ratio: float = 0.0015  # 目标最小面积占比
    max_area_ratio: float = 0.85    # 超过视为整幅
    open_ratio: float = 0.006       # 开运算核 / 短边
    close_ratio: float = 0.012      # 闭运算核 / 短边
    split_min_distance: float = 0.08  # 分水岭种子最小间距 / 短边
    split_min_area: float = 0.0008    # 拆分后子区域最小面积
    note_aspect: tuple = (1.20, 3.4)  # 折角纸币长宽比会降到 1.2 附近
    coin_aspect: tuple = (0.70, 1.40)
    note_extent: float = 0.50
    note_min_area: float = 0.015      # 面积够大也可判为纸币，不只看形状
    coin_max_area: float = 0.06
    coin_circularity: float = 0.55
    coin_fill: float = 0.70


def _odd(v: float, lo: int = 3) -> int:
    k = max(lo, int(round(v)))
    return k if k % 2 == 1 else k + 1


def border_reference_colors(bgr: np.ndarray, cfg: SceneConfig) -> np.ndarray:
    """四条边框带的中位色（Lab），返回 (4,3)。"""
    h, w = bgr.shape[:2]
    band = max(3, int(min(h, w) * cfg.border_ratio))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    strips = [
        lab[:band, :].reshape(-1, 3),
        lab[-band:, :].reshape(-1, 3),
        lab[:, :band].reshape(-1, 3),
        lab[:, -band:].reshape(-1, 3),
    ]
    return np.stack([np.median(s, axis=0) for s in strips]).astype(np.float32)


def background_residual(bgr: np.ndarray, cfg: SceneConfig | None = None) -> np.ndarray:
    """到最近背景参考色的 Lab 距离，归一到 0–255。"""
    cfg = cfg or SceneConfig()
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    refs = border_reference_colors(bgr, cfg)
    d = np.min(np.linalg.norm(lab[:, :, None, :] - refs[None, None, :, :], axis=3), axis=2)
    d = np.clip(d * cfg.residual_gain, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(d, (3, 3), 0)


def edge_mask(bgr: np.ndarray, cfg: SceneConfig | None = None) -> np.ndarray:
    """边缘线索：双边滤波压掉纹理后做自适应 Canny，闭运算连接、轮廓填充。

    与颜色线索互补——硬币与木纹颜色接近时，颜色距离失效，但币缘梯度仍在。
    """
    cfg = cfg or SceneConfig()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    med = float(np.median(smooth))
    lo = int(max(10, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(smooth, lo, hi)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    k = _odd(min(gray.shape[:2]) * cfg.close_ratio)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(edges)
    for c in contours:
        cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)
    return filled


def foreground_mask(bgr: np.ndarray, cfg: SceneConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """返回 (掩膜, 响应图)。掩膜 0/255。"""
    cfg = cfg or SceneConfig()
    residual = background_residual(bgr, cfg)
    h, w = residual.shape[:2]
    short = min(h, w)

    thr, mask = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thr < 8:  # 响应整体很弱：空场景或无对比目标
        return np.zeros_like(mask), residual

    k_open = _odd(short * cfg.open_ratio)
    k_close = _odd(short * cfg.close_ratio)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))

    # 融合边缘线索：颜色接近背景但轮廓清晰的目标（木纹上的硬币）由边缘补回
    if cfg.use_edge_cue:
        mask = cv2.bitwise_or(mask, edge_mask(bgr, cfg))

    # 填孔：用外轮廓重绘，避免 floodFill 在「前景贴边」时把整幅图当成空洞
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    for c in contours:
        cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)
    mask = filled

    # 保险：若掩膜几乎覆盖全图，说明阈值失效，按空场景处理
    if mask.mean() > 0.88 * 255:
        return np.zeros_like(mask), residual
    return mask, residual


def _split_component(comp: np.ndarray, cfg: SceneConfig) -> list[np.ndarray]:
    """对粘连连通块做距离变换分水岭，返回若干子掩膜。失败时返回原掩膜。"""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    h, w = comp.shape[:2]
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    if float(dist.max()) <= 1:
        return [comp]

    md = max(3, int(min(h, w) * cfg.split_min_distance))
    coords = peak_local_max(dist, min_distance=md, threshold_rel=0.35, exclude_border=False)
    if len(coords) <= 1:
        return [comp]

    markers = np.zeros_like(comp, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx
    labels = watershed(-dist, markers, mask=comp)

    parts = []
    for lab in range(1, int(labels.max()) + 1):
        m = (labels == lab).astype(np.uint8) * 255
        if m.sum() / 255 < w * h * cfg.split_min_area:
            continue
        parts.append(m)
    return parts if len(parts) > 1 else [comp]


def classify_component(contour: np.ndarray, image_shape: tuple[int, int], cfg: SceneConfig) -> dict:
    """按形状给轮廓打标签：note / coin / other。"""
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True)) or 1e-6
    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    if min(rw, rh) <= 2:
        return {"kind": "other", "area": area, "aspect": 0.0, "circularity": 0.0, "extent": 0.0}

    aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
    extent = area / max(rw * rh, 1e-6)
    circularity = 4 * np.pi * area / (peri * peri)
    fill = area / (np.pi * (max(rw, rh) / 2) ** 2 + 1e-6)

    kind = "other"
    aspect_ok = cfg.note_aspect[0] <= aspect <= cfg.note_aspect[1]
    if aspect_ok and (extent >= cfg.note_extent or area >= image_shape[0] * image_shape[1] * cfg.note_min_area):
        kind = "note"
    elif cfg.coin_aspect[0] <= aspect <= cfg.coin_aspect[1] and area <= image_shape[0] * image_shape[1] * cfg.coin_max_area and (
        circularity >= cfg.coin_circularity or fill >= cfg.coin_fill
    ):
        kind = "coin"
    return {
        "kind": kind, "area": area, "aspect": aspect,
        "circularity": circularity, "extent": extent, "rect": rect,
    }


def extract_objects(
    bgr: np.ndarray,
    cfg: SceneConfig | None = None,
    split: bool = True,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """检测纸币与硬币候选，返回 (objects, mask, residual)。

    objects 每项含 kind / contour / bbox / mask / area_ratio / 几何量。
    """
    cfg = cfg or SceneConfig()
    h, w = bgr.shape[:2]
    mask, residual = foreground_mask(bgr, cfg)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    objects: list[dict] = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        ratio = area / (h * w)
        if not (cfg.min_area_ratio <= ratio <= cfg.max_area_ratio):
            continue

        comp = (labels == i).astype(np.uint8) * 255
        parts = [comp]
        if split:
            # 先看整体形状：长条形（像纸币）不拆，避免把折痕当成分界；
            # 只有接近方形的大团块才可能是「纸币压硬币」，交给分水岭。
            cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c0 = max(cs, key=cv2.contourArea) if cs else None
            if c0 is not None:
                r0 = cv2.minAreaRect(c0)[1]
                asp0 = max(r0) / max(min(r0), 1e-6) if min(r0) > 2 else 1.0
                if asp0 < cfg.note_aspect[0] and ratio > 0.03:
                    parts = _split_component(comp, cfg)
        for part in parts:
            contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            info = classify_component(c, (h, w), cfg)
            if info["kind"] == "other":
                continue
            info.update({
                "contour": c,
                "mask": part,
                "bbox": cv2.boundingRect(c),
                "area_ratio": float(cv2.contourArea(c)) / (h * w),
            })
            objects.append(info)

    objects.sort(key=lambda o: -o["area"])
    return objects, mask, residual
