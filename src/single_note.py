"""单张纸币识别：定位 → 透视校正 → 灰度 → 二值 → 面额数字区域 → 识别。

与多目标场景的区别：这里假设画面中只有一张纸币且占主要面积，
因此可以用「票面先验」（长宽比约 2.05、面积占比 10%–40%、矩形度高）
在多个候选里挑最像纸币的那个，而不是依赖单一通道的阈值。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .segment import order_quad

NOTE_ASPECT = 2.05          # 第五套人民币长宽比 2.00–2.14
ASPECT_RANGE = (1.45, 2.90)
AREA_RANGE = (0.08, 0.95)   # 单张场景：票面通常占画面 40%–80%
AREA_TARGET = 0.55


@dataclass
class NoteQuad:
    quad: np.ndarray        # (4,2) 左上→右上→右下→左下
    area_ratio: float
    aspect: float
    rectangularity: float
    source: str
    contrast: float = 0.0   # 内部与外围环带的 Lab 色差


def _region_contrast(img: np.ndarray, quad: np.ndarray) -> float:
    """四边形内部与「外扩环带」的平均色差，用来排除「整幅图都是同一片底」的候选。"""
    h, w = img.shape[:2]
    mask_in = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask_in, [quad.astype(np.int32)], 255)
    k = max(3, int(min(h, w) * 0.02))
    inner = cv2.erode(mask_in, np.ones((k, k), np.uint8))
    outer = cv2.dilate(mask_in, np.ones((3 * k, 3 * k), np.uint8))
    ring = cv2.bitwise_and(outer, cv2.bitwise_not(mask_in))
    if inner.sum() == 0 or ring.sum() == 0:
        return 0.0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    m_in = cv2.mean(lab, mask=inner)[:3]
    m_out = cv2.mean(lab, mask=ring)[:3]
    return float(np.linalg.norm(np.array(m_in) - np.array(m_out)))


def _candidates_from_mask(mask: np.ndarray, src: str, min_area: float) -> list[NoteQuad]:
    h, w = mask.shape[:2]
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        area = float(cv2.contourArea(c))
        if area / (h * w) < min_area:
            continue
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if min(rw, rh) < 12:
            continue
        aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
        rect_area = rw * rh
        if rect_area <= 0:
            continue
        quad = order_quad(cv2.boxPoints(rect))
        out.append(NoteQuad(quad, area / (h * w), aspect, area / rect_area, src))
    return out


def candidate_quads(img: np.ndarray) -> list[NoteQuad]:
    """三条互补线索：Lab b 取反、灰度、边缘填充。"""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sources = {
        "b_inv": 255 - lab[..., 2],
        "gray": gray,
        "edge": None,
    }
    quads: list[NoteQuad] = []
    for name, ch in sources.items():
        if name == "edge":
            smooth = cv2.bilateralFilter(gray, 9, 75, 75)
            med = float(np.median(smooth))
            edges = cv2.Canny(smooth, int(max(10, 0.66 * med)), int(min(255, 1.33 * med)))
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            cs, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(edges)
            for c in cs:
                cv2.drawContours(filled, [c], -1, 255, cv2.FILLED)
            mask = filled
        else:
            blur = cv2.GaussianBlur(ch, (5, 5), 0)
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        quads.extend(_candidates_from_mask(mask, name, 0.03))
    return quads


def score(q: NoteQuad) -> float:
    """越小越像一张「占满画面」的纸币。不硬性拒绝，只按先验打分。"""
    if not (AREA_RANGE[0] <= q.area_ratio <= AREA_RANGE[1]):
        return 1e9
    if not (ASPECT_RANGE[0] <= q.aspect <= ASPECT_RANGE[1]):
        return 1e9
    return (abs(np.log(q.aspect / NOTE_ASPECT)) * 3.0
            + abs(q.area_ratio - AREA_TARGET) * 1.5
            + (1.0 - q.rectangularity) * 1.0
            + max(0.0, 18.0 - q.contrast) * 0.25)


def locate_note(img: np.ndarray) -> NoteQuad | None:
    quads = candidate_quads(img)
    if not quads:
        return None
    for q in quads:
        q.contrast = _region_contrast(img, q.quad)
    quads.sort(key=score)
    best = quads[0]
    return best if score(best) < 2.2 else None


def _orientation_score(img: np.ndarray) -> float:
    """横向文字结构越强，得分越高：用边缘投影的行/列方差比衡量。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(g, 50, 150)
    rows = e.sum(axis=1).astype(np.float32)
    cols = e.sum(axis=0).astype(np.float32)
    return float(rows.var() / (cols.var() + 1e-6))


def warp_note(img: np.ndarray, q: NoteQuad, width: int = 900) -> np.ndarray:
    """透视校正到统一尺寸，并在两个 90° 候选里选「文字横向」的那个。"""
    h = int(round(width / NOTE_ASPECT))
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, h - 1], [0, h - 1]])

    best, best_score = None, -1.0
    for shift in (0, 1):
        src = np.roll(q.quad.astype(np.float32), shift, axis=0)
        mat = cv2.getPerspectiveTransform(src, dst)
        warp = cv2.warpPerspective(img, mat, (width, h))
        s = _orientation_score(warp)
        if s > best_score:
            best, best_score = warp, s
    return best


def binarize(warped: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 12)


def numeral_region(warped: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """找面额数字区域：高瘦连通域横向聚成一组，取笔画面积最大的一组。

    不依赖固定版面坐标，只用「数字比花纹更瘦高、笔画更粗」这一形状先验。
    """
    binimg = binarize(warped)
    h, w = binimg.shape[:2]
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binimg, 8)

    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bh < h * 0.18 or bh > h * 0.75:
            continue
        if not (0.12 <= bw / max(bh, 1) <= 1.2):
            continue
        if area / max(bw * bh, 1) < 0.25:
            continue
        boxes.append((x, y, bw, bh, area))

    if not boxes:
        return None

    # 按高度聚类成组，取总面积最大的组
    boxes.sort(key=lambda b: b[4], reverse=True)
    seed = boxes[0]
    group = [b for b in boxes if abs(b[3] - seed[3]) < 0.35 * seed[3]
             and abs((b[1] + b[3] / 2) - (seed[1] + seed[3] / 2)) < 0.5 * seed[3]]
    x0 = max(0, min(b[0] for b in group) - 6)
    y0 = max(0, min(b[1] for b in group) - 6)
    x1 = min(w, max(b[0] + b[2] for b in group) + 6)
    y1 = min(h, max(b[1] + b[3] for b in group) + 6)
    crop = warped[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop, (x0, y0, x1 - x0, y1 - y0)
