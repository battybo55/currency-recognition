"""分割：纸币四边形定位与透视校正；硬币霍夫圆与分水岭。

所有尺度参数以图像短边为基准，配置中只写比例。

已知限制（2026-11-23 实测，见 docs/validation_log.md）：
    当前默认参数只保证「跑通」，计数精度不可用。
    - 霍夫：minDist 需 ≥ 2×rmin，否则同一枚币会产生大量同心候选；
    - 半径区间固定时无法同时覆盖缩略图（r≈0.05）与特写（r≈0.6）；
    - 分水岭：0.30×dist.max 的种子阈值在密集场景过于严格；
    - 二值化：亮面金属币 + 白背景时 Otsu 会把币面高光与背景归为一类。
    M3 将改为多尺度检测 + 边缘回退，勿直接以本模块默认值作为评测结果。
"""

from __future__ import annotations

import cv2
import numpy as np

from .preprocess import to_gray


def order_quad(points: np.ndarray) -> np.ndarray:
    """把四点排成 左上、右上、右下、左下。"""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def warp_quad(image: np.ndarray, quad: np.ndarray, out_size=(512, 256)) -> np.ndarray:
    """按四点做透视变换，输出宽×高 = out_size。"""
    ow, oh = out_size
    dst = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(order_quad(quad), dst)
    return cv2.warpPerspective(image, m, (ow, oh))


def binary_foreground(gray: np.ndarray) -> np.ndarray:
    """Otsu 二值化，并按边框极性自动决定是否反相（背景应占多数）。"""
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([b[0, :], b[-1, :], b[:, 0], b[:, -1]])
    if float(border.mean()) > 127.0:
        b = cv2.bitwise_not(b)
    return b


def find_banknote_quad(
    image: np.ndarray,
    min_area_ratio: float = 0.02,
    corner_method: str = "best_of",
):
    """定位纸币四角。

    返回 (quad, method, area_ratio)；失败返回 (None, reason, 0.0)。
    """
    h, w = image.shape[:2]
    gray = to_gray(image)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    b = binary_foreground(gray)
    k = max(3, int(min(h, w) * 0.01))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "no_contour", 0.0
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < min_area_ratio * h * w:
        return None, "too_small", area / (h * w)

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    rect = cv2.boxPoints(cv2.minAreaRect(c))

    cand = []
    if len(approx) == 4:
        cand.append(("approx", order_quad(approx.reshape(4, 2))))
    cand.append(("rect", order_quad(rect)))

    if corner_method != "best_of":
        for name, q in cand:
            if name == corner_method:
                return q, name, area / (h * w)
        return None, "method_unavailable", area / (h * w)

    # best_of：优先四边形，但若其面积明显小于最小外接矩形则退化为矩形
    best_name, best_q = cand[0]
    q_area = abs(cv2.contourArea(best_q.astype(np.float32)))
    r_area = abs(cv2.contourArea(cand[-1][1].astype(np.float32)))
    if q_area < 0.75 * r_area:
        best_name, best_q = cand[-1]
    return best_q, best_name, area / (h * w)


def detect_coins_hough(
    image: np.ndarray,
    radius_ratio=(0.02, 0.45),
    dp: float = 1.2,
    min_dist_ratio: float = 0.05,
    param1: int = 100,
    param2: int = 30,
) -> np.ndarray:
    """霍夫圆检测。返回 (N,3) 的 x,y,r；无结果返回空数组。"""
    gray = to_gray(image)
    h, w = gray.shape[:2]
    short = min(h, w)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=max(1, int(short * min_dist_ratio)),
        param1=param1,
        param2=param2,
        minRadius=max(2, int(short * radius_ratio[0])),
        maxRadius=max(4, int(short * radius_ratio[1])),
    )
    if circles is None:
        return np.empty((0, 3), dtype=np.float32)
    return np.round(circles[0]).astype(np.float32)


def detect_coins_watershed(
    image: np.ndarray,
    min_distance_ratio: float = 0.05,
    min_area_ratio: float = 0.004,
) -> np.ndarray:
    """距离变换 + 局部极大值作种子，按种子数计数。返回 (N,3) 的 x,y,r 近似。"""
    gray = to_gray(image)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape[:2]
    b = binary_foreground(gray)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    dist = cv2.distanceTransform(b, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return np.empty((0, 3), dtype=np.float32)

    k = max(3, int(min(h, w) * min_distance_ratio))
    k = k if k % 2 == 1 else k + 1
    local_max = cv2.dilate(dist, np.ones((k, k), np.uint8))
    peaks = ((dist >= local_max - 1e-4) & (dist > 0.30 * dist.max())).astype(np.uint8)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks, connectivity=8)
    out = []
    min_area = min_area_ratio * h * w
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < max(1, min_area * 0.01):
            continue
        x, y = centroids[i]
        r = float(dist[labels == i].max())
        if r < 2:
            continue
        out.append((x, y, r))
    return np.asarray(out, dtype=np.float32) if out else np.empty((0, 3), dtype=np.float32)
