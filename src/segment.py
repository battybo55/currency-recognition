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


def _quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    """用外接矩形近似计算两个四边形的 IoU，用于跨掩膜去重。"""
    ra = cv2.boundingRect(np.asarray(a, np.float32).astype(np.int32))
    rb = cv2.boundingRect(np.asarray(b, np.float32).astype(np.int32))
    ax, ay, aw, ah = ra
    bx, by, bw, bh = rb
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def find_banknote_quads(
    image: np.ndarray,
    min_area_ratio: float = 0.02,
    aspect_range=(1.15, 3.2),
    min_rectangularity: float = 0.80,
    max_per_mask: int = 6,
    iou_thresh: float = 0.35,
) -> list[tuple[np.ndarray, str, float]]:
    """返回图中全部候选纸币四边形 [(quad, method, area_ratio), ...]，按面积降序。

    与 v2 的区别：不再只取第一个，而是遍历每个掩膜的前若干个轮廓，
    跨掩膜按外接矩形 IoU 去重。用于多张纸币同框的真实场景。
    """
    h, w = image.shape[:2]
    gray = cv2.GaussianBlur(to_gray(image), (5, 5), 0)
    k = max(3, int(min(h, w) * 0.01))
    kernel = np.ones((k, k), np.uint8)

    masks = [("otsu", binary_foreground(gray)), ("edge", foreground_mask_edge(gray))]
    found: list[tuple[np.ndarray, str, float]] = []

    for mask_name, mask in masks:
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[: max_per_mask * 3]

        for c in contours:
            area = float(cv2.contourArea(c))
            ratio = area / (h * w)
            if ratio < min_area_ratio:
                continue

            rect = cv2.minAreaRect(c)
            rw, rh = rect[1]
            if min(rw, rh) <= 2:
                continue
            aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
            if not (aspect_range[0] <= aspect <= aspect_range[1]):
                continue
            if area / max(rw * rh, 1e-6) < min_rectangularity:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                quad, kind = order_quad(approx.reshape(4, 2)), "approx"
            else:
                quad, kind = order_quad(cv2.boxPoints(rect)), "rect"

            if any(_quad_iou(quad, q) > iou_thresh for q, _, _ in found):
                continue
            found.append((quad, f"{mask_name}:{kind}", ratio))
            if sum(1 for _, mth, _ in found if mth.startswith(mask_name)) >= max_per_mask:
                break

    found.sort(key=lambda t: -t[2])
    return found


def find_banknote_quad_v2(
    image: np.ndarray,
    min_area_ratio: float = 0.02,
    aspect_range=(1.15, 3.2),
    min_rectangularity: float = 0.80,
    max_candidates: int = 8,
):
    """带几何约束的纸币定位。

    在最大连通域之外增加三条物理先验，避免在多张叠放或桌面场景中框错：
      - 面积占比下限；
      - 最小外接矩形的长宽比落在纸币合理区间（透视会压低长宽比，故下限取 1.15）；
      - 矩形度（轮廓面积 / 最小外接矩形面积）足够高；用矩形度而非凸包实心度，
        是因为带阴影或圆角的钞票轮廓常呈非凸，实心度会误杀。
    返回 (quad, method, area_ratio, found)。
    """
    h, w = image.shape[:2]
    gray = cv2.GaussianBlur(to_gray(image), (5, 5), 0)
    k = max(3, int(min(h, w) * 0.01))
    kernel = np.ones((k, k), np.uint8)

    # 两套掩膜：阈值法优先（票面与背景亮度差大时最稳），
    # 边缘法兜底（浅色票面压在白纸上时 Otsu 会把两者归为一类）。
    masks = [("otsu", binary_foreground(gray)), ("edge", foreground_mask_edge(gray))]

    for mask_name, mask in masks:
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_candidates]

        for c in contours:
            area = float(cv2.contourArea(c))
            ratio = area / (h * w)
            if ratio < min_area_ratio:
                continue

            rect = cv2.minAreaRect(c)
            rw, rh = rect[1]
            if min(rw, rh) <= 2:
                continue
            aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
            if not (aspect_range[0] <= aspect <= aspect_range[1]):
                continue

            rect_area = max(rw * rh, 1e-6)
            rectangularity = area / rect_area
            if rectangularity < min_rectangularity:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return order_quad(approx.reshape(4, 2)), f"{mask_name}:approx", ratio, True
            return order_quad(cv2.boxPoints(rect)), f"{mask_name}:rect", ratio, True

    return None, "no_valid_candidate", 0.0, False


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


# ---------------------------------------------------------------------------
# M3：多尺度霍夫与边缘回退（针对 V3 暴露的四个问题）
# ---------------------------------------------------------------------------

DEFAULT_BANDS = ((0.02, 0.05), (0.05, 0.12), (0.12, 0.30), (0.30, 0.60))


def _dedupe_circles(circles: np.ndarray, overlap: float = 0.7, ratio_tol: float = 0.5) -> np.ndarray:
    """按半径从大到小贪心去重：同心或高度重叠且半径接近的只保留一个。"""
    if len(circles) == 0:
        return circles
    order = np.argsort(-circles[:, 2])
    kept: list[tuple[float, float, float]] = []
    for i in order:
        x, y, r = (float(v) for v in circles[i])
        duplicate = False
        for kx, ky, kr in kept:
            d = float(np.hypot(x - kx, y - ky))
            if d < overlap * max(r, kr) and abs(r - kr) / max(r, kr) < ratio_tol:
                duplicate = True
                break
        if not duplicate:
            kept.append((x, y, r))
    return np.asarray(kept, dtype=np.float32)


def detect_coins_multiscale(
    image: np.ndarray,
    bands=DEFAULT_BANDS,
    dp: float = 1.2,
    param2: int = 40,
    min_dist_factor: float = 2.0,
    min_support: float = 0.0,
) -> np.ndarray:
    """按半径分带跑霍夫，跨带去重，并可按圆周边缘支持度过滤。

    分带让每一带内的 `minDist` 与半径同量级（≥2r），避免同一枚硬币产生
    大量同心候选；`min_support > 0` 时再剔除纹理造成的假圆。
    """
    gray = to_gray(image)
    short = min(gray.shape[:2])
    blur = cv2.medianBlur(gray, 5)
    found: list[np.ndarray] = []

    for r0, r1 in bands:
        rmin = max(2, int(short * r0))
        rmax = max(rmin + 2, int(short * r1))
        min_dist = max(3, int(min_dist_factor * rmin))
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=100,
            param2=param2,
            minRadius=rmin,
            maxRadius=rmax,
        )
        if circles is not None:
            found.append(circles[0].astype(np.float32))

    if not found:
        return np.empty((0, 3), dtype=np.float32)

    merged = _dedupe_circles(np.vstack(found))
    if min_support <= 0 or len(merged) == 0:
        return merged

    scores = circle_edge_support(gray, merged)
    return merged[scores >= min_support]


def circle_edge_support(
    gray: np.ndarray,
    circles: np.ndarray,
    n_samples: int = 48,
    align_tol: float = 0.7,
    mag_percentile: float = 70.0,
) -> np.ndarray:
    """圆周边缘支持度：边界上有多少比例的点存在「沿半径方向」的强梯度。

    这是对霍夫候选的事后校验——真币的圆形轮廓会在整圈产生径向梯度，
    而纹理形成的假圆只在局部有响应。梯度阈值取本图分位数，不用绝对阈值。
    """
    from scipy.ndimage import map_coordinates

    if len(circles) == 0:
        return np.zeros((0,), dtype=np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    thr = float(np.percentile(mag, mag_percentile))

    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    scores = np.zeros((len(circles),), dtype=np.float32)

    h, w = gray.shape[:2]
    for i, (x, y, r) in enumerate(circles):
        xs = np.clip(x + r * cos_a, 0, w - 1)
        ys = np.clip(y + r * sin_a, 0, h - 1)
        gxs = map_coordinates(gx, [ys, xs], order=1, mode="nearest")
        gys = map_coordinates(gy, [ys, xs], order=1, mode="nearest")
        mags = np.hypot(gxs, gys)
        align = np.abs(gxs * cos_a + gys * sin_a) / np.maximum(mags, 1e-6)
        scores[i] = float(np.mean((mags > thr) & (align > align_tol)))
    return scores


def foreground_mask_edge(gray: np.ndarray, close_ratio: float = 0.01) -> np.ndarray:
    """Canny + 闭运算 + 孔洞填充。

    用于亮面金属币压在浅色背景上的情形：此时 Otsu 会把币面高光与背景
    归为同一类，得到破碎的掩膜；靠边缘闭合反而能拿到实心目标。
    """
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 50, 150)
    k = max(3, int(min(h, w) * close_ratio))
    k = k if k % 2 == 1 else k + 1
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    flood = edges.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(edges, holes)


def detect_coins_watershed_v2(
    image: np.ndarray,
    min_distance_ratio: float = 0.03,
    threshold_rel: float = 0.10,
) -> np.ndarray:
    """边缘回退掩膜 + 相对阈值峰值。

    与 v1 的差别有两处：掩膜改用 Canny 闭合（解决亮面金属），
    种子阈值改为相对峰值（解决密集场景下小币被 dist.max 压掉）。
    """
    from skimage.feature import peak_local_max

    gray = cv2.medianBlur(to_gray(image), 5)
    h, w = gray.shape[:2]
    b = foreground_mask_edge(gray)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    dist = cv2.distanceTransform(b, cv2.DIST_L2, 5)
    if float(dist.max()) <= 0:
        return np.empty((0, 3), dtype=np.float32)

    md = max(3, int(min(h, w) * min_distance_ratio))
    coords = peak_local_max(dist, min_distance=md, threshold_rel=threshold_rel, exclude_border=False)
    if len(coords) == 0:
        return np.empty((0, 3), dtype=np.float32)
    out = [(float(c), float(r), float(dist[r, c])) for r, c in coords]
    return np.asarray(out, dtype=np.float32)
