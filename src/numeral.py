"""面额数字模板匹配：从二值/灰度票面中匹配数字模板。

模板来自用户自拍样张（scripts/build_numeral_templates.py 生成）。
识别时对每个模板做 4 个旋转 × 多尺度归一化互相关，取最高分，
因此对票面整体 90°/180° 旋转不敏感。

局限：只能识别「有模板」的面额。目前有 10 与 100。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "numeral"
ROTATIONS = (0, 90, 180, 270)
SCALES = (0.6, 0.75, 0.9, 1.05, 1.2, 1.4)


def _rotate(img: np.ndarray, deg: int) -> np.ndarray:
    if deg == 0:
        return img
    return cv2.rotate(img, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                            270: cv2.ROTATE_90_COUNTERCLOCKWISE}[deg])


def load_templates(template_dir: Path | str = MODEL_DIR) -> dict[str, np.ndarray]:
    """返回 {面额标签: 灰度模板}。"""
    d = Path(template_dir)
    out: dict[str, np.ndarray] = {}
    for p in sorted(d.glob("*.png")):
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            out[p.stem] = img
    return out


def match_numeral(target: np.ndarray, templates: dict[str, np.ndarray],
                  min_score: float = 0.55) -> dict:
    """在 target（灰度）上匹配数字模板，返回 best label/score。

    target 会被归一化：模板与目标都缩放到模板高度与目标的比例附近比较。
    对每张模板做 4 旋转 × 6 尺度，用 TM_CCOEFF_NORMED 搜索。
    """
    if not templates:
        return {"label": None, "score": 0.0}
    if target.ndim == 3:
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    best = {"label": None, "score": 0.0}
    for label, tpl in templates.items():
        for rot in ROTATIONS:
            t = _rotate(tpl, rot)
            # 限制模板不小于目标高度的 20%，避免小模板在大图上乱匹配
            for s in SCALES:
                h = max(6, int(t.shape[0] * s))
                w = max(6, int(t.shape[1] * s))
                if h > target.shape[0] or w > target.shape[1]:
                    continue
                resized = cv2.resize(t, (w, h), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(target, resized, cv2.TM_CCOEFF_NORMED)
                _, mx, _, _ = cv2.minMaxLoc(res)
                if mx > best["score"]:
                    best = {"label": label, "score": float(mx)}
    if best["score"] < min_score:
        best["label"] = None
    return best


def build_template(img: np.ndarray, box: tuple[float, float, float, float],
                   height: int = 64) -> np.ndarray:
    """按比例框 (x0,y0,x1,y1) 裁出数字并二值化、去边、归一化高度。"""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    crop = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    binimg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 10)
    ys, xs = np.nonzero(binimg)
    if len(xs) == 0:
        return np.zeros((height, height), np.uint8)
    binimg = binimg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = height / max(binimg.shape[0], 1)
    return cv2.resize(binimg, (max(1, int(binimg.shape[1] * scale)), height),
                      interpolation=cv2.INTER_AREA)
