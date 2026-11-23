"""端到端管线：一张图 → 纸币/硬币的定位、计数与（可选）面额。

输出结构与 README 第 7 节的数据契约一致。面额依赖外部模型文件；
没有模型时字段为 null，并在 low_confidence 中标记，不伪造结果。
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from .features import color_features, lbp_features
from .io_utils import imread_unicode
from .preprocess import resize_long_side
from .segment import (
    detect_coins_multiscale,
    find_banknote_quads,
    warp_quad,
)

DEFAULT_CFG = Path(__file__).resolve().parents[1] / "configs/baseline.yaml"
LABEL_VALUE = {"500_old": 500.0}


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CFG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def label_to_value(label: str) -> float | None:
    """把类别名转成面额数值；无法解析时返回 None。"""
    if label in LABEL_VALUE:
        return LABEL_VALUE[label]
    try:
        return float(label)
    except (TypeError, ValueError):
        return None


def _features(image: np.ndarray, size=(512, 256)) -> np.ndarray:
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return np.concatenate([color_features(small), lbp_features(small)])[None, :]


def _note_like(quad: np.ndarray, area_ratio: float) -> bool:
    """四边形是否像一张纸币：长宽比 1.5–2.8，且面积占比不超过 0.75。"""
    q = np.asarray(quad, np.float32)
    sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    aspect = max(sides) / max(min(sides), 1e-6)
    return 1.5 <= aspect <= 2.8 and 0.01 <= area_ratio <= 0.75


def analyze(
    image_path: str | Path,
    cfg: dict | None = None,
    banknote_model=None,
    draw: bool = False,
):
    """返回 (result_dict, overlay_or_None)。"""
    cfg = cfg or load_config()
    t0 = time.perf_counter()

    img = imread_unicode(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    img = resize_long_side(img, cfg["image"]["long_side"])
    overlay = img.copy() if draw else None

    # 纸币先定位，用于把硬币检测从票面区域里排除
    quads = find_banknote_quads(
        img,
        min_area_ratio=cfg["banknote"]["min_area_ratio"],
        min_rectangularity=cfg["banknote"].get("min_rectangularity", 0.80),
    )
    found = bool(quads)

    # 硬币
    circles = detect_coins_multiscale(
        img,
        bands=tuple(tuple(b) for b in cfg["coin"].get("bands", ((0.02, 0.05), (0.05, 0.12), (0.12, 0.30), (0.30, 0.60)))),
        param2=cfg["coin"]["hough"]["param2"],
        min_support=cfg["coin"].get("min_support", 0.55),
    )
    if found and len(circles):
        # 票面上的圆形花纹（国徽、印章、水印圈）会被误当成硬币，
        # 这里把落在「可信纸币」四边形内的候选剔除。
        # 注意：木纹桌面上 Otsu 常把纸币与硬币并成一团，产出的四边形
        # 长宽比不像纸币，若拿它去排除硬币会连真硬币一起删掉，
        # 所以只在四边形长得像纸币（长宽比 1.5–2.8、面积占比合理）时才排除。
        short = min(img.shape[:2])
        polys = [q.astype(np.float32) for q, _, r in quads if _note_like(q, r)]
        if polys:
            circles = np.asarray(
                [c for c in circles
                 if not (any(cv2.pointPolygonTest(p, (float(c[0]), float(c[1])), False) >= 0 for p in polys)
                         and c[2] < 0.35 * short)],
                dtype=np.float32,
            ).reshape(-1, 3)

    coins = [{"denomination": None, "center": [round(float(x), 1) + 0.0, round(float(y), 1) + 0.0],
              "radius": round(float(r), 1), "score": None} for x, y, r in circles]
    if draw:
        for x, y, r in circles:
            cv2.circle(overlay, (int(x), int(y)), int(r), (0, 200, 255), 2)

    banknotes = []
    for quad, method, area_ratio in quads:
        warped = warp_quad(img, quad, tuple(cfg["banknote"]["output_size"]))
        label, score = None, None
        if banknote_model is not None:
            proba = banknote_model.predict_proba(_features(warped))[0]
            idx = int(np.argmax(proba))
            label = str(banknote_model.classes_[idx])
            score = round(float(proba[idx]), 3)
        banknotes.append({
            "denomination": label_to_value(label) if label else None,
            "label": label,
            "quad": [[round(float(x), 1), round(float(y), 1)] for x, y in quad],
            "score": score,
            "area_ratio": round(float(area_ratio), 3),
        })
        if draw:
            cv2.polylines(overlay, [quad.astype(np.int32)], True, (0, 255, 0), 3)

    values = [b["denomination"] for b in banknotes] + [c["denomination"] for c in coins]
    total = float(sum(v for v in values if v is not None)) if all(v is not None for v in values) else None

    result = {
        "image": Path(image_path).name,
        "banknotes": banknotes,
        "coins": coins,
        "total": total,
        "low_confidence": bool(not found or banknote_model is None or len(coins) > 0),
        "debug": {
            "method": quads[0][1] if quads else "none",
            "banknote_count": len(banknotes),
            "coin_count": len(coins),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            "model": None if banknote_model is None else type(banknote_model).__name__,
        },
    }
    return result, overlay
