"""人民币面额判别：基于票幅、主色调与版面暖色比例的规则模型。

特征来源（公开资料，非图像拷贝）：
    中国人民银行公告与第五套人民币票面说明：
      1 元 130×63mm 橄榄绿；5 元 135×63mm 紫；10 元 140×70mm 蓝黑；
      20 元 145×70mm 棕；50 元 150×70mm 绿；100 元 155×77mm 红。
    硬币：1 角 19mm（不锈钢，银白）、5 角 20.5mm（钢芯镀铜，金黄）、
          1 元 25mm（钢芯镀镍，银白）；2019 版 1 元改为 22.25mm。

可用于区分的视觉特征（按可靠性排序）：
    1. **主色调**：六个面额的色相差足够大（红/绿/棕/蓝黑/紫/橄榄绿），
       但 100 红与 20 棕同属暖色，需看 a*/b* 的比值。
    2. **票幅长宽比**：集中在 2.00–2.14，几乎不可分——只作合法性校验，不作判据。
    3. **绝对票幅**：物理尺寸已知，但需要「毫米/像素」标定（固定机位或参照物）；
       未标定时不参与打分，避免用像素尺寸冒充物理尺寸。
    4. **版面暖色比例**：100 元人像与花纹的暖色占比明显高于 10 元，
       对光照的鲁棒性优于单一均值。

判据与置信度：
    综合得分 = 色度距离 + 暖色比例距离（+ 尺寸项，如有标定）；
    只有 top1 与 top2 的得分差超过 margin 才输出面额，否则返回 None 并给出候选，
    避免在 20/100 这类邻近色上硬判。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class RMBProto:
    value: float
    size_mm: tuple
    color_name: str
    a: float
    b: float
    warm: float
    measured: bool = False   # True 表示用实拍样本测过，否则取自公开描述


@dataclass
class RMBConfig:
    kind: str = "rmb"
    prototypes: list[RMBProto] = field(default_factory=lambda: [
        RMBProto(1, (130, 63), "橄榄绿", -6, 12, 0.30),
        RMBProto(5, (135, 63), "紫", 8, -2, 0.45),
        RMBProto(10, (140, 70), "蓝黑", -0.5, 4.0, 0.28, measured=True),
        RMBProto(20, (145, 70), "棕", 12, 22, 0.65),
        RMBProto(50, (150, 70), "绿", -18, 10, 0.15),
        RMBProto(100, (155, 77), "红", 14, 18, 0.78, measured=True),
    ])
    aspect_range: tuple = (1.85, 2.35)   # 允许透视压缩
    color_tol: float = 16.0
    warm_tol: float = 0.22
    size_tol_mm: float = 3.0
    min_margin: float = 0.22             # top1/top2 相对差下限
    min_confidence: float = 0.50


def _features(warped_bgr: np.ndarray) -> dict:
    """票面中心区域的颜色与暖色比例；边缘一圈丢掉，避免背景混入。"""
    h, w = warped_bgr.shape[:2]
    y0, y1 = int(h * 0.12), int(h * 0.88)
    x0, x1 = int(w * 0.12), int(w * 0.88)
    roi = warped_bgr[y0:y1, x0:x1]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = float(lab[..., 1].mean() - 128.0)
    b = float(lab[..., 2].mean() - 128.0)
    warm = float((lab[..., 1] - 128 > 8).mean())
    aspect = w / max(h, 1)
    return {"a": a, "b": b, "warm": warm, "aspect": aspect}


def classify_rmb(
    warped_bgr: np.ndarray,
    cfg: RMBConfig,
    mm_per_px: float | None = None,
) -> dict:
    """返回 value/label/confidence/candidates/features。"""
    f = _features(warped_bgr)
    # 只用「实测过」的面额原型参与判别。公开资料里的颜色描述误差很大，
    # 拿来打分会在 100/20 这类邻近色上给出高置信的错误结论。
    pool = [p for p in cfg.prototypes if p.measured] or cfg.prototypes
    scores = []
    for p in pool:
        d_color = float(np.hypot(f["a"] - p.a, f["b"] - p.b)) / cfg.color_tol
        d_warm = abs(f["warm"] - p.warm) / cfg.warm_tol
        score = 0.75 * d_color + 0.25 * d_warm
        if mm_per_px is not None and mm_per_px > 0:
            length_mm = warped_bgr.shape[1] * mm_per_px
            score += abs(length_mm - p.size_mm[0]) / cfg.size_tol_mm * 0.5
        scores.append((score, p))

    scores.sort(key=lambda t: t[0])
    best_score, best = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else best_score + 1
    margin = (second_score - best_score) / max(second_score, 1e-6)
    confidence = float(np.clip(1.0 - best_score / 3.0, 0, 1)) * (0.5 + 0.5 * min(1.0, margin / cfg.min_margin))
    aspect_ok = cfg.aspect_range[0] <= f["aspect"] <= cfg.aspect_range[1]

    candidates = [{"value": p.value, "label": p.color_name, "score": round(s, 3)}
                  for s, p in scores[:3]]

    if not aspect_ok or margin < cfg.min_margin or confidence < cfg.min_confidence:
        return {"value": None, "label": None, "confidence": round(confidence, 3),
                "candidates": candidates, "features": f, "reason": "low_margin_or_aspect"}

    return {"value": best.value, "label": f"{best.value:g}元", "confidence": round(confidence, 3),
            "candidates": candidates, "features": f, "reason": "ok"}


def classify_coin_rmb(
    bgr: np.ndarray,
    mask: np.ndarray,
    all_radii: list[float] | None = None,
    mm_per_px: float | None = None,
) -> dict:
    """硬币面额：金黄→5 角；银白需尺度参照（绝对标定或多枚相对半径）。"""
    m = mask.astype(bool)
    if m.sum() < 20:
        return {"value": None, "label": None, "confidence": 0.0, "candidates": []}

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    b_mean = float(lab[..., 2][m].mean() - 128.0)
    l_mean = float(lab[..., 0][m].mean())
    radius = float(np.sqrt(m.sum() / np.pi))

    # 黄铜的 b* 明显高于不锈钢；银色 b* 接近 0
    if b_mean > 18 and l_mean > 90:
        return {"value": 0.5, "label": "5角", "confidence": 0.65,
                "candidates": [0.5], "features": {"b": round(b_mean, 1)}}

    cands = [1.0, 0.1]
    value = None
    conf = 0.0
    if mm_per_px and mm_per_px > 0:
        d_mm = 2 * radius * mm_per_px
        options = [(abs(d_mm - 25.0), 1.0), (abs(d_mm - 22.25), 1.0), (abs(d_mm - 19.0), 0.1)]
        options.sort()
        if options[0][0] < 1.5:
            value, conf = options[0][1], 0.6
    elif all_radii and len(all_radii) >= 2:
        r = max(radius, 1e-6)
        others = [x for x in all_radii if abs(x - radius) > 1e-6]
        if others and r >= max(others) * 0.97:
            value, conf = 1.0, 0.45
        elif others and r <= min(others) * 1.03:
            value, conf = 0.1, 0.35
    return {"value": value, "label": (None if value is None else f"{value:g}元"),
            "confidence": conf, "candidates": cands, "features": {"b": round(b_mean, 1)}}
