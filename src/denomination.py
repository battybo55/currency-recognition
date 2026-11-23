"""面额判别：颜色原型（纸币）与材质颜色（硬币）。

设计取舍：
    - 纸币面额以 Lab 色度 (a,b) 的最近原型判别。原型来自标定文件，
      未标定时给出的是占位值，置信度会被压低——**不会伪造结果**。
    - 硬币只有「金黄 vs 银白」这一条颜色判据是可靠的（5 角 vs 1 元/1 角）；
      同为银白的 1 元与 1 角需要尺度参照，无参照时返回 None 并给出候选。

标定方式见 scripts/calibrate_denomination.py。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "configs/denomination.yaml"


@dataclass
class NoteProto:
    value: float
    a: float
    b: float
    name: str = ""
    spread: float = 0.0   # 标定时该类到原型的中位距离，可用于自适应容差
    n: int = 0            # 标定样本数


@dataclass
class DenominationConfig:
    calibrated: bool = False
    coin_calibrated: bool = False   # 硬币材质判据需实测样本标定，默认关闭
    notes: list[NoteProto] = field(default_factory=list)
    note_tol: float = 18.0          # Lab 色度距离容差
    min_confidence: float = 0.45
    coin_gold_hue: tuple = (12, 38)
    coin_gold_sat: int = 70


def load_config(path: str | Path | None = None) -> DenominationConfig:
    p = Path(path or DEFAULT_PATH)
    if not p.exists():
        return DenominationConfig()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    notes = [NoteProto(**n) for n in data.get("notes", [])]
    return DenominationConfig(
        calibrated=bool(data.get("calibrated", False)),
        coin_calibrated=bool(data.get("coin_calibrated", False)),
        notes=notes,
        note_tol=float(data.get("note_tol", 18.0)),
        min_confidence=float(data.get("min_confidence", 0.45)),
        coin_gold_hue=tuple(data.get("coin_gold_hue", (12, 38))),
        coin_gold_sat=int(data.get("coin_gold_sat", 70)),
    )


def _mean_chroma(bgr: np.ndarray, inset: float = 0.15) -> tuple[float, float]:
    """取中心区域的平均 Lab 色度，避开边缘与背景。"""
    h, w = bgr.shape[:2]
    y0, y1 = int(h * inset), int(h * (1 - inset))
    x0, x1 = int(w * inset), int(w * (1 - inset))
    roi = bgr[y0:y1, x0:x1]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = float(lab[..., 1].mean() - 128.0)
    b = float(lab[..., 2].mean() - 128.0)
    return a, b


def classify_note(warped_bgr: np.ndarray, cfg: DenominationConfig) -> dict:
    """按色度最近原型判面额。返回 value/confidence/label/distance。"""
    if not cfg.notes:
        return {"value": None, "label": None, "confidence": 0.0, "distance": None}

    a, b = _mean_chroma(warped_bgr)
    dists = [float(np.hypot(a - p.a, b - p.b)) for p in cfg.notes]
    idx = int(np.argmin(dists))
    d = dists[idx]
    conf = float(max(0.0, 1.0 - d / max(cfg.note_tol, 1e-6)))
    if not cfg.calibrated:
        conf *= 0.5  # 未标定：降权，避免误当可信结果
    p = cfg.notes[idx]
    if conf < cfg.min_confidence:
        return {"value": None, "label": None, "confidence": round(conf, 3), "distance": round(d, 1)}
    return {"value": p.value, "label": p.name or str(p.value),
            "confidence": round(conf, 3), "distance": round(d, 1)}


def classify_coin(bgr: np.ndarray, mask: np.ndarray, cfg: DenominationConfig) -> dict:
    """硬币材质判别：金黄 → 5 角；银白 → 需尺度参照，返回 None + 候选。"""
    m = mask.astype(bool)
    if m.sum() < 20:
        return {"value": None, "label": None, "confidence": 0.0, "candidates": []}

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = float(np.median(hsv[..., 0][m]) * 2.0)   # OpenCV H 为 0–179
    s = float(np.median(hsv[..., 1][m]))
    v = float(np.median(hsv[..., 2][m]))

    lo, hi = cfg.coin_gold_hue
    # 未标定时不输出面额：暖光会把银币推到金色色相区间（实测 H≈22、S≈90），
    # 单靠色相与饱和度会误判，必须用实测的金/银样本标定后才启用。
    if not cfg.coin_calibrated:
        return {"value": None, "label": None, "confidence": 0.0,
                "candidates": [1.0, 0.5, 0.1], "hsv": [round(h, 1), round(s, 1), round(v, 1)]}
    if lo <= h <= hi and s >= cfg.coin_gold_sat and v > 60:
        return {"value": 0.5, "label": "5角", "confidence": 0.7, "candidates": [0.5],
                "hsv": [round(h, 1), round(s, 1), round(v, 1)]}
    return {"value": None, "label": None, "confidence": 0.0,
            "candidates": [1.0, 0.1], "hsv": [round(h, 1), round(s, 1), round(v, 1)]}


def calibrate(folder: Path, out: Path, inset: float = 0.15) -> DenominationConfig:
    """按 <folder>/<面额>/<图片> 的结构计算色度原型并写 yaml。"""
    notes: list[NoteProto] = []
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        imgs = [q for q in sub.iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png")]
        chroma = []
        for q in imgs:
            img = cv2.imdecode(np.fromfile(str(q), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            chroma.append(_mean_chroma(img, inset))
        if not chroma:
            continue
        arr = np.asarray(chroma)
        notes.append(NoteProto(value=float(sub.name), a=float(np.median(arr[:, 0])),
                               b=float(np.median(arr[:, 1])), name=sub.name))
    data = {
        "calibrated": True,
        "note_tol": 18.0,
        "min_confidence": 0.45,
        "notes": [{"value": n.value, "a": round(n.a, 1), "b": round(n.b, 1), "name": n.name} for n in notes],
        "coin_gold_hue": [12, 38],
        "coin_gold_sat": 70,
        "source": str(folder),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return load_config(out)
