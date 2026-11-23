"""中文路径安全的图像读写与 EXIF 清理。

Windows 下 cv2.imread 无法读取含中文的绝对路径（返回 None），
统一改用 np.fromfile + cv2.imdecode。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """读取图像，支持中文路径。失败返回 None。"""
    p = Path(path)
    if not p.is_file():
        return None
    buf = np.fromfile(str(p), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite_unicode(path: str | Path, image: np.ndarray) -> bool:
    """写出图像，支持中文路径。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix if p.suffix else ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(p))
    return True


def strip_exif(src: str | Path, dst: str | Path, quality: int = 92) -> bool:
    """重新编码以剥离 EXIF（含 GPS）。用于提交前的隐私自查。"""
    img = imread_unicode(src)
    if img is None:
        return False
    p = Path(dst)
    p.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if p.suffix.lower() in {".jpg", ".jpeg"} else []
    ok, buf = cv2.imencode(p.suffix or ".png", img, params)
    if not ok:
        return False
    buf.tofile(str(p))
    return True
