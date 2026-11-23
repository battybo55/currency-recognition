"""纸币面值识别 + 硬币计数 GUI（单进程，无外部依赖）。

运行环境：系统 Python 3.14（tkinter + cv2 + numpy + Pillow 齐全）。
打包成 exe 后解压即用，不再依赖项目 venv。

按钮：
    打开图片 —— 选择一张纸币或硬币照片
    识别纸币 —— 定位 → 校正 → 灰度 → 二值 → 面额识别
    识别硬币 —— 霍夫圆计数
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.io_utils import imread_unicode  # noqa: E402
from src.preprocess import resize_long_side  # noqa: E402
from src.rmb import RMBConfig, classify_rmb  # noqa: E402
from src.segment import detect_coins_multiscale  # noqa: E402
from src.single_note import binarize, locate_note, numeral_region, warp_note  # noqa: E402

BG = "#f4f4f5"
PANEL_W, PANEL_H = 260, 150


def to_photo(img: np.ndarray, w: int, h: int) -> ImageTk.PhotoImage:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil.thumbnail((w, h), Image.LANCZOS)
    return ImageTk.PhotoImage(pil)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("纸币面值识别与硬币计数")
        root.configure(bg=BG)
        self.path: Path | None = None
        self.img: np.ndarray | None = None
        self.photos: list = []

        bar = tk.Frame(root, bg=BG)
        bar.pack(fill="x", padx=12, pady=10)
        tk.Button(bar, text="打开图片", width=12, command=self.open).pack(side="left")
        tk.Button(bar, text="识别纸币", width=12, command=self.recognize_note).pack(side="left", padx=6)
        tk.Button(bar, text="识别硬币", width=12, command=self.recognize_coin).pack(side="left", padx=6)
        self.status = tk.Label(bar, text="请打开一张图片", bg=BG, fg="#666", anchor="w")
        self.status.pack(side="left", fill="x", expand=True, padx=8)

        grid = tk.Frame(root, bg=BG)
        grid.pack(padx=12, pady=4)
        self.panels: dict[str, tk.Label] = {}
        for i, name in enumerate(["原图", "定位", "透视校正", "灰度", "二值", "结果"]):
            cell = tk.Frame(grid, bg="white", highlightthickness=1, highlightbackground="#ddd")
            cell.grid(row=i // 3, column=i % 3, padx=4, pady=4)
            tk.Label(cell, text=name, bg="white", font=("Microsoft YaHei", 10)).pack(anchor="w", padx=4, pady=2)
            lab = tk.Label(cell, bg="white")
            lab.pack(padx=4, pady=2)
            self.panels[name] = lab

        res = tk.Frame(root, bg="white", highlightthickness=1, highlightbackground="#ddd")
        res.pack(fill="x", padx=12, pady=8)
        tk.Label(res, text="识别结果", bg="white", font=("Microsoft YaHei", 10)).pack(anchor="w", padx=10, pady=4)
        self.result = tk.Label(res, text="—", bg="white", fg="#c00000", font=("Microsoft YaHei", 42, "bold"))
        self.result.pack(pady=4)
        self.result_sub = tk.Label(res, text="", bg="white", fg="#666", font=("Microsoft YaHei", 9))
        self.result_sub.pack(pady=4)

    def open(self) -> None:
        p = filedialog.askopenfilename(title="选择图片",
                                       filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")])
        if not p:
            return
        img = imread_unicode(p)
        if img is None:
            self.status.config(text="无法读取该图片")
            return
        self.path = Path(p)
        self.img = resize_long_side(img, 1000)
        self._set("原图", self.img)
        for k in ["定位", "透视校正", "灰度", "二值", "结果"]:
            self._blank(k)
        self.result.config(text="—")
        self.result_sub.config(text="")
        self.status.config(text=f"已载入 {self.path.name}")

    def recognize_note(self) -> None:
        if self.img is None:
            self.status.config(text="请先打开图片")
            return
        q = locate_note(self.img)
        if q is not None:
            marked = self.img.copy()
            cv2.polylines(marked, [q.quad.astype(np.int32)], True, (0, 200, 0), 3)
            warped = warp_note(self.img, q)
            self._set("定位", marked)
            self._set("透视校正", warped)
            self._set("灰度", cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
            self._set("二值", binarize(warped))
            r = classify_rmb(warped, RMBConfig())
            conf = r["confidence"]
        else:
            warped = cv2.resize(self.img, (900, 439), interpolation=cv2.INTER_AREA)
            self._blank("定位")
            self._set("透视校正", warped)
            r = classify_rmb(warped, RMBConfig())
            conf = round(r["confidence"] * 0.8, 3)
            self.status.config(text="未定位到纸币，按整图判定")
        nr = numeral_region(warped)
        self._set("结果", nr[0] if nr is not None else warped)
        if r["label"]:
            self.result.config(text=r["label"])
            sub = f"置信度 {conf:.2f}"
            cands = [c["label"] for c in r["candidates"][:3] if c["label"]]
            if cands:
                sub += f"　候选：{' / '.join(cands)}"
        else:
            self.result.config(text="未识别")
            sub = "可能是其他面额，或图片背景/角度不合适"
        self.result_sub.config(text=sub)
        self.status.config(text=f"面额识别完成")

    def recognize_coin(self) -> None:
        if self.img is None:
            self.status.config(text="请先打开图片")
            return
        circles = detect_coins_multiscale(self.img, param2=45, min_support=0.45)
        vis = self.img.copy()
        for (x, y, r) in circles:
            cv2.circle(vis, (int(x), int(y)), int(r), (0, 200, 0), 3)
        self._set("结果", vis)
        self.result.config(text=f"{len(circles)} 枚")
        self.result_sub.config(text="霍夫圆计数（纯色/浅色背景更稳）")
        self.status.config(text=f"检测到 {len(circles)} 枚硬币")

    def _set(self, name: str, img: np.ndarray) -> None:
        photo = to_photo(img, PANEL_W, PANEL_H)
        self.photos.append(photo)
        self.panels[name].config(image=photo)

    def _blank(self, name: str) -> None:
        self.panels[name].config(image="")


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        try:
            from tkinter import messagebox
            messagebox.showerror("错误", f"{type(e).__name__}: {e}")
        except Exception:
            pass
        traceback.print_exc()
        sys.exit(1)
