"""纸币面值识别 GUI。

运行环境：系统 Python 3.14（有 tkinter + Pillow）。
图像处理由项目 venv 的 opencv 完成，通过子进程调用 `single_note_demo.py`，
避免在两个解释器之间搬运 opencv。

用法（由 纸币识别.bat 启动）：
    python scripts/gui.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# 本机项目 venv 没有 tkinter；若当前解释器缺 tkinter，则自重启到系统 Python 3.14。
# 系统解释器路径由 LOCALAPPDATA 动态求出，不写死用户名。


def _has_tk() -> bool:
    try:
        import tkinter  # noqa: F401
        return True
    except ImportError:
        return False


if not _has_tk():
    cands = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        cands.append(Path(local) / "Python" / "bin" / "python.exe")
    for c in cands:
        if c.exists():
            print(f"[gui] 当前解释器无 tkinter，改用 {c}")
            subprocess.run([str(c), __file__, *sys.argv[1:]])
            sys.exit(0)
    print("未找到带 tkinter 的 Python 解释器，无法启动 GUI")
    sys.exit(1)

import json  # noqa: E402
import tempfile  # noqa: E402
import tkinter as tk  # noqa: E402
from tkinter import filedialog  # noqa: E402

from PIL import Image, ImageTk  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT.parent / ".venv" / "Scripts" / "python.exe"
DEMO = ROOT / "scripts" / "single_note_demo.py"

BG = "#f4f4f5"


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("纸币面值识别")
        root.configure(bg=BG)
        self.path: Path | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.tmp = Path(tempfile.gettempdir()) / "note_gui_montage.png"

        bar = tk.Frame(root, bg=BG)
        bar.pack(fill="x", padx=12, pady=10)
        tk.Button(bar, text="打开图片", width=12, command=self.open).pack(side="left")
        tk.Button(bar, text="识别", width=12, command=self.recognize).pack(side="left", padx=8)
        self.status = tk.Label(bar, text="请先打开一张纸币图片", bg=BG, fg="#666", anchor="w")
        self.status.pack(side="left", fill="x", expand=True, padx=8)

        # 中间结果面板（六幅拼接图）
        frame = tk.Frame(root, bg="white", highlightthickness=1, highlightbackground="#ddd")
        frame.pack(padx=12, pady=6)
        self.panel = tk.Label(frame, bg="white", text="此处显示中间结果\n（原图 → 定位 → 校正 → 灰度 → 二值 → 数字区域）",
                              font=("Microsoft YaHei", 12), fg="#999")
        self.panel.pack(padx=8, pady=8)

        # 识别结果
        res = tk.Frame(root, bg="white", highlightthickness=1, highlightbackground="#ddd")
        res.pack(fill="x", padx=12, pady=6)
        tk.Label(res, text="识别结果", bg="white", font=("Microsoft YaHei", 10)).pack(anchor="w", padx=10, pady=4)
        self.result = tk.Label(res, text="—", bg="white", fg="#c00000", font=("Microsoft YaHei", 42, "bold"))
        self.result.pack(pady=4)
        self.result_sub = tk.Label(res, text="", bg="white", fg="#666", font=("Microsoft YaHei", 9))
        self.result_sub.pack(pady=4)

    def open(self) -> None:
        p = filedialog.askopenfilename(title="选择纸币图片",
                                       filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")])
        if not p:
            return
        self.path = Path(p)
        img = Image.open(p)
        img.thumbnail((760, 480))
        self.photo = ImageTk.PhotoImage(img)
        self.panel.config(image=self.photo, text="")
        self.result.config(text="—")
        self.result_sub.config(text="")
        self.status.config(text=f"已载入 {self.path.name}，点「识别」")

    def recognize(self) -> None:
        if self.path is None:
            self.status.config(text="请先打开图片")
            return
        if not VENV_PY.exists():
            self.status.config(text="未找到项目虚拟环境，无法识别")
            return
        self.status.config(text="识别中…")
        self.root.update_idletasks()

        try:
            proc = subprocess.run(
                [str(VENV_PY), str(DEMO), "--image", str(self.path), "--save", str(self.tmp)],
                capture_output=True, text=True, encoding="utf-8", timeout=180)
        except subprocess.TimeoutExpired:
            self.status.config(text="识别超时")
            return

        try:
            result = None
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    result = json.loads(line)
                    break
            if result is None:
                raise ValueError("no json")
        except Exception:
            self.status.config(text=f"识别失败：{proc.stderr.strip()[-200:]}")
            return

        if self.tmp.exists():
            img = Image.open(self.tmp)
            img.thumbnail((860, 560))
            self.photo = ImageTk.PhotoImage(img)
            self.panel.config(image=self.photo, text="")

        if result.get("label"):
            self.result.config(text=result["label"])
            sub = f"置信度 {result.get('confidence', 0):.2f}"
            cands = [c["label"] for c in result.get("candidates", [])[:3] if c.get("label")]
            if cands:
                sub += f"　候选：{' / '.join(cands)}"
        else:
            self.result.config(text="未识别")
            sub = "可能是其他面额，或图片背景/角度不合适"
        self.result_sub.config(text=sub)
        self.status.config(text=f"定位面积 {result.get('quad_area', 0):.2f}")


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
