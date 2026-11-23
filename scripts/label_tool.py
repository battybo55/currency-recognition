"""纸币/硬币轮廓标注工具（OpenCV 窗口，无额外依赖）。

用法：
    python scripts/label_tool.py --folder data/raw/self --pattern "s*.jpg"
    python scripts/label_tool.py --folder data/raw/self --pattern "s1*.jpg" --start 5

操作：
    左键        多边形模式：加一个顶点
    右键 / 回车  闭合当前多边形
    Backspace   撤销上一个顶点
    r           切换「矩形模式」：按住左键拖一个框，松开即完成
    u           撤销最后一个标注
    d / →       下一张（自动保存）
    a / ←       上一张
    s           立即保存
    q / Esc     退出（自动保存）

产物：
    <out>/labels.json     每张图的多边形（原图坐标）
    <out>/masks/<stem>.png 0/255 掩膜（原图尺寸）

标注建议：只标纸币/硬币的可见轮廓，不要包含阴影与桌面；
被遮挡的部分按可见部分闭合即可。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import imread_unicode  # noqa: E402

WINDOW = "label_tool"
INSTR = "L-click add | R-click/Enter close | r rect | u undo | d next | a prev | s save | q quit"


class Labeler:
    def __init__(self, files: list[Path], out_dir: Path, start: int = 0):
        self.files = files
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "masks").mkdir(exist_ok=True)
        self.json_path = out_dir / "labels.json"
        self.data: dict[str, list[list[list[float]]]] = {}
        if self.json_path.exists():
            self.data = json.loads(self.json_path.read_text(encoding="utf-8"))

        self.idx = start
        self.img = None
        self.scale = 1.0
        self.polys: list[list[list[float]]] = []
        self.current: list[list[float]] = []
        self.rect_mode = False
        self.dragging = False
        self.drag_start = (0, 0)
        self.drag_now = (0, 0)

    # ---------- 文件 ----------
    def load(self) -> bool:
        if not (0 <= self.idx < len(self.files)):
            return False
        p = self.files[self.idx]
        img = imread_unicode(p)
        if img is None:
            print(f"读取失败：{p}")
            return False
        self.img = img
        h, w = img.shape[:2]
        self.scale = min(1.0, 1400.0 / w, 900.0 / h)
        self.polys = [list(map(list, poly)) for poly in self.data.get(p.name, [])]
        self.current = []
        print(f"[{self.idx + 1}/{len(self.files)}] {p.name}  {w}×{h}  已有 {len(self.polys)} 个标注")
        return True

    def save(self) -> None:
        if self.img is None:
            return
        name = self.files[self.idx].name
        self.data[name] = self.polys
        self.json_path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")

        h, w = self.img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        for poly in self.polys:
            pts = np.asarray(poly, dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 255)
        cv2.imencode(".png", mask)[1].tofile(str(self.out_dir / "masks" / f"{Path(name).stem}.png"))

    # ---------- 交互 ----------
    def to_orig(self, x: float, y: float) -> list[float]:
        return [round(x / self.scale, 1), round(y / self.scale, 1)]

    def on_mouse(self, event, x, y, flags, param):
        if self.img is None:
            return
        if self.rect_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.dragging, self.drag_start = True, (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
                self.drag_now = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self.dragging:
                self.dragging = False
                x0, y0 = self.drag_start
                if abs(x - x0) > 4 and abs(y - y0) > 4:
                    self.polys.append([self.to_orig(x0, y0), self.to_orig(x, y0),
                                       self.to_orig(x, y), self.to_orig(x0, y)])
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current.append(self.to_orig(x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.close_polygon()

    def close_polygon(self) -> None:
        if len(self.current) >= 3:
            self.polys.append(self.current)
            print(f"  已闭合，共 {len(self.polys)} 个")
        self.current = []

    def render(self) -> np.ndarray:
        view = cv2.resize(self.img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        for poly in self.polys:
            pts = (np.asarray(poly, dtype=np.float32) * self.scale).astype(np.int32)
            cv2.fillPoly(view, [pts], (0, 255, 0))
            cv2.polylines(view, [pts], True, (0, 160, 0), 2)
        if self.current:
            pts = (np.asarray(self.current, dtype=np.float32) * self.scale).astype(np.int32)
            cv2.polylines(view, [pts], False, (0, 200, 255), 2)
            for p in pts:
                cv2.circle(view, tuple(p), 3, (0, 140, 255), -1)
        if self.dragging:
            cv2.rectangle(view, self.drag_start, self.drag_now, (255, 160, 0), 2)

        cv2.rectangle(view, (0, 0), (view.shape[1], 30), (32, 32, 32), -1)
        mode = "RECT" if self.rect_mode else "POLY"
        cv2.putText(view, f"{mode} | {self.files[self.idx].name} | n={len(self.polys)}",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(view, INSTR, (8, view.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)
        return view

    def run(self) -> None:
        if not self.load():
            print("没有可标注的图像")
            return
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            cv2.imshow(WINDOW, self.render())
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                self.save()
                break
            elif key in (13, 10):
                self.close_polygon()
            elif key == 8:
                if self.current:
                    self.current.pop()
                elif self.polys:
                    self.polys.pop()
            elif key == ord("r"):
                self.rect_mode = not self.rect_mode
                self.current = []
            elif key == ord("u"):
                if self.polys:
                    self.polys.pop()
            elif key == ord("s"):
                self.save()
                print("已保存")
            elif key in (ord("d"), 83, 3):
                self.save()
                self.idx += 1
                if not self.load():
                    print("已是最后一张")
                    self.idx -= 1
                    self.load()
            elif key in (ord("a"), 81, 2):
                self.save()
                self.idx -= 1
                if not self.load():
                    print("已是第一张")
                    self.idx += 1
                    self.load()
        cv2.destroyAllWindows()
        self.save()
        total = len(self.data)
        done = sum(1 for v in self.data.values() if v)
        print(f"\n已保存 {done} 张标注（共浏览 {total} 张）→ {self.json_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data/raw/self")
    ap.add_argument("--pattern", default="s*.jpg")
    ap.add_argument("--out", default="data/raw/annotations")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    folder = ROOT / args.folder
    files = sorted(folder.glob(args.pattern))
    if not files:
        raise SystemExit(f"{folder} 下没有匹配 {args.pattern} 的图像")
    print(f"共 {len(files)} 张：{[p.name for p in files]}")
    Labeler(files, ROOT / args.out, start=args.start).run()


if __name__ == "__main__":
    main()
