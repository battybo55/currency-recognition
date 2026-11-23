"""提交前自检：扫描将被 git 跟踪的文件，拦截数据、凭据与隐私泄漏。

用法：
    python scripts/preflight.py          # 退出码 0 = 可以提交
    python scripts/preflight.py --all    # 同时检查已跟踪文件

检查项：
    1. 图像 / 压缩包 / 视频（仓库不托管任何数据）
    2. 单文件体积（>50 MB 阻断，>2 MB 警告）
    3. 绝对路径、凭据关键词、学号等敏感串
    4. 系统垃圾文件
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_EXT = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
    ".zip", ".7z", ".rar", ".tar", ".gz",
    ".mp4", ".avi", ".mov", ".gif",
}
ALLOW_NAMES = {"data/README.md"}
# 只放行自绘的统计图（散点、混淆矩阵等），且限 500 KB；
# 原始票币照片一律拦截，不含例外。
ALLOW_FIGURE = re.compile(r"^results/figures/[\w.\-]+\.png$")
FIGURE_MAX_BYTES = 500 * 1024

PATTERNS = [
    ("绝对路径", re.compile(r"[A-Za-z]:[\\/](?:Users|杂物|workbuddy|Windows)", re.I)),
    ("私钥", re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY")),
    ("GitHub Token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("OpenAI 风格 Key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("AWS Key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Kaggle 凭据", re.compile(r"\"key\"\s*:\s*\"[0-9a-f]{32}\"")),
    ("学号", re.compile(r"\b2024370\d{3}\b")),
]
JUNK = {"Thumbs.db", ".DS_Store"}


def git_files(include_tracked: bool) -> list[str]:
    cmd = ["git", "ls-files", "-co", "--exclude-standard"]
    if include_tracked:
        cmd = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        print("不是 git 仓库，或 git 不可用：", out.stderr.strip())
        sys.exit(2)
    return [line for line in out.stdout.splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="连同已跟踪文件一起检查")
    args = ap.parse_args()

    files = git_files(args.all)
    blocked: list[str] = []
    warned: list[str] = []
    problems: list[tuple[str, str, str]] = []

    for rel in files:
        p = ROOT / rel
        if not p.is_file():
            continue
        if p.name in JUNK or p.name.startswith("~$"):
            blocked.append(f"{rel}（系统垃圾文件）")
            continue
        if rel in ALLOW_NAMES:
            continue
        size = p.stat().st_size
        if p.suffix.lower() in FORBIDDEN_EXT:
            if ALLOW_FIGURE.match(rel) and size <= FIGURE_MAX_BYTES:
                pass
            else:
                blocked.append(f"{rel}（数据/媒体文件不入库）")
                continue
        if size > 50 * 1024 * 1024:
            blocked.append(f"{rel}（{size / 1e6:.1f} MB，超过 GitHub 上限）")
        elif size > 2 * 1024 * 1024:
            warned.append(f"{rel}（{size / 1e6:.1f} MB，偏大）")

        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pat in PATTERNS:
            m = pat.search(text)
            if m:
                line_no = text[: m.start()].count("\n") + 1
                problems.append((rel, f"{label} 第 {line_no} 行", m.group(0)[:60]))

    print(f"扫描 {len(files)} 个待提交文件\n")

    if blocked:
        print("[阻断]")
        for b in blocked:
            print("  -", b)
    if problems:
        print("[敏感内容]")
        for rel, where, snippet in problems:
            print(f"  - {rel} :: {where} :: {snippet}")
    if warned:
        print("[提醒]")
        for w in warned:
            print("  -", w)
    if not (blocked or problems or warned):
        print("通过：无数据文件、无凭据、无绝对路径、无学号。")

    sys.exit(1 if (blocked or problems) else 0)


if __name__ == "__main__":
    main()
