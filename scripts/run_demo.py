"""单图端到端演示：输出 JSON 契约，可另存标注图。

用法：
    python scripts/run_demo.py --image data/raw/_demo/sample.jpg
    python scripts/run_demo.py --image ... --model models/demo_indian.joblib --draw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import imwrite_unicode  # noqa: E402
from src.denomination import load_config as load_denom  # noqa: E402
from src.pipeline import analyze, load_config  # noqa: E402

EVID = ROOT / "data/raw/_evidence/demo"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--denom", default="", help="面额原型配置，如 configs/denomination_indian.yaml")
    ap.add_argument("--rmb", action="store_true", help="使用人民币规则面额模型")
    ap.add_argument("--engine", default="legacy", choices=["legacy", "scene"])
    ap.add_argument("--draw", action="store_true", help="在 data/raw/_evidence/demo 下保存标注图")
    args = ap.parse_args()

    model = joblib.load(ROOT / args.model) if args.model else None
    if args.rmb:
        from src.rmb import RMBConfig
        dcfg = RMBConfig()
    elif args.denom:
        dcfg = load_denom(ROOT / args.denom)
    else:
        dcfg = None
    cfg = load_config()
    result, overlay = analyze(
        args.image, cfg, banknote_model=model, draw=args.draw,
        engine=args.engine, denom_cfg=dcfg,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.draw and overlay is not None:
        out = EVID / (Path(args.image).stem + "_demo.png")
        imwrite_unicode(out, overlay)
        print(f"标注图：{out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
