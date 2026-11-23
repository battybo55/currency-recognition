"""训练并评估像素级前景分类器（方案③）。

弱标注策略：
    正样本 —— 现有检测框内、向内腐蚀 7px 的像素（避免边界混样）；
    负样本 —— 距任一检测框 >10px、且颜色接近背景色的像素（确保是桌面而非目标边缘）。

评估：
    按图划分（前 5 张训练、后 5 张测试），只比"张数是否完全正确"。

用法：
    python scripts/train_pixelclf.py --model models/pixel_rf.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.background import detect_background, raw_residual  # noqa: E402
from src.io_utils import imread_unicode  # noqa: E402
from src.pipeline import analyze, load_config  # noqa: E402
from src.pixelclf import collect_samples, predict_map  # noqa: E402
from src.preprocess import resize_long_side  # noqa: E402
from src.scene import SceneConfig, classify_component, extract_objects  # noqa: E402

SELF = ROOT / "data/raw/self"
GT = {"s06": 3, "s07": 3, "s08": 3, "s09": 4, "s10": 4,
      "s11": 4, "s12": 4, "s13": 4, "s14": 4, "s15": 4}


def _bounts(result: dict) -> list[tuple[int, int, int, int]]:
    out = []
    for b in result["banknotes"]:
        xs = [p[0] for p in b["quad"]]
        ys = [p[1] for p in b["quad"]]
        out.append((int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/pixel_rf.joblib")
    ap.add_argument("--max-each", type=int, default=6000)
    ap.add_argument("--masks", default="data/raw/annotations/masks",
                    help="人工标注掩膜目录（由 label_tool.py 生成）；存在则优先使用")
    args = ap.parse_args()

    cfg = load_config()
    names = sorted(GT)
    train_names, test_names = names[:5], names[5:]

    Xs, ys = [], []
    for n in train_names:
        p = SELF / f"{n}.jpg"
        img = resize_long_side(imread_unicode(p), 1000)
        h, w = img.shape[:2]
        # 优先用人工标注掩膜；没有时退回检测掩膜（弱标签，仅供冒烟）
        manual = ROOT / args.masks / f"{n}.png"
        if manual.exists():
            raw = cv2.imdecode(np.fromfile(str(manual), np.uint8), cv2.IMREAD_GRAYSCALE)
            mask_pos = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_pos = cv2.erode(mask_pos, np.ones((5, 5), np.uint8))
            src = "人工"
        else:
            objs, _, _ = extract_objects(img, SceneConfig())
            mask_pos = np.zeros((h, w), np.uint8)
            for o in objs:
                if o["kind"] == "note":
                    mask_pos = cv2.bitwise_or(mask_pos, o["mask"])
            mask_pos = cv2.erode(mask_pos, np.ones((11, 11), np.uint8))
            src = "弱标签"
        far = cv2.dilate(mask_pos, np.ones((31, 31), np.uint8)) == 0

        model = detect_background(img)
        resid = raw_residual(img, model)
        neg = (far & (resid < 15)).astype(np.uint8) * 255

        X, y = collect_samples(img, mask_pos, neg, model, max_each=args.max_each)
        Xs.append(X)
        ys.append(y)
        print(f"  {n}: [{src}] 正样本面积 {mask_pos.mean() / 255:.1%} 负样本面积 {neg.mean() / 255:.1%} 采样 {len(y)}")

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    print(f"训练样本 {len(y)}（正 {int(y.sum())}）")

    clf = RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=20,
                                 n_jobs=-1, random_state=42, class_weight="balanced")
    clf.fit(X, y)
    print("训练完成")

    out = ROOT / args.model
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out)

    print("\n测试集（未参与训练）")
    print(f"{'图':<8}{'分类器':>8}{'现有引擎':>10}{'真值':>6}")
    ok_new = ok_old = 0
    for n in test_names:
        p = SELF / f"{n}.jpg"
        img = resize_long_side(imread_unicode(p), 1000)
        pmap = predict_map(clf, img, detect_background(img))
        mask = (pmap > 128).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        scfg = SceneConfig()
        notes = 0
        for i in range(1, n_comp):
            if stats[i, 4] / mask.size < 0.002:
                continue
            comp = (labels == i).astype(np.uint8) * 255
            cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            c = max(cs, key=cv2.contourArea)
            if classify_component(c, img.shape[:2], scfg)["kind"] == "note":
                notes += 1
        old = len(analyze(p, cfg, engine="scene")[0]["banknotes"])
        ok_new += notes == GT[n]
        ok_old += old == GT[n]
        print(f"{n:<8}{notes:>8}{old:>10}{GT[n]:>6}")

    print(f"\n张数正确：像素分类器 {ok_new}/{len(test_names)}，现有引擎 {ok_old}/{len(test_names)}")
    print(f"模型已写出 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
