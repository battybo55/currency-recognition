# results/ 目录约定

本目录只保存**可复现的汇总结果**，不保存原始数据与中间图。

```
results/
├─ summary.md          # 报告用主表，每个数字注明数据与划分
├─ metrics/
│  ├─ e0_baseline.csv
│  ├─ e2_segmentation.csv
│  ├─ e4_features.csv
│  └─ e6_slices.csv
└─ figures/
   ├─ confusion_banknote.png
   ├─ confusion_coin.png
   └─ slices_heatmap.png
```

规则：

1. 每个 CSV 至少包含 `run_id, config, dataset, split, metric, value`，可加 `seed`。
2. `summary.md` 的数字必须由 CSV 汇总生成，不允许手填。
3. 图表若含票面完整图案，按合规要求处理（单面、长边 ≤ 640 px、关键信息打码），否则改用灰度边缘图或混淆矩阵。
