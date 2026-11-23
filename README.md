# 纸币与硬币识别 · Banknote & Coin Recognition

> 数字图像处理课程设计｜基于经典算子的可解释管线
> 当前版本：**v0.1.0（脚手架）** —— 算法尚未实现，仓库结构、数据合规与评测口径已就位

一句话：输入一张手机俯拍图，输出图中纸币与硬币的**面额、数量与合计金额**，并用消融实验说明每一步预处理与每一族特征的实际贡献。

---

## 1. 项目状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| M0 方案与准备 | 技术方案、开源项目调研、数据合规清单、仓库骨架 | 已完成 |
| M1 外部素材验证 | 公开数据跑通性验证（V0–V5） | 进行中 |
| M2 采集与标注 | 自采 550+ 张、半自动标注、分组划分 | 未开始 |
| M3 分割与校正 | 纸币四角定位 + 透视校正；硬币霍夫圆 / 分水岭 | 未开始 |
| M4 特征与分类 | 颜色 / LBP / GLCM / Gabor / 几何 + SVM/RF/kNN | 未开始 |
| M5 消融与切片 | E1–E6 单变量实验、鲁棒性切片 | 未开始 |
| M6 报告与发布 | 误差分析、课程报告、v1.0 发布 | 未开始 |

进度看板见 [`PROGRESS.md`](PROGRESS.md)；技术方案见 [`00_课题研究方案.md`](00_课题研究方案.md)。

## 2. 能做什么 / 不做什么

**做**：第五套人民币纸币 6 种面额（1/5/10/20/50/100 元）的正面识别；1 角 / 5 角 / 1 元硬币的检测、计数与面额判别；单图多目标的金额合计。

**不做**：假币与残缺币鉴别（单张可见光图不足以支撑结论）、外币与纪念币、视频实时流、以端到端深度学习为主线（CNN 仅作对照实验）。

## 3. 技术路线

```
输入图 → 公共预处理（白平衡 / 去噪 / 光照校正 / HSV-Lab）
       ├─ 纸币：二值化 → 形态学 → 四角定位 → 透视校正（512×256）
       └─ 硬币：霍夫圆 或 分水岭 → ROI → warpPolar 齿纹展开
     → 统一特征（颜色直方图 / LBP / GLCM / Gabor / 半径圆度 / DFT-DCT）
     → 规则先验 + SVM / RF / kNN
     → 面额 · 数量 · 合计金额 · 可视化叠加图
```

## 4. 目录结构

```
.
├─ README.md
├─ LICENSE                 # MIT，仅覆盖代码与文档
├─ CITATION.cff
├─ CHANGELOG.md
├─ PROGRESS.md             # 进度看板
├─ requirements.txt
├─ configs/                # 每组实验一个 yaml
├─ src/                    # 预处理 / 分割 / 特征 / 分类 / 评测
├─ scripts/                # 验证与诊断入口（validate_v3 / diagnose_v3）
├─ data/                   # 仅 README；图像不入库
├─ docs/
│  ├─ data_sources.csv     # 素材登记与许可证据
│  └─ validation_log.md    # V0–V5 验证记录
└─ results/                # 指标 CSV 与图表（待生成）
```

## 5. 环境

- Python **3.13**（3.14 无 `opencv-python` wheel，装不上）
- 依赖见 [`requirements.txt`](requirements.txt)，全部可在 CPU 上运行
- Windows 下读取含中文的路径需用 `cv2.imdecode(np.fromfile(...))`，不要直接 `cv2.imread`

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## 6. 快速开始

> 算法模块尚未落地，以下命令在 M3 完成后可用；当前仅完成环境与数据结构。

```bash
# 1) 将数据放入 data/raw/（见 data/README.md），无需提交
# 2) 生成固定划分
python scripts/make_splits.py --config configs/baseline.yaml
# 3) 跑单图端到端
python -m src.pipeline --image data/raw/demo.jpg --config configs/baseline.yaml
# 4) 复现实验表
python scripts/run_all.py --config configs/baseline.yaml
```

## 7. 数据与合规（重要）

**本仓库不托管任何图像数据。** 包括自采图、公开数据集、以及任何人民币票面图片。原因有三：许可不可传递、仓库体积限制、以及人民币图样的使用规范。

- 第三方素材的来源、许可与核验状态见 [`docs/data_sources.csv`](docs/data_sources.csv)
- 验证过程与结果见 [`docs/validation_log.md`](docs/validation_log.md)
- 获取方式见 [`data/README.md`](data/README.md)

关于人民币图样：《人民币图样使用管理办法》第十五条对教学、学术研究的免审批，限定为**在宣传品、出版物上**使用**央行图样库**公布的图样，并须遵守第九条（单面、不失真、不使人误认为真币）。公开代码仓库不在该条的直接适用范围内，因此本项目采取保守策略：**仓库零人民币原图**；确需展示时，结果为单面、长边不超过 640 px、关键信息打码的示意图。

## 8. 结果

> 待 M4–M5 完成后填写。每个数字必须能追溯到 `results/metrics/*.csv`，并注明数据集、样本量与划分方式。不接受无口径的「准确率 xx%」。
> 外部数据集的结果仅用于验证流程可运行，**与人民币目标域不可直接比较**。

| 实验 | 数据 | 指标 | 数值 | 备注 |
|---|---|---|---|---|
| V3 流程验证（硬币） | DS07 Count Coins / 1,444 张 | 计数 F1（霍夫 / 分水岭） | 0.218 / 0.282 | 默认参数、未调参；仅证明管线可运行 |
| V3 流程验证（纸币） | DS04 Indian / 60 张 | 四角定位返回率 | 100%（60/60） | 目检 20 张中约 3 张框错，多目标场景待改造 |
| M3 硬币计数（多尺度 + 边缘校验） | DS07 / held-out 213 张 | 计数 F1 | **0.811**（P 0.818 / R 0.805） | 阈值仅在 dev 188 张上选定 |
| E0 纸币基线（整图 + 颜色 + LBP） | DS04 Indian / 200 张，7:3 划分 | accuracy / macro-F1 | **0.964 / 0.960** | 无分割、无校正 |
| E2 纸币（四角校正 + 颜色 + LBP） | 同上 | accuracy / macro-F1 | 0.946 / 0.940 | 定位成功 68.5%，失败回退整图 |
| E2 纸币分割对比 | 待填 | macro-F1 | 待填 | Otsu / 自适应 / Canny |
| E4 特征消融 | DS04 Indian / 400 张，5 折 | accuracy | 颜色 0.975；颜色+四族 0.965 | 累加无增益，颜色已近饱和 |
| E5 分类器对比 | 同上，203 维 | accuracy | LR 0.980 / RF 0.975 / SVM 0.965 / kNN 0.948 | 差异小于特征族差异 |
| E6 鲁棒性切片 | 待填 | macro-F1 | 待填 | 按光照/角度/重叠率 |
| 端到端金额 | 待填 | 完全正确率 | 待填 | 钞币混排 |

## 9. 可复现约定

1. 固定随机种子（默认 42），写入 `configs/`；
2. 数据划分写入 `data/annotations/splits.json` 并入库，按拍摄场次分组，禁止按单张随机拆分；
3. 所有指标由 `results/metrics/*.csv` 汇总生成，不手填；
4. 增强只作用于训练集，验证/测试集保持原图。

## 10. 路线图

- [x] v0.1.0 仓库骨架、许可与合规框架
- [ ] v0.2.0 外部数据跑通性验证（V1–V4）
- [ ] v0.3.0 纸币分割 + 透视校正
- [ ] v0.4.0 硬币检测 + 计数
- [ ] v0.5.0 特征与分类基线（E0–E5）
- [ ] v0.6.0 消融与鲁棒性切片（E6）
- [ ] v1.0.0 课程报告与完整复现

## 11. 许可与引用

代码与文档采用 [MIT License](LICENSE)。该许可**不覆盖**任何第三方数据集、票币图像或人民币图样；本仓库不分发任何图像数据，获取方式见 [`data/README.md`](data/README.md)。引用信息见 [`CITATION.cff`](CITATION.cff)。

## 12. 发布

本地已是 git 仓库（`main` 分支，标签 `v0.1.0`）。推送步骤与首推检查见 [`docs/publish.md`](docs/publish.md)；提交前先跑自检：

```bash
python scripts/preflight.py --all
```

CI 在每次 push 时执行语法检查与同一套自检（`.github/workflows/ci.yml`）。

## 13. 致谢

早期调研参考了 `microsoft/banknote-net`、`igortopcin/banknote-recog`、`djeada/Coin-Detection` 等开源项目，具体条目与借鉴点见技术方案第 9 节。
