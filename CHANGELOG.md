# Changelog

本文件记录项目的功能性变更。版本号遵循语义化版本。

## [Unreleased]

### Added
- `src/pipeline.py`：端到端分析入口，输出 README 约定的 JSON 结构
- `src/scene.py`：场景自适应前景提取（边框颜色模型 + 边缘线索融合 + 分水岭拆分），不依赖固定背景模板
- `src/denomination.py` 与 `scripts/calibrate_denomination.py`：色度原型的面额判别与标定流程
- `scripts/test_scene.py`：四种背景 × 三种尺度的背景鲁棒性测试
- `scripts/run_demo.py --engine scene`：木纹等纹理背景下的端到端路径
- `scripts/run_demo.py`：单图演示，可另存标注图
- `scripts/train_banknote.py`：面额分类器训练（当前用外部数据，模型不入库）
- 纸币—硬币互斥：落在纸币四边形内的圆形候选会被剔除
- `src/features.py`：Lab/HSV 颜色直方图与矩、LBP、GLCM、Gabor、DFT/DCT、四边形几何特征
- `src/classify.py`：标准化 + SVM/RF/kNN/LR 的统一构造
- `scripts/exp_banknote.py`：E0（整图）与 E2（四角校正）对照实验
- `scripts/exp_features.py`：E4 特征消融与 E5 分类器对比，分层 5 折交叉验证
- `scripts/viz_features.py`：PCA / t-SNE 可分性可视化
- `scripts/slice_coins.py`：按真值密度与图像尺度做误差切片

### Changed
- `LICENSE` 改为纯 MIT 文本。原先附在末尾的范围声明会让 GitHub 识别为 `NOASSERTION`；该声明已并入 README 第 11 节。
- `find_banknote_quad_v2` 改用矩形度 + 边缘掩膜兜底，定位成功率由 38.5% 提升到 68.5%。
- 纸币定位改为多候选（`find_banknote_quads` + 跨掩膜 IoU 去重），支持同框多张。
- 硬币剔除增加「可信纸币」守卫：长宽比 1.5–2.8 的四边形才用于排除，避免合并区域误删真硬币。

### Planned
- 自采人民币数据与标注（M2）
- E6 完整切片：光照 / 角度 / 重叠率
- 硬币面额分类（当前只有计数）
- 端到端金额准确率与混排图评测
- 课程报告与 v1.0 发布（M6）

## [0.1.0] - 2026-11-23

### Added
- 仓库骨架：README、MIT LICENSE、`.gitignore`、`requirements.txt`、`CITATION.cff`
- 技术方案 `00_课题研究方案.md`：目标、对象、数据协议、预处理、特征、E0–E8、指标、排期、风险
- 上线准备 `01_公开素材与GitHub上线准备.md`：素材许可判定、V0–V6 验证协议、合规边界、必备件清单
- 进度看板 `PROGRESS.md`
- 素材登记 `docs/data_sources.csv`（12 份，含核验状态）
- 验证记录模板 `docs/validation_log.md`
- 基线配置 `configs/baseline.yaml`

### Decided
- 代码许可证采用 MIT，明确不覆盖数据与第三方素材
- 采用「仓库零人民币原图」的保守合规策略
- 第三方图像一律不入库，只保留来源登记与指标数字
