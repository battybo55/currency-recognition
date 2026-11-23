# Changelog

本文件记录项目的功能性变更。版本号遵循语义化版本。

## [Unreleased]

### Changed
- `LICENSE` 改为纯 MIT 文本。原先附在末尾的范围声明会让 GitHub 识别为 `NOASSERTION`；该声明已并入 README 第 11 节。

### Planned
- 纸币分割与四点透视校正（M3）
- 硬币霍夫圆 / 分水岭双路线（M3）
- 特征提取与分类基线 E0–E5（M4）
- 消融矩阵与鲁棒性切片 E6（M5）
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
