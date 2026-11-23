# 公开素材验证 与 GitHub 上线准备

> 配套：`00_课题研究方案.md`（技术方案）
> 版本：v0.1｜2026-11-23｜状态：清单已核，验证待执行

---

## 一、现有要求整理

| 你的要求 | 当前状态 | 落点 |
|---|---|---|
| 暂不自采，先用公开素材验证算法 | 已完成检索与**逐项许可判定**；验证协议已定，尚未下载执行 | 本文 §二.1 |
| 检查上传 GitHub 前需准备的内容 | 已清点 10 项必需件，识别出 12 个易漏点 | 本文 §二.2、§二.4 |
| 按「整理—补充—优先级」输出 | 本文即按此结构 | 全文 |

原方案中已覆盖、本次不重复的部分：研究目标、识别对象、预处理、特征与判别、实验矩阵 E0–E8、评价指标、8 周排期、难点与备选、14 个 GitHub 参考项目。见 `00_课题研究方案.md`。

**一句话结论**：技术方案可执行，真正的缺口不在算法，而在**数据合规**与**仓库工程化**——这两项若等到上传前才处理，返工成本最高。

---

## 二、遗漏项补充

### 2.1 公开素材：来源、许可、可用性判定

#### （1）候选清单（检索日 2026-11-23）

| # | 素材 | 内容与规模 | 许可（页面标注） | 含人民币 | 建议用途 | 核验状态 |
|---|---|---|---|---|---|---|
| 1 | [bat67/RMB-Dataset](https://github.com/bat67/RMB-Dataset) | 约 30 MB，1/5/10/20/50/100 元照片，按面额分目录 | **无 LICENSE 文件** | 是 | 仅本地冒烟，**不得入库** | 已核：根目录仅 README、.gitignore、RMBDataset |
| 2 | [a252937166/identifyRMB](https://github.com/a252937166/identifyRMB) | 面值与冠字号识别，12 G 训练 / 4 G 测试，走网盘 | **未声明许可** | 是 | 仅作方法参考，不下载进仓库 | 已核：README 只有网盘/Drive 链接 |
| 3 | [央行人民币图样库](https://www.pbc.gov.cn/huobijinyinju/147948/3917987/index.html)（转 [cbpm.cn/cn/rmbpic](http://www.cbpm.cn/cn/rmbpic)） | 各套人民币、普通纪念币官方图样 | 法规许可：教学科研等目的**可免审批**（第十五条），须守第九条 | 是 | 单张模板、演示图 | 已核：人行页面原文 |
| 4 | [Kaggle Indian Currency Dataset](https://www.kaggle.com/datasets/vishwasjraval/indian-currency-dataset) | 18.6 k 文件，正拍/旋转各 10 类 | CC0 | 否 | **分类流程验证 + 旋转鲁棒性** | 已核 |
| 5 | [Kaggle Afghani & PKR Banknotes](https://www.kaggle.com/datasets/musahassanjalili/afghani-and-pkr-banknotes-dataset) | 263 文件，13 类 | CC BY 4.0 | 否 | 小样本分类冒烟 | 已核 |
| 6 | [Mendeley Bangladeshi Currency](https://data.mendeley.com/datasets/xn44yz596n)（DOI 10.17632/xn44yz596n.2） | 100,000 张，256×256，10 类钞+币 | CC BY 4.0 | 否 | 小图分类、类别不平衡实验 | 已核（含 DOI） |
| 7 | [Kaggle Count Coins Image](https://www.kaggle.com/datasets/balabaskar/count-coins-image-dataset) | 多国硬币（含中国），附 `coins_count_values.csv` 真值 | CC0 | 否（含中国硬币） | **计数与分割主验证** | 已核 |
| 8 | [Kaggle Multi-Camera World Coin](https://www.kaggle.com/datasets/irfanhossain2025/multi-camera-world-coin-dataset) | 多机位同币，含严格防泄漏划分与元数据 | CC BY 4.0 | 否 | **划分方法参考**（按同币分组） | 已核 |
| 9 | [Kaggle U.S. Coins](https://www.kaggle.com/datasets/sergiosaharovskiy/uscoins) | 29,473 张，44/77 类 | CC0 | 否 | 细粒度分类压力测试 | 已核 |
| 10 | [Kaggle Swiss Coins](https://www.kaggle.com/datasets/ai-first/swisscoins) | 279 文件 | CC0（**二手页面标注**） | 否 | 快速冒烟 | 待核：需回 Kaggle 页面确认 |
| 11 | [UCI Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication) | 1,372 条 × 4 维小波特征，**非图像** | CC0 | 否 | 仅验证分类器，不能验证图像管线 | 已核 |
| 12 | [Roboflow coin 系列](https://universe.roboflow.com/aas-bdzur/coin-iumm0) | 约 552 张，YOLO 格式标注 | CC BY 4.0 | 否 | 检测框对照（本项目不做深度检测，价值有限） | 已核（单例） |

#### （2）许可判定规则（四条，先于一切下载）

- **R1 有明确 CC/OSI 许可**：可用，但须保留署名与 NOTICE，且**只在本地**参与计算。
- **R2 无 LICENSE 文件**：默认「保留所有权利」。即便公开可访问，也**不可再分发、不可入库**，只能本地研究使用。
- **R3 二手聚合站标注不算证据**：必须回到原始托管页（Kaggle / Mendeley / UCI）核对，截图存档。
- **R4 爬取类数据集有权利瑕疵**：#7 的描述明写「scraped from Google Image searches」，上传者的 CC0 声明并不能替原作者授权。此类数据**只能本地用，绝不能再分发**。

> 关键结论：**仓库永不托管第三方图像**。无论许可多宽松，都只提交「下载脚本 + 来源登记 + 指标数字」，不提交图片本体。这一条同时解决许可、体积、隐私三个问题。

#### （3）验证方式（V0–V6，可照做）

| 步骤 | 动作 | 产出 | 判据 |
|---|---|---|---|
| V0 登记 | 填 `docs/data_sources.csv`：id、名称、URL、许可、获取方式、样本数、用途、核验人、核验日期、证据截图 | 登记表 | 每条都能追溯到原始页面 |
| V1 完整性 | 记录文件数、总大小；压缩包记 SHA256 | `docs/validation_log.md` 首段 | 与页面声明量级一致（±10%） |
| V2 抽样目检 | 每类随机 20 张，查重复、错标、水印、非目标币种 | 抽样拼图 1 张 | 错标率 < 5% 才继续 |
| V3 跑通性 | **只跑预处理 + 分割**，不调参、不训练 | 12 张中间结果拼图 | 能稳定产出掩膜/四角即通过 |
| V4 基线数值 | 跑 E0（颜色直方图 + SVM），5 折 | `results/external/<id>/metrics.csv` | 记录 accuracy 与 macro-F1，**仅作参考** |
| V5 域差归因 | 写清外部集与目标域的差异（币种、色彩、版式、分辨率、背景） | 半页说明 | 明确哪些结论不可外推 |
| V6 落盘 | 指标、图、结论写入 `docs/validation_log.md` | 验证日志 | 第三人可复算 |

#### （4）结果记录模板

`docs/validation_log.md` 每数据集一节，固定字段：

```
dataset_id / 名称 / 许可 / 获取日期
样本数（声明 vs 实到）：      一致 / 偏差 x%
V2 抽样：抽 140 张，错标 3 张（2.1%）
V3 跑通：纸币四角定位成功 17/20；硬币计数 18/20 图正确
V4 基线：accuracy 0.83 / macro-F1 0.79（5 折均值±std）
V5 域差：币种不同、版式不同，颜色特征不可直接迁移
结论：可用（仅流程验证） / 仅参考 / 不可用
```

#### （5）人民币素材的合规边界（本方案最容易踩的坑）

依据《人民币图样使用管理办法》（中国人民银行令〔2019〕第 2 号）：

- **第三条**：「使用」指在宣传品、出版物或者其他商品上使用放大、缩小和同样大小的人民币图样。
- **第九条**：须单面使用、不得损害人民币形象、不得使公众误认为人民币、不得使人像与国徽变形，并按规定标注「图样」。
- **第十五条**：图书出版、教学研究、新闻媒体、文博机构等单位，出于**教学、学术研究**目的，**在宣传品、出版物上**使用**央行图样库公布**的图样，**可以不经审批**，但须遵守第九条并接受监督检查。
  → 注意适用对象是「宣传品、出版物」而非「任何公开网络仓库」。因此 GitHub 托管人民币原图**不能直接套用本条豁免**，这正是下文采用 L1 保守策略的原因。

据此给出三层处置，按保守程度递增：

| 层级 | 做法 | 适用 |
|---|---|---|
| L1（默认） | 仓库不含任何人民币原图；`data/` 整目录忽略；仅保留 `data/README.md` 说明获取方式 | 公开仓库 |
| L2（需展示时） | 单面、长边 ≤ 640 px、加「图样」字样、冠字号码与人像局部打码、注明「仅算法演示」 | README 效果图 |
| L3（报告/答辩） | 按第十五条以教学研究目的使用图样库图片，标注来源；正式发表前请导师或学校法务确认 | 课程报告 |

补充两点：**硬币不受图样办法约束**，但照片仍有拍摄者版权；**结果截图同样算展示**，混淆矩阵可以放，带完整币面的识别结果图要按 L2 处理。

#### （6）当前可立即执行的验证组合

| 目的 | 用哪份 | 理由 |
|---|---|---|
| 分割/计数管线跑通 | #7 Count Coins（CC0，带计数真值） | 唯一自带逐图真值、可直接算 MAE/F1 |
| 分类管线跑通 + 旋转鲁棒性 | #4 Indian Currency（CC0，正拍/旋转两套） | 类别齐、许可最干净 |
| 小样本与不平衡 | #6 Bangladeshi（CC BY 4.0，10 万张 256×256） | 规模大、分辨率统一，适合快速迭代 |
| 划分方法对照 | #8 Multi-Camera（CC BY 4.0） | 已按「同一实物」分组，可验证我们的防泄漏划分 |
| 人民币冒烟 | #1 或 #3（本地，不入库） | 只验证「颜色/版式先验是否成立」，不出现在任何公开产物里 |

**拉取命令**（全部落在 `data/raw/`，已被 `.gitignore` 覆盖）：

```bash
# Kaggle：公开数据集实测可匿名下载，无需 Token（见下方执行状态）
curl -L -o data/raw/count_coins.zip \
  "https://www.kaggle.com/api/v1/datasets/download/balabaskar/count-coins-image-dataset"

# 若走 CLI，则需要 ~/.kaggle/kaggle.json 或 kaggle auth login
kaggle datasets download -d balabaskar/count-coins-image-dataset -p data/raw/count_coins --unzip
kaggle datasets download -d vishwasjraval/indian-currency-dataset -p data/raw/indian --unzip
kaggle datasets download -d irfanhossain2025/multi-camera-world-coin-dataset -p data/raw/world_coin --unzip

# Mendeley：直接 curl 会 403（需要浏览器会话），改用页面下载
# https://data.mendeley.com/datasets/xn44yz596n

# 人民币：无许可证，仅本地冒烟，克隆后确认 data/ 已被忽略
git clone --depth 1 https://github.com/bat67/RMB-Dataset.git data/raw/rmb_local
```

**必须写进报告的限定**：外部集只能证明「管线正确」，不能证明「人民币面额判别性能」。域不同（币种、主色、版式、拍摄设备），跨域数字不可直接对比。

> **执行状态（2026-11-23 实测）**：Kaggle **公开数据集可匿名下载**，不需要 API Token——`https://www.kaggle.com/api/v1/datasets/download/<owner>/<slug>` 直接 302 到带签名的 GCS 链接；`kaggle` CLI 2.x 虽提示需要认证，但 HTTP 通道可用。
> 已下载：DS07 Count Coins（108 MB / 1,444 张，V1、V2 通过）、DS05 Afghani-PKR（65 MB / 263 张）、DS04 Indian（809 MB，下载中）。
> Mendeley 的 `public-api/zip` 直链返回 403（需要浏览器会话）；UCI 可直连。央行图样库仅网页浏览，不提供批量接口。

---

### 2.2 GitHub 仓库必备件清单

| 项 | 必需 | 内容要点 | 验收标准 |
|---|---|---|---|
| `README.md` | 必须 | 一句话定位、效果图（合规版）、环境、3 条命令跑通、目录、结果表、数据与许可声明、致谢 | 新人 10 分钟内能跑出第一张结果图 |
| `LICENSE` | 必须 | 代码 MIT 或 Apache-2.0；**明确声明不覆盖数据集与第三方素材** | GitHub 侧栏正确显示协议 |
| `.gitignore` | 必须 | 见 §2.3 | `git status` 无数据、无缓存、无 IDE 文件 |
| `requirements.txt` | 必须 | 锁版本；注明 Python 3.13（3.14 无 opencv wheel） | 干净环境 `pip install -r` 成功 |
| 运行步骤 | 必须 | 安装→下载/放置数据→运行→看结果，四步以内 | 照做可复现 |
| 测试结果 | 必须 | `results/summary.md`：实验、数据、划分、指标定义、数值 | 每个数字都能追到 CSV |
| `docs/data_sources.csv` | 必须 | 素材登记（许可证据） | 每条有原始链接与核验日期 |
| `docs/validation_log.md` | 必须 | V0–V6 记录 | 含失败样例，不只报好结果 |
| `CITATION.cff` | 推荐 | 作者、标题、版本、年份 | GitHub 显示「Cite this repository」 |
| `CHANGELOG.md` | 推荐 | 按版本记变更 | v0.1 起 |
| `docs/images/` | 推荐 | 效果图、流程图（合规处理） | 无原图泄漏 |
| `tests/test_smoke.py` | 推荐 | 用 1 张图跑通全链路 | `pytest` 30 秒内通过 |
| `.github/workflows/ci.yml` | 可选 | lint + smoke test | 徽章变绿 |
| `CONTRIBUTING.md` | 可选 | 课程项目可省 | — |

---

### 2.3 目录结构与 `.gitignore`

```
currency-recognition/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ requirements.txt
├─ .gitignore
├─ configs/                 # 每组实验一个 yaml
├─ src/
│  ├─ io_utils.py           # 中文路径安全的读写
│  ├─ preprocess.py
│  ├─ segment.py
│  ├─ features.py
│  ├─ classify.py
│  └─ evaluate.py
├─ scripts/
│  ├─ fetch_external.py     # 仅处理许可清晰的数据源
│  ├─ make_splits.py
│  └─ run_all.py
├─ data/README.md           # 只放说明，不放图
├─ docs/
│  ├─ data_sources.csv
│  ├─ validation_log.md
│  ├─ design.md
│  └─ images/
├─ results/
│  ├─ summary.md
│  ├─ metrics/*.csv
│  └─ figures/
└─ tests/test_smoke.py
```

`.gitignore` 最小可用集：

```
# 数据与中间产物
data/*
!data/README.md
processed/
augmented/
results/metrics/raw_*.csv

# 环境
.venv/
venv/
__pycache__/
*.py[cod]
.ipynb_checkpoints/

# 大型与临时
*.zip
*.7z
*.log
~$*
.DS_Store
Thumbs.db

# 本地路径与凭据
.env
*.local
.vscode/settings.json
```

---

### 2.4 容易遗漏的 12 项

1. **数据集不入库**：GitHub 单文件 >100 MB 直接拒收，>50 MB 会警告；数据一律走下载脚本 + Release 附件。
2. **EXIF 清除**：手机照片含 GPS 与设备型号，提交前统一剥离（`exiftool -all=` 或 Pillow 重写）。
3. **绝对路径泄漏**：代码里写死的本机路径必须清掉，全部改相对路径或配置项（`scripts/preflight.py` 会自动扫描）。
4. **许可覆盖范围**：MIT 只管代码，不能替数据授权；README 要单独写「本仓库不包含数据集」。
5. **第三方图表**：论文插图、Kaggle 示例图转载需署名，最好自己重绘。
6. **Notebook 输出清理**：`.ipynb` 里常残留本地路径、内存占用、报错栈；提交前 Clear All Outputs。
7. **依赖锁版本**：写明 Python 3.13（3.14 无 opencv-python wheel），opencv 走官方源而非清华源。
8. **中文路径与编码**：仓库内文件与目录一律 ASCII，避免跨平台与 CI 失败。
9. **可复现三件套**：固定种子、`splits.json` 入库、配置文件入库；缺一不可。
10. **README 数字要有口径**：写「macro-F1 0.86（6 类，自采 118 张测试集）」而不是「准确率 95%」。
11. **学校政策**：部分院校要求课程作业公开前报备或授权，先确认再推公开仓库。
12. **结果图合规**：含完整币面的截图按 §2.1(5) 的 L2 处理，否则只放灰度边缘图与混淆矩阵。

---

## 三、优先级建议

### P0：本周完成（约 1–2 天，决定项目能否安全公开）

| # | 事项 | 产出 | 预估 |
|---|---|---|---|
| 1 | 建仓库骨架：README 骨架、LICENSE(MIT)、`.gitignore`、`requirements.txt` | 可 `git init` 的空壳 | 2 h |
| 2 | 建 `docs/data_sources.csv`，登记 §2.1 中 3 份 CC0/CC BY 数据 | 合规证据链 | 1 h |
| 3 | 跑 V1–V3（下载 + 抽样 + 跑通性），**不调参** | 12 张中间结果拼图 + 日志 | 4 h |
| 4 | `data/` 全忽略 + EXIF 剥离脚本 + 路径清理 | 无泄漏的提交 | 1 h |

### P1：进入 W2 前（决定报告质量）

5. E0 基线 + `results/summary.md`（含指标定义与划分说明）
6. 人民币合规声明写入 README，结果图按 L2 处理
7. `CITATION.cff` + `CHANGELOG.md` + 依赖锁版本
8. 冻结 `splits.json` 并入库

### P2：有余力再做

9. CI（lint + smoke test）、README 徽章
10. 演示 GIF / 单页项目主页
11. `tests/` 覆盖到分割与计数两个关键函数

---

## 决议与后续（2026-11-23 已确认）

| 事项 | 决定 | 落地位置 |
|---|---|---|
| 代码许可证 | **MIT**，末尾附范围声明「不覆盖数据集与第三方素材」 | `LICENSE` |
| 人民币素材 | 接受保守策略：**仓库零人民币原图**；必要时按 L2 处理 | `README.md` §7、`data/README.md` |
| Kaggle 凭据 | 由用户提供 | `_private/kaggle.json` 或 `~/.kaggle/access_token` |

**P0 执行状态**：仓库骨架、素材登记表、`data/` 忽略规则与合规声明均已完成；仅 V1–V4 因缺凭据未执行。

### 仍待完成

1. 放入 Kaggle 凭据 → 执行 V1（下载与完整性）、V2（抽样目检）、V3（跑通性）。
2. V4 需先有 E0 代码，与 M3 一并推进。
3. 公开前跑一次泄漏自查：`git status`、EXIF、绝对路径、`_private/` 是否被忽略。
