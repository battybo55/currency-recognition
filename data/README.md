# data/ 目录说明

**本目录不存放任何图像。** 目录本身只保留这份说明，其余内容全部被 `.gitignore` 排除。

## 为什么

1. **许可不可传递**：公开数据集多为 CC0 / CC BY，但部分数据（尤其是爬取所得）存在权利瑕疵，不能再分发。
2. **合规**：人民币票面图样的使用受《人民币图样使用管理办法》约束，公开仓库不在教学科研免审批的直接适用范围内。
3. **体积**：GitHub 单文件超过 100 MB 会被拒收，数据集动辄数百 MB 至数 GB。

## 建议的本地布局

```
data/
├─ README.md              # 本文件（入库）
├─ raw/                   # 原始数据，不入库
│  ├─ count_coins/
│  ├─ indian/
│  ├─ world_coin/
│  ├─ rmb_local/          # 仅本地冒烟，禁止外传
│  └─ self/               # 自采图，按 session 分目录
├─ annotations/           # 标注与划分，splits.json 需入库
│  ├─ labels.csv
│  ├─ coin_points.csv
│  └─ splits.json
└─ processed/             # 中间结果，可重建，不入库
```

## 获取方式

见 `docs/data_sources.csv` 的 `url` 与 `fetch_note` 字段。常用命令：

```bash
# Kaggle 公开数据集：实测可匿名下载，无需 Token
curl -L -o data/raw/count_coins.zip \
  "https://www.kaggle.com/api/v1/datasets/download/balabaskar/count-coins-image-dataset"

# 走 CLI 时才需要 ~/.kaggle/kaggle.json 或 kaggle auth login
kaggle datasets download -d balabaskar/count-coins-image-dataset -p data/raw/count_coins --unzip

# Mendeley 直链会 403，需从页面下载
# https://data.mendeley.com/datasets/xn44yz596n
```

## 提交前自检

```bash
git status --porcelain data/     # 应只出现 data/README.md
git check-ignore -v data/raw/self/IMG_0001.jpg
```

若 `git status` 中出现任何图像文件，先修 `.gitignore`，不要提交。
