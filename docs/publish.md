# 发布到 GitHub

本地仓库已就绪：`main` 分支、首个提交、标签 `v0.1.0`、工作区干净。
推送前先跑一次自检：

```bash
python scripts/preflight.py --all
```

返回 `通过` 再往下走。

---

## 路线 A：网页建库 + 命令行推送（推荐）

1. 打开 <https://github.com/new>
2. 填写：
   - **Repository name**：`currency-recognition`（建议，ASCII，简短）
   - **Visibility**：课程作业一般选 Public；若担心姓名公开选 Private
   - **不要**勾选 “Add a README / .gitignore / license”，本地已有，勾了会冲突
3. 创建后在本目录执行：

```bash
git remote add origin https://github.com/<你的用户名>/currency-recognition.git
git push -u origin main
git push origin v0.1.0
```

> 若网页把默认分支设成了 `master`，在仓库 Settings → Branches 改为 `main`，或推送后删除远端 `master`。

## 路线 B：GitHub CLI

本机当前未安装 `gh`。装好后一条命令建库并推送：

```bash
gh auth login
gh repo create currency-recognition --public --source=. --remote=origin --push
git push origin v0.1.0
```

---

## 首推之后检查五件事

| 项 | 期望 |
|---|---|
| 仓库体积 | 约 250 KB，不含 `data/` 与 `_private/` |
| LICENSE | 侧栏显示 “MIT license” |
| README | 表格、代码块、相对链接均正常渲染 |
| Actions | `check` 工作流变绿（语法 + 自检） |
| 文件数 | 24 个左右；`git ls-files | wc -l` 与网页一致 |

---

## 常见问题

**`error: remote origin already exists`**
```bash
git remote set-url origin https://github.com/<用户名>/<仓库>.git
```

**推送被拒（non-fast-forward）**
远端初始化过 README，先合并再推：
```bash
git pull --rebase origin main
git push -u origin main
```

**误把数据推上去了**
不要用 `git rm --cached` 就以为删干净了——历史里仍有。二选一：
- 仓库还没人克隆：删除远端仓库重建，修好 `.gitignore` 再推；
- 已有历史：用 `git filter-repo --path data --invert-paths` 重写后再强推（会影响协作者）。

**中文文件名在网页上显示为 `%E8%AF%BE...`**
正常现象，GitHub 会 URL 编码。若希望链接更干净，可在后续版本把两篇中文文档改名为 `docs/design.md`、`docs/github-readiness.md`，并同步更新 README 内链。

---

## 后续版本

```bash
git add -A
python scripts/preflight.py
git commit -m "feat: 多尺度硬币检测"
git tag -a v0.2.0 -m "v0.2.0 多尺度检测"
git push origin main --tags
```
