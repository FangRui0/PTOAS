# GitCode 代码提交与合入指南（cann/pto-as）

本文面向向 GitCode `cann/pto-as` 提交代码的贡献者，介绍从 Fork、开发、创建合并请求（Pull Request，简称 PR）到自动合入 `master` 的完整流程。

## 1. 合入 PR 的三个门禁

PR 要由合入机器人自动合入目标分支，通常必须同时满足以下条件：

| 门禁 | 通过表现 | 含义 |
| --- | --- | --- |
| CLA 签署 | `cla/yes`（绿色） | PR 中每个 commit 的 Author 和 Committer 邮箱都能匹配已签署的 CLA。 |
| CI 流水线 | `ci-pipeline-passed` | 仓库配置的门禁流水线全部通过。 |
| 评审加分 | 评审加分累计至少 2 分 | 具备评审权限的贡献者完成评审并为 PR 加分。 |

三个门禁缺一不可。具体标签名称、评审计分方式和机器人命令以 `cann/pto-as` 的 PR 页面及机器人提示为准；标签全绿后通常由合入机器人自动合并，不需要手动点击 Merge。

## 2. 最重要的规则：追加 commit 会重新触发门禁

PR 创建后，只要再次向同一分支 push 新 commit，通常都会发生以下变化：

1. **CLA 重新校验**：新 commit 的 Author / Committer 邮箱也必须在 CLA 签署名单中。
2. **CI 重新运行**：旧的 `ci-pipeline-passed` 结果不再代表最新提交，需要等待新流水线通过。
3. **评审加分失效或重置**：代码内容变化后，已有的评审加分通常需要重新获取。

因此，每次 push 后都重新确认“CLA + CI + 2 个加分”三个门禁。尽量在创建 PR 前整理好提交，减少反复追加 commit。

## 3. 开工前准备

### 3.1 签署 CLA

首次贡献前，先在社区提供的 CLA 页面签署个人 CLA 或企业 CLA。签署时登记的邮箱会用于匹配 commit，务必记住准确地址。

### 3.2 统一 Git 提交邮箱

在仓库中检查和设置提交身份：

```bash
git config user.name "你的名字"
git config user.email "你在 CLA 登记的邮箱@example.com"
git config --get user.name
git config --get user.email
```

CLA 机器人按 commit 中记录的邮箱匹配签署记录，不是按 GitCode 用户名匹配。换机器、换仓库或使用不同 Git 配置后，应再次检查邮箱。

## 4. 标准提交流程

### 4.1 Fork 并克隆仓库

在 GitCode 上 Fork `cann/pto-as` 到自己的命名空间，例如 `你的用户名/pto-as`，然后执行：

```bash
git clone https://gitcode.com/<你的用户名>/pto-as.git
cd pto-as
git remote add upstream https://gitcode.com/cann/pto-as.git
git remote -v
```

`origin` 指向自己的 Fork，`upstream` 指向官方仓库。后续从 `upstream` 同步主干，从 `origin` 推送开发分支。

### 4.2 基于最新主干创建分支

```bash
git fetch upstream master
git switch -c my-feature upstream/master
```

如果 Git 版本较旧，也可以使用 `git checkout -b my-feature upstream/master`。

### 4.3 开发并提交

只暂存本次改动相关的文件：

```bash
git status
git add <改动的文件>
git commit -m "fix(pto): 修复 xxx"
```

建议 commit 首行简洁（不超过约 70 个字符），正文说明为什么修改以及如何验证，不要只罗列文件变化。提交前可检查最终内容：

```bash
git diff --cached
git show --stat --oneline HEAD
```

### 4.4 推送并创建 PR

```bash
git push -u origin my-feature
```

在 GitCode 上创建从 `你的 Fork:my-feature` 到 `cann/pto-as:master` 的 PR。标题应简洁准确，描述至少包含：

- 背景和要解决的问题；
- 主要改动及影响范围；
- 本地或 CI 测试情况；
- 已知限制（如有）。

## 5. 创建 PR 后检查门禁

### 5.1 CLA

CLA 机器人会自动检查并在失败时给出提示和签署链接。签署完成后，按机器人评论中的说明重新检查；社区常见命令是：

```text
/check-cla
```

如果邮箱不匹配，先按第 6 节修正 commit，再重新 push。

### 5.2 CI 流水线

```text
/compile
```

会触发ci流水，等待 PR 页面出现 `ci-pipeline-passed`。失败时打开流水线日志，定位具体步骤，在本地修复后 push 新 commit。确认只是基础设施偶发抖动时, 可在“检查”页面点击重试，会只跑失败的流水项。

### 5.3 评审加分

邀请具备评审权限的同事 review 并加分，直到 PR 页面显示累计至少 2 个有效加分。常见命令包括：

```text
/lgtm
/approve
```

这些命令是否计入自动合入条件取决于本仓库机器人配置，以 PR 中的实际状态为准。

## 6. 常见问题排查

### Q1：`cla/yes` 一直是红色怎么办？

先检查 PR 中每个 commit 的 Author 和 Committer 邮箱：

```bash
git log upstream/master..HEAD \
  --format='%h %an <%ae> | committer: %cn <%ce> %s'
```

确认这些邮箱与 CLA 登记邮箱完全一致（包括大小写和域名）。如果某个 commit 邮箱错误，需要改写该 commit 的历史。常见做法是对开发分支执行交互式 rebase，逐个编辑相关 commit，然后运行：

```bash
git commit --amend --reset-author --no-edit
git rebase --continue
git push --force-with-lease origin my-feature
```

`--reset-author` 会把 Author 更新为当前 Git 身份。改写他人署名的 commit 前必须取得作者同意；不要为了通过 CLA 擅自替换他人的身份信息。

### Q2：CI 失败怎么办？

先看失败 job 的日志和失败 commit，区分代码问题、依赖问题与基础设施抖动。代码问题应本地修复并 push；只有在确认是偶发环境问题时才使用机器人提供的重跑命令。

### Q3：评审加分为什么突然没了？

追加 commit、rebase 或其他改变 PR 内容的操作可能使原有加分失效。等待 CI 更新后，重新邀请评审并确认 2 个有效加分已恢复。

### Q4：PR 提示与主干冲突或落后？

执行第 6 节的 fetch + rebase 流程，解决冲突并用 `--force-with-lease` 更新分支。更新后 CLA、CI 和评审加分都可能重新计算。

## 7. 机器人指令速查

以下是社区中常见的命令，最终以 PR 机器人实际支持的指令为准：

| 指令 | 常见作用 |
| --- | --- |
| `/lgtm` | 评审通过（Looks Good To Me） |
| `/approve` | 批准合入 |
| `/compile` | 触发 全量CI |
| `/check-cla` | 重新校验 CLA |
| `/close` | 关闭 PR |
| `/reopen` | 重新打开 PR |

## 8. 一句话总结

PR 要合入：`cla/yes`（每个 commit 的提交者邮箱都有 CLA）+ `ci-pipeline-passed` + 评审加分至少 2 分。每追加一次 commit，这三项都要重新满足。
