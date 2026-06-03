# 0010 — preprocess scope 从项目级 download 下沉到 version 级 train

**状态**：Proposed
**日期**：2026-06-03
**决策者**：@WalkingMeatAxolotl
**Supersedes**：[ADR 0004 — 预处理状态用单 manifest 替代「双 bucket + per-image sidecar」](0004-preprocess-manifest.md)（含 Addendum 1）

> **维护约定**：本 ADR 承袭 [`docs/design/preprocess-train-scope-plan.md`](../design/preprocess-train-scope-plan.md) 的实施计划（含 30 条决策点 D1-D30）。本 ADR 只固化"选哪个 / 否决谁 / 为什么"，落地细节（file:line / PR 切片 / 测试 case）见 plan。

## 背景

ADR 0004 把 preprocess 状态定型为「项目级单 manifest + 隐式 original + 双 bucket resolver」，工作流是：

```
download → preprocess（项目级）→ curate（选入 version 的 train/）→ tag → train
```

beta 用户使用后投票出四个真实痛点（详 plan §1 与 `tmp/preprocess-redesign-user-feedback.md`）：

1. **前置时间浪费**：booru scrape 来的图质量参差不齐，最终只用很小一部分。用户需要对**全集每张图**逐一操作（放大要等 / 裁剪要跳过），其中大部分图不会进 train
2. **统计指标无意义**：分辨率分布 / 宽高比分布基于 `download/` 全集计算，不是最终 train 集——对训练决策没有指导价值
3. **智能聚类失效**：聚类的目标是统一 ARB 桶，但聚类在 download 全集做完后下游的 curate 阶段还要再筛，桶又乱了
4. **scope 错位**：用户心智上"我想训练的图"是 train 集合；preprocess 在 download 全集上做，跟用户心智不对齐

ADR 0004 选择项目级 scope 的核心论据（"upscale 几百张是真贵，跨版本复用预处理结果远比每版独立预处理更符合实际工作流"）被两个事实变化推翻：

- ADR 0007 落地后 `create_version(fork_from=...)` + `_copytree("train")` 已经支持**整树 fork**，train/ 含已 upscale 产物会自然跟随到子 version，不需要项目级缓存来实现跨版本复用
- 用户调研：「新版本绝大多数从上一个 version 复制，甚至只改参数不改训练集」——v2 完全重做 preprocess 这种最坏情况几乎不会发生

## 候选方案

### A — 维持 ADR 0004 现状（项目级 preprocess/）

- 优点：0 工时
- 缺点：四个痛点不解决；统计 / 聚类基于错误集合是**正确性 bug**而不是 UX 缺陷
- **否决**

### B — UI filter 折中（Preprocess 页加"只看 train"过滤）

R1/R2 三方审阅一致推荐过这条（详 `tmp/preprocess-redesign-r{1,2}-*.md`）：保留项目级磁盘结构，只在 UI 加 filter 让用户视图聚焦到 train 集合。

- 优点：成本最低（~2.5d）；不动 ADR 0004 / 0007 / 0008
- 缺点：**只换视图，不解任何真痛点**：
  - 前置时间已经花掉了（filter 是事后过滤）
  - 统计 / 聚类底层数据集没变（仍是 download 全集）
  - 用户花在"不会用的图"上的操作时间不可挽回
- **否决**——三方 R3 复审一致认定 filter 0/4 真解（详 R3-ux §4）

### C — preprocess scope 下沉到 version 级 train/（**采纳**）

工作流变为：

```
download → curate（选入 train/）→ preprocess（在 train/ 上原地处理）→ tag → train
```

数据模型核心改变：

- 项目级 `preprocess/` 目录**保留不删**（作 fallback 重建源）但不再是新写入目标
- 新增 per-version manifest：`versions/{label}/train/manifest.json`
- preprocess 产物**直接落 `train/{name}`**，跟训练 bytes 同位
- 跨版本复用通过 ADR 0007 现有 fork 机制（`_copytree("train")`）自动实现
- 老项目通过 `ensure_train_manifest()` 隐式 lazy 重建，**零用户感知**

## 决策

选**方案 C**。

### Manifest schema v2

`versions/{label}/train/manifest.json`：

```json
{
  "version": 2,
  "images": {
    "X.png":    { "origin": "X.jpg", "mtime": 1731000000, "size": 1234567 },
    "Y_c0.png": { "origin": "Y.jpg", "mtime": ..., "size": ... },
    "Y_c1.png": { "origin": "Y.jpg", "mtime": ..., "size": ... }
  }
}
```

- key = `train/` 下产物文件名（含扩展名）
- `origin` = 对应 `download/` 原图文件名，用于 restore 反查
- `mtime / size` 用于检测外部修改
- **状态从字段差异隐含推断**——不存 `kind / source / model / scale / action / state` 等过程信息（承袭 ADR 0004 极简精神 + MEMORY `feedback_preprocess_data_model_simple`）

### `ensure_train_manifest()` 隐式 fallback 重建

老项目里 train/ 已经有 preprocess 产物（curate 阶段已复制进去），唯一丢失的是 train ↔ download 的 origin 关系。**fallback 函数**在所有 manifest read 入口防御性调用：

1. 目标 manifest 存在 → 直接返回（O(1) stat）
2. 不存在 + 老 `preprocess/manifest.json` 存在 → 按 train/ 实际文件名匹配老 entry origin 重建
3. 老 manifest 也不存在 → 写空 manifest

这条**完全不需要显式迁移脚本 / UI 弹窗 / 用户决策**——也是用户主动提出的方案，是本 ADR 跟 R3 三方汇总报告（v1 推荐显式迁移）的最大差异。

### Resolver 单点消亡

ADR 0004 的 `resolve(name) → Path` resolver 是为消除"双 bucket fallback（download/ vs preprocess/）"而设计的中心抽象。新模型下 train/ 是 self-contained：thumbnail / curation / tagging / training materialize 全部直接读 `train/{name}`，没有歧义。`manifest.py:resolve` / `resolve_origin` 删除。仍保留 `entry_origin` / `restore`（按 manifest 反查 download/ 做复原）。

### 复原（restore）语义

`restore(name)` = 从 `download/{entry.origin}` 复制覆盖回 `train/{name}`。

- `download/{origin}` 缺失 → 复原失败，UI 显式提示具体图 + 提供三选项 `[拖入替换 / 保留处理后版本 / 从 train 移除]`
- **不**做 per-version `.backup/` 隐藏备份——违反"download 唯一备份"用户偏好（MEMORY `feedback_preprocess_data_model_simple`）
- ADR 0004 §215-219 "外部删 download 不主动 reconcile" 原则继承

### Phase 状态机加 `preprocessing`（ADR 0007 amendment）

跟随本 ADR 给 ADR 0007 加 Addendum：

```python
VersionPhase.ORDER = (curating, preprocessing, tagging, editing, regularizing, ready)
VersionPhase.SKIPPABLE = {preprocessing, regularizing}
```

`check_preprocessing` 校验 = 无 preprocess job pending/running（跟 `check_regularizing` 同 pattern；可跳过不强求处理过任何图）。

`_v11_preprocessing_phase` migration 隐式回填：phase=curating + train/ 非空 → 推进到 preprocessing；其他不动。零用户感知。

### 去重 / blur / 聚类 / 统计 scope 跟随下沉

`studio/services/preprocess/duplicates.py` 的 scope 从 `download/` 改成 `versions/{label}/train/`。**不拆模块**——R2 推荐的"去重上提到 ingestion 阶段"在本方案下失意义，因为去重就是为了清理 train 集本身。

## 理由

**为什么 ADR 0004 选项目级被推翻**：

不是决策当时的论证错了，是决策当时的两个**事实假设**变了：

1. ADR 0007 落地的 fork 机制（`_copytree("train")`）让"跨版本复用"不需要项目级缓存就能实现；
2. 用户实际工作流是「v2 从 v1 复制 + 微调」远多于「v2 从空 template 重建」——重做 upscale 几乎不会发生。

ADR 0004 §227 "跨版本复用"论据在新事实下**不构成项目级 scope 的支撑**，反而是 ADR 0007 fork 路径在做这件事。

**为什么不选方案 B（UI filter）**：

R1/R2 三方审阅一致推荐 filter 折中，但 R3 在用户反馈下集体翻盘——filter 解的是 R1/R2 自己脑补的"视觉干扰"痛点，没解任何用户列出的真痛点（前置时间 / 统计无意义 / 聚类失效）。把 filter 当方案 = 把症状当根因。R3-ux §1.2 给了详细方法论检讨（建议沉淀为 feedback memory）。

**为什么 fallback 而非显式迁移脚本**：

主审 v1 汇总推荐"一次性迁移脚本 + UI 弹窗"，被用户原话否决：

> 提议是做一个 fallback，因为我们不会主动去删除 preprocess 文件夹，如果 train 里面没有 manifest 文件，选中当前 version 或者是复制升级 version 时候，根据老的 preprocess 的 manifest 重新构建一个 train 的 manifest 就可以，全部是隐式的，不需要提示用户，而且保持了最小代码。

事实是：用户的 train/ 已经是 preprocess 复制过来的产物（curate 阶段已发生）——删 project 级 preprocess/ 不影响 train/ 物理 bytes。唯一损失是 origin 反查，而这条用 `ensure_train_manifest` 30 行代码 lazy 重建就够。比"一次性脚本 + UI 弹窗"省 ~1d 工时 + 零用户感知 + 零失败回滚成本。

**为什么 schema 极简化（不存 kind / state / 过程信息）**：

承袭 ADR 0004 Addendum 1 的极简精神 + MEMORY `feedback_preprocess_data_model_simple`「manifest 只记 origin/mtime/size；download 唯一备份；stage 自由覆盖不强时序；老 schema 读兼容写丢弃」。状态从字段差异（扩展名变 / mtime 漂移 / 命名含 `_c0` 后缀）隐含推断，避免存"过程信息"后字段语义漂移。

**为什么加 phase 而不是只动 Sidebar UI**：

用户原话："加上状态机更严格符合我们的设计"——前提是只有隐式 DB migration 无用户感知。现有 `_v8_version_status_phase.py` 已经证明加 phase 值是干净的 add-only migration pattern。`check_preprocessing` 跟 `check_regularizing` 同 pattern，零设计代价。

**为什么不撕 ADR 0007 §70 "数据集 version 级否决"**：

§70 否决的是"每 version 独立的 train 集（数据集本身 version 级）"——本 ADR 下 train 集合的 source-of-truth 仍是项目级 `download/` 池（curating phase 从 download 复制进 train）；本 ADR 下沉的是**预处理产物**，不是数据集归属。ADR 0007 加 amendment 精确化措辞即可，不需要反转主决策。

## 后果

### 好处

- 用户痛点 4/4 真解：前置时间 / 统计 / 聚类 / scope 错位全部修复
- 跟用户心智模型对齐：preprocess 是"对我要训练的图做精细处理"
- ADR 0007 fork 机制自然承接跨版本复用，不重复发明缓存
- manifest 模块大幅瘦身（25 函数 → 一半瘦身 + 极简 schema）
- 老项目零感知升级（隐式 fallback）
- 去重 / 聚类指标 finally 对训练有意义

### 代价 / 新增约束

- `versions.py:_copytree("train")` fork 时复制几 GB train + manifest 是真实磁盘代价；用户已接受（"v3 通常只改参数"频次最高，磁盘代价摊销低）
- `restore` 在 download 缺失时**真的失败**——这是有意的，跟 ADR 0004 原则一致，UI 显式失败优于静默假复原
- 老 `projects/{id}/preprocess/` 目录**永远不主动删**，长期占磁盘——next minor release 提示用户可手动清，不强制
- 11 个 API endpoint URL 从 pid → (pid, vid) **breaking change**（beta 心智 + 前后端同 PR 切换，不做 redirect 兼容期）
- `_v11_preprocessing_phase` migration 是 ADR 0007 `_v9 destructive` 之后第二次动 phase 列——必须严格在 _v9 之后跑

### 还的债 / 未来扩展

- 老 `preprocess/` 目录的清理：0.13.0 release notes 提醒用户可删；0.14.0 起 `ensure_train_manifest` 可考虑加"老 manifest 不存在则忽略"分支（一旦绝大多数老项目都已 lazy 重建过，老 fallback 路径就是死代码可清）
- 如果未来出现"v1 复制到 v2 后想重做某 stage"的高频需求：当前是用户进 v2 显式点"重做 upscale"按钮（手动），将来如果需要可加 fork 时 dialog 询问。但不预投资
- multi-crop 派生在 fork 间的隔离：v1 crop 后 fork v2，v2 改 crop 不应回流 v1——已经被 fork 物理复制天然保证，但测试 case 要覆盖

## 参考

- 用户反馈来源：`tmp/preprocess-redesign-user-feedback.md`（含投票背景 + 4 个真痛点原话）
- 三方审阅档案：`tmp/preprocess-redesign-r{1,2,3}-{arch,ux,impl}.md`（9 份）
- 汇总演进：`tmp/preprocess-redesign-final-summary.md`（v1）→ `tmp/preprocess-redesign-final-summary-v2.md`（v2，本 ADR ground truth）
- 实施计划（含 30 条决策点 + file:line 改动清单 + 4 PR 切片）：[`docs/design/preprocess-train-scope-plan.md`](../design/preprocess-train-scope-plan.md)
- 被取代的 ADR：[ADR 0004](0004-preprocess-manifest.md)（含 Addendum 1 multi-crop schema 演进）
- 牵连的 ADR：[ADR 0007](0007-project-version-lifecycle-refactor.md)（加 Addendum 1 加 `preprocessing` phase）
- 不动的 ADR：[ADR 0008](0008-studio-restructure-0.11.0.md)（模块边界）/ [ADR 0009](0009-logging-error-system.md)（日志体系）
