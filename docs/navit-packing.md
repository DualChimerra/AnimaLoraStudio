# NaViT / Patch-n-Pack 块对角打包训练（本 fork 移植版）

> 状态：**已移植** —— 模型内核 + 数据层 + 训练循环接线 + schema。opt-in / default-off，
> 关掉时与改动前逐字节等价（本地 146 测试全绿，含 default-path 回归）。
> **块对角打包前向硬依赖 xformers varlen 内核，只能在 Colab GPU 上验证** —— 本地
> 无 GPU/xformers 无法跑通打包前向；首次真训练务必在 Colab 观察（见 §4）。
> 从上游 v0.18.0 移植，裁掉了本 fork 没有的 leap/sra/tlora 互斥项。

## 1. 解决什么问题

小数据集 + 多分辨率 + 想开高 batch 时，ARB 分桶按精确 `(h, w)` 分桶 → 每桶几张图 →
填不满高 batch。**NaViT/Patch-n-Pack（arXiv 2307.06304）** 把多张异构图拼进一条序列，
用块对角注意力让每图只注意自己的 token（self）与自己的 caption（cross），每图带自己
的 timestep。于是"每步处理多少图"与"单图形状"解耦——零 padding、走 xformers varlen 快内核。

配合 `navit_native_resolution` 时单图按**原生尺寸** floor 到 16px 定尺寸、绕过桶量化——
这正是"保留原始织物/纹理细节，不被 downscale 抹平"的来源。

## 2. 怎么开（config / yaml）

```yaml
cache_latents: true            # 必需（打包按 latent token 数预算分包，需预编码缓存）
navit_packing: true            # 总开关（默认 false）
navit_token_budget: 16384      # 一个 pack 的 token 数之和上限（按显存定；见下表）
navit_max_images_per_pack: 0   # 单 pack 最多几张图，0=不限
# 打包策略
navit_pack_strategy: next_fit  # next_fit（默认，顺序贪心）/ ffd（窗口内 FFD，包更满）
navit_pack_ffd_window: 256     # ffd 窗口大小（0=全局 FFD；>0=窗口内 FFD + 跨 epoch reshuffle）
navit_drop_last: false         # 丢弃每 epoch 最后未满预算的包
navit_text_trim_padding: false # cross-attn 按每图有效 T5 长度打包文本，去 512-pad
# 原生分辨率（可选，纹理保留的关键）
navit_native_resolution: true       # 需 navit_packing + cache_latents；默认 false
navit_native_over_budget: downscale # 超大图：downscale（默认，永不 OOM）/ fail
```

`navit_packing=true` 时 `attention_backend` 会被自动强制为 `xformers`（schema 层 coercion）。

### 显存 ↔ token_budget 对照（grad_checkpoint=true 下的保守起点）

| 显存 | 起步 token_budget | ≈（4096-token/张） |
|---|---|---|
| 16 GB | 16384 | ~4 张 |
| 24 GB | 32768 | ~8 张 |
| 48 GB | 65536 | ~16 张 |
| 80 GB | 98304 | ~24 张 |

**务必首跑观察峰值显存再调**——卡、rank、底模大小都会左右它。

## 3. v1 支持范围与门控（schema fail-fast）

**互斥（同时开 → 启动即报错）**：`infonoise_enabled`（I-MMSE record 需标准路径 per-sample
MSE，navit 的 per-image loss 语义不同）。*上游还互斥 leap/sra/tlora，本 fork 无这些特性。*

**前置**：`cache_latents=true`、`navit_token_budget>0`；`navit_native_resolution` 需 `navit_packing`。

**支持**：basic flow-matching（逐图 t + 噪声 + per-image loss）、`grad_accum`、逐块梯度检查点
（`grad_checkpoint`）、LoRA/LoKr 保存/恢复、`loss_weighting`（按 per-image t 算权重）、正则集
降权（`loss_weight`，逐图应用）。

## 4. Colab 验证（本地无法跑打包前向）

本地已验证：schema 校验、原生定尺寸、打包 sampler/collate、token 计数（`tests/test_navit.py`
+ `tests/test_multires_and_local_base.py`）。**块对角前向必须在 GPU 上验证**：

1. Colab 装 xformers（与 torch 版本匹配）：`pip install xformers`。
2. 拿一个小数据集，config 里 `cache_latents=true` + `navit_packing=true` +
   `navit_token_budget=16384`，跑几十步。观察：
   - 启动无 `forward_packed_navit requires xformers` 报错（= xformers 生效）。
   - loss 有限、下降；显存峰值在预期内。
   - 出图预览正常（与非 navit 训练的同数据集对照，风格一致）。
3. 再开 `navit_native_resolution=true` 验证原生尺寸（大图纹理保留、无 OOM）。
4. 与 LoRA/LoKr 注入实跑一遍（首跑重点）。

若打包前向数值有疑，参考上游 `tests/test_navit_packed_objective.py`（GPU + xformers）断言
`forward_packed_navit ≡ 各图单独前向拼接`——可移植到本 fork 在 Colab 跑。

## 5. 实现地图（本 fork）

| 层 | 文件 |
|---|---|
| 块对角 attention op + 逐图 AdaLN token 前向 + 打包前向 | `models/cosmos_predict2_modeling.py`（`torch_attention_op` attn_mask 分支、`Block/FinalLayer.forward_tokens`、`patchify_latents_to_tokens`、`_packed_rope_from_grid`、`forward_packed_navit`） |
| 训练步核心 | `runtime/training/navit.py`（`navit_packed_forward_and_loss` / `pack_cross_embeddings`） |
| token 预算打包 + 原生定尺寸 | `runtime/training/dataset.py`（`NavitPackBatchSampler` / `collate_fn_navit_pack` / `plan_native_fit_image` / `dataset_token_counts`） |
| 训练循环接线 | `runtime/training/loop.py`（navit 分支：pack cross → per-image 权重 → 打包前向） |
| 数据加载接线 | `runtime/training/phases/dataset.py`（navit_packing → NavitPackBatchSampler + collate；native → ImageDataset native params） |
| config 键 + 互斥校验 | `studio/domain/training.py`（`navit_*` / `cache_encode_*` 字段 + `_validate_navit_exclusive` + `_coerce_navit_attention_backend`） |
