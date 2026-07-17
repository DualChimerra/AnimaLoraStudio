"""dataset_phase：build datasets + dataloader + VAE roundtrip 自检。

抽自 main() L257-342（ADR 0003 PR-B）。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.context import TrainingContext
from training.dataset import (
    BucketBatchSampler,
    BucketManager,
    CachedLatentDataset,
    ImageDataset,
    MergedDataset,
    NavitPackBatchSampler,
    collate_fn,
    collate_fn_cached,
    collate_fn_navit_pack,
)


logger = logging.getLogger(__name__)


def run(ctx: TrainingContext) -> None:
    """
    - 主数据集 / 正则数据集 + per-folder repeat
    - cache_latents 包 CachedLatentDataset
    - MergedDataset 串联主集 + 正则集
    - Windows num_workers > 0 兜底为 0（多进程 spawn 易崩）
    - BucketBatchSampler / DataLoader
    - VAE encode-decode 循环自检（vae_roundtrip.png）
    """
    args = ctx.args

    # 数据集
    # 多分辨率 ARB 分桶（可配置桶比例）。getattr 兜底旧 yaml —— 缺字段时用与
    # BucketManager / trainBuckets.ts 一致的默认值（512/2048/64/2.0），行为不变。
    ctx.bucket_mgr = BucketManager(
        base_reso=args.resolution,
        min_reso=int(getattr(args, "bucket_min_reso", 512) or 512),
        max_reso=int(getattr(args, "bucket_max_reso", 2048) or 2048),
        step=int(getattr(args, "bucket_step", 64) or 64),
        max_ar=float(getattr(args, "bucket_max_ar", 2.0) or 2.0),
    )
    # NaViT 原生定尺寸参数（navit_native_resolution，opt-in）。关闭时全为中性值 →
    # ImageDataset 走原有 ARB 桶路径，行为不变。RoPE 单边上限从模型取（max_img_h//
    # patch_spatial）；取不到则 0（不设限，前向 _packed_rope_from_grid 会 fail-fast）。
    _navit_native = bool(getattr(args, "navit_native_resolution", False))
    _native_max_tokens = int(getattr(args, "navit_token_budget", 0) or 0) if _navit_native else 0
    _native_max_side = 0
    if _navit_native:
        _m = getattr(ctx, "model", None)
        _mh = getattr(_m, "max_img_h", None)
        _ps = getattr(_m, "patch_spatial", 2) or 2
        if _mh:
            _native_max_side = int(_mh) // int(_ps)
    _native_kwargs = dict(
        native_resolution=_navit_native,
        native_max_tokens=_native_max_tokens,
        native_max_side_tokens=_native_max_side,
        native_over_budget=str(getattr(args, "navit_native_over_budget", "downscale") or "downscale"),
    )
    ctx.base_dataset = ImageDataset(
        args.data_dir, args.resolution, ctx.bucket_mgr,
        shuffle_caption=args.shuffle_caption,
        keep_tokens=args.keep_tokens,
        flip_augment=args.flip_augment,
        tag_dropout=args.tag_dropout,
        prefer_json=args.prefer_json,
        **_native_kwargs,
    )
    ctx.dataset = ctx.base_dataset

    # 正则数据集（Kohya 风格，防过拟合）
    reg_data_dir = getattr(args, "reg_data_dir", "") or ""
    ctx.reg_dataset = None
    if reg_data_dir:
        if not Path(reg_data_dir).exists():
            logger.warning(f"正则数据集路径不存在，已跳过: {reg_data_dir}")
        elif len(ctx.base_dataset) == 0:
            logger.warning("主数据集为空，正则集已跳过")
        else:
            reg_caption = (getattr(args, "reg_caption", "") or "").strip()
            reg_base = ImageDataset(
                reg_data_dir, args.resolution, ctx.bucket_mgr,
                shuffle_caption=args.shuffle_caption,
                keep_tokens=args.keep_tokens,
                flip_augment=args.flip_augment,
                tag_dropout=0.0,  # 正则集通常不用 dropout
                prefer_json=args.prefer_json,
                caption_override=reg_caption if reg_caption else None,
                **_native_kwargs,
            )
            ctx.reg_dataset = reg_base
            reg_weight = float(getattr(args, "reg_weight", 1.0) or 1.0)
            cap_preview = f", caption=\"{reg_caption[:50]}{'...' if len(reg_caption) > 50 else ''}\"" if reg_caption else ""
            weight_info = f", weight={reg_weight}" if reg_weight != 1.0 else ""
            logger.info(f"正则数据集: {reg_data_dir} ({len(reg_base)} 样本, per-folder repeat{weight_info}){cap_preview}")

    # 缓存 VAE latents（在 repeat 之前）
    ctx.use_cached = getattr(args, "cache_latents", False)
    if ctx.use_cached:
        # 0 = 跟随训练 batch size（对齐 kohya GUI 的 VAE batch size 语义）
        cache_batch_size = int(getattr(args, "vae_cache_batch_size", 0) or 0)
        if cache_batch_size <= 0:
            cache_batch_size = int(getattr(args, "batch_size", 1) or 1)
        ctx.dataset = CachedLatentDataset(
            ctx.dataset, ctx.vae, ctx.device, ctx.dtype,
            cache_batch_size=cache_batch_size,
            encode_tiled=getattr(args, "cache_encode_tiled", False),
            encode_tile_px=getattr(args, "cache_encode_tile_px", 1024),
            encode_tile_overlap=getattr(args, "cache_encode_tile_overlap", 128),
            encode_max_pixels=getattr(args, "cache_encode_max_pixels", 0),
        )
    if ctx.reg_dataset is not None and ctx.use_cached:
        ctx.reg_dataset = CachedLatentDataset(
            ctx.reg_dataset, ctx.vae, ctx.device, ctx.dtype,
            cache_batch_size=cache_batch_size,
            encode_tiled=getattr(args, "cache_encode_tiled", False),
            encode_tile_px=getattr(args, "cache_encode_tile_px", 1024),
            encode_tile_overlap=getattr(args, "cache_encode_tile_overlap", 128),
            encode_max_pixels=getattr(args, "cache_encode_max_pixels", 0),
        )

    # repeat: 主数据集和正则数据集均通过文件夹名 Kohya 风格 repeat（如 5_concept），无需全局 repeat
    if ctx.reg_dataset is not None:
        reg_weight = float(getattr(args, "reg_weight", 1.0) or 1.0)
        ctx.dataset = MergedDataset(ctx.dataset, ctx.reg_dataset, reg_weight=reg_weight)

    if args.num_workers > 0 and os.name == "nt":
        logger.warning("num_workers > 0 在 Windows 上容易崩溃：已强制设为 0（避免多进程 spawn 问题）")
        args.num_workers = 0

    if bool(getattr(args, "navit_packing", False)):
        # NaViT / Patch-n-Pack：按 token 预算把异构图打成块对角包（需 cache_latents，
        # schema 层已 fail-fast 校验）。collate 保留 per-image latent 列表，loop.py 的
        # navit 分支做块对角打包前向。
        navit_sampler = NavitPackBatchSampler(
            ctx.dataset,
            token_budget=int(getattr(args, "navit_token_budget", 16384) or 16384),
            max_images_per_pack=int(getattr(args, "navit_max_images_per_pack", 0) or 0),
            shuffle=True,
            seed=getattr(args, "seed", 42),
            drop_last=bool(getattr(args, "navit_drop_last", False)),
            strategy=str(getattr(args, "navit_pack_strategy", "next_fit") or "next_fit"),
            ffd_window=int(getattr(args, "navit_pack_ffd_window", 256) or 0),
        )
        ctx.dataloader = DataLoader(
            ctx.dataset, batch_sampler=navit_sampler,
            collate_fn=collate_fn_navit_pack,
            num_workers=args.num_workers,
        )
    elif ctx.use_cached:
        # drop_last=False：桶尾不足 batch_size 出短 batch 而非丢图。
        # 对齐 kohya sd-scripts / ostris ai-toolkit；diffusion 用 LayerNorm/GroupNorm，
        # 对动态 batch 不敏感，loop.py 也按 latents.shape[0] 动态读 bs。
        batch_sampler = BucketBatchSampler(
            ctx.dataset, batch_size=args.batch_size,
            drop_last=False, shuffle=True,
            seed=getattr(args, "seed", 42),
        )
        ctx.dataloader = DataLoader(
            ctx.dataset, batch_sampler=batch_sampler,
            collate_fn=collate_fn_cached,
            num_workers=args.num_workers,
        )
    else:
        ctx.dataloader = DataLoader(
            ctx.dataset, batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )

    # 训练前自检：VAE encode->decode 循环（快速排除 VAE/scale/shape 问题）
    try:
        if len(ctx.base_dataset) > 0:
            from PIL import Image
            item0 = ctx.base_dataset[0]
            pixels0 = item0["pixel_values"].unsqueeze(0).to(ctx.device, dtype=ctx.dtype)  # [1,3,H,W]
            with torch.no_grad():
                z0 = ctx.vae.encode(pixels0.unsqueeze(2))                        # [1,16,1,h,w]
                recon0 = ctx.vae.decode(z0).squeeze(2)                           # [1,3,H,W]
                recon0 = (recon0.clamp(-1, 1) + 1) / 2
            arr0 = (recon0[0].permute(1, 2, 0).detach().cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
            Image.fromarray(arr0).save(ctx.sample_dir / "vae_roundtrip.png")
            logger.info("VAE roundtrip 自检已保存: samples/vae_roundtrip.png")
    except Exception as e:
        logger.warning(f"VAE roundtrip 自检失败（若 sample 仍是噪点，请优先修这个）: {e}")
