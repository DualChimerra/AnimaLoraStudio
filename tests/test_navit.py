"""NaViT / Patch-n-Pack 的 CPU 可测面：schema 校验 + 数据层接线。

块对角打包前向（forward_packed_navit）硬依赖 xformers varlen 内核，只能在
Colab GPU 上验证（见 docs 与 test_navit_packed_objective 上游版）。本文件只覆盖
不需要 GPU 的部分：schema 互斥/前置校验、ImageDataset 原生定尺寸、collate。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# schema 校验（opt-in / fail-fast）
# ---------------------------------------------------------------------------

def _base():
    return dict(data_dir="x", output_dir="o", output_name="n")


def test_navit_default_off_is_neutral():
    from studio.schema import TrainingConfig
    c = TrainingConfig(**_base())
    assert c.navit_packing is False
    assert c.navit_token_budget == 16384
    assert c.navit_native_resolution is False
    assert c.navit_pack_strategy == "next_fit"


def test_navit_forces_xformers_backend():
    from studio.schema import TrainingConfig
    c = TrainingConfig(**_base(), navit_packing=True, cache_latents=True, navit_token_budget=8192)
    assert c.attention_backend == "xformers"


def test_navit_conflicts_infonoise():
    from studio.schema import TrainingConfig
    with pytest.raises(Exception):
        TrainingConfig(**_base(), navit_packing=True, cache_latents=True, infonoise_enabled=True)


def test_navit_requires_cache_latents():
    from studio.schema import TrainingConfig
    with pytest.raises(Exception):
        TrainingConfig(**_base(), navit_packing=True, cache_latents=False)


def test_native_resolution_requires_packing():
    """navit_packing 关闭时孤儿 navit_native_resolution=true 静默收敛为 False。

    UI 上该字段 show_when="navit_packing==true"：关掉 packing 后字段从界面消失但值
    仍透传，若 fail-fast 会让每次保存都报错（用户看不见原因）。故改为跟随父开关关闭。
    """
    from studio.schema import TrainingConfig
    cfg = TrainingConfig(**_base(), navit_native_resolution=True, navit_packing=False)
    assert cfg.navit_native_resolution is False


# ---------------------------------------------------------------------------
# 数据层：ImageDataset 原生定尺寸 + collate
# ---------------------------------------------------------------------------

def _load_dataset_module():
    name = "navit_ds_mod"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).resolve().parents[1] / "runtime" / "training" / "dataset.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_dataset(ds, tmp_path, native: bool, max_tokens: int = 0):
    """构造一个最小 ImageDataset（一张图 + txt caption）。"""
    from PIL import Image
    d = tmp_path / "1_concept"
    d.mkdir(parents=True)
    Image.new("RGB", (1000, 1500), (128, 64, 32)).save(d / "a.png")
    (d / "a.txt").write_text("1girl, solo", encoding="utf-8")
    bm = ds.BucketManager()
    return ds.ImageDataset(
        tmp_path, resolution=1024, bucket_mgr=bm,
        native_resolution=native,
        native_max_tokens=max_tokens if native else 0,
        native_over_budget="downscale",
    )


def test_imagedataset_target_size_bucket_vs_native(tmp_path):
    ds = _load_dataset_module()
    # 默认（桶）路径：目标尺寸来自 bucket_mgr
    d_bucket = _make_dataset(ds, tmp_path / "b", native=False)
    tw, th = d_bucket.target_size_for(1000, 1500)
    assert (tw, th) == d_bucket.bucket_mgr.get_bucket(1000, 1500)
    # 原生路径（预算不限）：纯 floor-16 定尺寸，与桶不同
    d_native = _make_dataset(ds, tmp_path / "n", native=True, max_tokens=0)
    ntw, nth = d_native.target_size_for(1000, 1500)
    assert ntw == (1000 // 16) * 16 and nth == (1500 // 16) * 16   # 992 x 1488
    assert (ntw, nth) != (tw, th)   # 原生绕过桶量化
    # 原生 + 超预算：等比 downscale 到 token 预算内，仍 16 对齐
    d_budget = _make_dataset(ds, tmp_path / "s", native=True, max_tokens=4096)
    bw, bh = d_budget.target_size_for(1000, 1500)
    assert bw % 16 == 0 and bh % 16 == 0
    assert (bw // 16) * (bh // 16) <= 4096


def test_collate_fn_navit_pack_keeps_list(tmp_path):
    import torch
    ds = _load_dataset_module()
    batch = [
        {"latent": torch.zeros(16, 1, 32, 32), "caption": "a"},
        {"latent": torch.zeros(16, 1, 64, 16), "caption": "b"},   # 不同形状 → 不能 stack
    ]
    out = ds.collate_fn_navit_pack(batch)
    assert isinstance(out["navit_latents"], list) and len(out["navit_latents"]) == 2
    assert out["captions"] == ["a", "b"]
    assert out["navit_latents"][0].shape != out["navit_latents"][1].shape
