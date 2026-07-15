"""多分辨率 ARB 分桶（可配置桶比例）+ 本地 base 模型注册的回归测试。

覆盖两块新功能：
1. BucketManager 的 min/max/step/max_ar/area_tolerance 变成可配置，默认值
   与前端 trainBuckets.ts DEFAULTS 对齐（默认集恒为 37 个桶）。
2. selected_anima / resolve_anima_main_path / build_catalog 支持把
   diffusion_models/ 里的本地 .safetensors 注册为自定义 base（镜像 upscaler custom）。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1) 多分辨率分桶
# ---------------------------------------------------------------------------

def _load_bucket_manager():
    # 单独 load dataset.py，避开 training 包 __init__ 的重依赖（torch 等已装）。
    # 注册进 sys.modules —— dataset.py 用 `from __future__ import annotations` +
    # @dataclass，dataclass 需能从 sys.modules 解析字符串注解，否则 exec 报
    # AttributeError('NoneType' has no __dict__)。
    import sys
    name = "ds_for_bucket"
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).resolve().parents[1] / "runtime" / "training" / "dataset.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.BucketManager


def test_bucket_defaults_yield_37():
    BM = _load_bucket_manager()
    # 与 studio/web/src/lib/trainBuckets.test.ts 的 count==37 不变量对齐。
    assert len(BM().buckets) == 37
    explicit = BM(base_reso=1024, min_reso=512, max_reso=2048, step=64,
                  max_ar=2.0, area_tolerance=0.1)
    assert explicit.buckets == BM().buckets


def test_bucket_max_ar_configurable():
    BM = _load_bucket_manager()
    assert len(BM(max_ar=1.5).buckets) < 37 < len(BM(max_ar=3.0).buckets)


def test_bucket_schema_fields_and_validator():
    from studio.schema import TrainingConfig
    c = TrainingConfig(data_dir="x", output_dir="o", output_name="n")
    assert (c.bucket_min_reso, c.bucket_max_reso, c.bucket_step, c.bucket_max_ar) == (512, 2048, 64, 2.0)
    # min > max fail-fast
    with pytest.raises(Exception):
        TrainingConfig(data_dir="x", output_dir="o", output_name="n",
                       bucket_min_reso=2048, bucket_max_reso=512)


# ---------------------------------------------------------------------------
# 2) 本地 base 模型注册
# ---------------------------------------------------------------------------

@pytest.fixture()
def models_root(tmp_path: Path) -> Path:
    dm = tmp_path / "diffusion_models"
    dm.mkdir(parents=True)
    (dm / "anima-base-v1.0.safetensors").write_bytes(b"PRESET")   # 预设文件名
    (dm / "my-finetune.safetensors").write_bytes(b"CUSTOM")       # 自定义本地 base
    (dm / "notes.txt").write_bytes(b"junk")                       # 非白名单扩展名
    return tmp_path


def test_is_custom_anima_guards(models_root: Path):
    from studio.services.models import paths as P
    assert P.is_custom_anima("my-finetune.safetensors", models_root) is True
    assert P.is_custom_anima("anima-base-v1.0.safetensors", models_root) is False  # 预设
    assert P.is_custom_anima("../evil.safetensors", models_root) is False          # 穿越
    assert P.is_custom_anima("missing.safetensors", models_root) is False          # 不存在
    assert P.is_custom_anima("notes.txt", models_root) is False                    # 扩展名
    assert P.is_custom_anima("1.0", models_root) is False                          # 预设 label


def test_resolve_anima_main_path(models_root: Path):
    from studio.services.models import paths as P
    assert P.resolve_anima_main_path("1.0", models_root).name == "anima-base-v1.0.safetensors"
    assert P.resolve_anima_main_path("my-finetune.safetensors", models_root) == \
        models_root / "diffusion_models" / "my-finetune.safetensors"
    # 未知值回退到 latest 预设路径
    assert P.resolve_anima_main_path("gone.safetensors", models_root).name == "anima-base-v1.0.safetensors"


def test_catalog_lists_custom_base(models_root: Path):
    from studio.services.models.catalog import build_catalog
    variants = build_catalog(models_root)["anima_main"]["variants"]
    by_name = {v["variant"]: v for v in variants}
    assert by_name["my-finetune.safetensors"]["kind"] == "custom"
    assert by_name["my-finetune.safetensors"]["exists"] is True
    assert by_name["1.0"]["kind"] == "preset"
    assert "notes.txt" not in by_name
