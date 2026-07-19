"""本 fork：ARB 分桶显式旋钮（bucket_min/max_reso / bucket_step / bucket_max_ar 别名）。

0.20 backport 后：
- 旋钮为 Optional（None = 按 base_reso 自动推导，与上游一致）；
- 显式 512/2048/64 时桶集合与老版 fork 逐字节一致（trainBuckets.ts 预测依赖）；
- ``bucket_max_ar`` 是 ``aspect_ratio_limit`` 的兼容别名（老配置迁移）。

本地自定义 base 模型（旧 is_custom_anima / resolve_anima_main_path）已被上游
#446 统一候选列表（models_storage + secrets.model_sources）取代，相关测试见
tests/test_models_storage*.py。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


# 0.20 后 dataset.py 内部走相对导入（families spec），不能再单文件 exec —— 直接
# 包导入（conftest 已把 runtime/ 放进 sys.path；torch 在测试环境可用）。
from training.dataset import BucketManager  # noqa: E402


def test_bucket_defaults_yield_37():
    """默认（自动推导）在 base=1024 下与老版 512/2048/64 桶集合完全一致。"""
    derived = BucketManager(1024, aspect_ratio_limit=2.0)
    explicit = BucketManager(1024, min_reso=512, max_reso=2048, step=64,
                             aspect_ratio_limit=2.0)
    assert derived.buckets == explicit.buckets
    assert len(derived.buckets) == 37


def test_bucket_explicit_knobs_honored():
    """显式旋钮改变桶集合（更粗 step → 更少桶）。"""
    fine = BucketManager(1024, min_reso=512, max_reso=2048, step=64,
                         aspect_ratio_limit=2.0)
    coarse = BucketManager(1024, min_reso=512, max_reso=2048, step=128,
                           aspect_ratio_limit=2.0)
    assert len(coarse.buckets) < len(fine.buckets)


def test_bucket_max_ar_configurable():
    narrow = BucketManager(1024, aspect_ratio_limit=1.0)
    wide = BucketManager(1024, aspect_ratio_limit=3.0)
    assert len(narrow.buckets) < len(wide.buckets)
    for w, h in wide.buckets:
        assert max(w / h, h / w) <= 3.0 + 1e-9


def test_bucket_schema_fields_and_validator():
    """schema 端：旋钮字段存在、默认 None；bucket_max_ar 别名映射。"""
    from studio.domain.training import TrainingConfig

    c = TrainingConfig()
    assert c.bucket_min_reso is None
    assert c.bucket_max_reso is None
    assert c.bucket_step is None
    assert c.aspect_ratio_limit == 2.0

    c2 = TrainingConfig(bucket_min_reso=512, bucket_max_reso=2048, bucket_step=64)
    assert (c2.bucket_min_reso, c2.bucket_max_reso, c2.bucket_step) == (512, 2048, 64)

    # 老 fork 配置的 bucket_max_ar → aspect_ratio_limit（显式后者优先）
    c3 = TrainingConfig.model_validate({"bucket_max_ar": 3.0})
    assert c3.aspect_ratio_limit == 3.0
    c4 = TrainingConfig.model_validate({"bucket_max_ar": 3.0, "aspect_ratio_limit": 2.5})
    assert c4.aspect_ratio_limit == 2.5
