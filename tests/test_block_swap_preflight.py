"""block swap 预检（training/block_swap_preflight.py）。

核心 ``evaluate`` 是纯算术：预算输入全部由参数传入，不查系统、不碰 CUDA、
不读权重，所以可以直接拿数字断言。``run`` 的用例只覆盖「什么时候不下判断」
这条防误拒的主线。

场景数字取自真机目标配置：12GB 卡 + 32GB 内存 + Krea 2（fp8 13.1GB /
bf16 26.3GB，28 层，主干占全模型参数 ≈94.5%）。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training import block_swap_preflight as preflight  # noqa: E402
from training import sysmem  # noqa: E402


_GIB = 1024 ** 3
_TOTAL_BLOCKS = 28
#: Krea 2 的 28 层主干占全模型参数的比例（其余是嵌入 / 输出层，换不出去）
_BLOCK_SHARE = 0.945

_FP8_BYTES = int(13.1 * _GIB)
_BF16_BYTES = int(26.3 * _GIB)
#: 12GB 卡的实际空闲（驱动 + 桌面占掉一点）
_FREE_VRAM_12G = int(11.5 * _GIB)
#: 32GB 机器上训练开始时的典型可用内存
_AVAIL_RAM_32G = int(24 * _GIB)


def _ratio(blocks: int) -> float:
    """换出 N 层占全模型参数的比例（各层等大，与 krea2 实际结构一致）。"""
    return min(max(blocks, 0), _TOTAL_BLOCKS) / _TOTAL_BLOCKS * _BLOCK_SHARE


def _evaluate(
    *,
    file_bytes=_FP8_BYTES,
    blocks_to_swap=0,
    free_vram=_FREE_VRAM_12G,
    avail_ram=_AVAIL_RAM_32G,
):
    return preflight.evaluate(
        file_bytes=file_bytes,
        blocks_to_swap=blocks_to_swap,
        total_blocks=_TOTAL_BLOCKS,
        ratio_fn=_ratio,
        free_vram_bytes=free_vram,
        avail_ram_bytes=avail_ram,
        vram_base_bytes=sysmem._VRAM_BASE_BYTES,
        pinned_limit_bytes=(
            None if avail_ram is None else sysmem.pinned_safe_limit(avail_ram)
        ),
    )


# ---------------------------------------------------------------------------
# 主线：默认 blocks_to_swap=0 在 12GB 上必须被拦下并给出建议
# ---------------------------------------------------------------------------


def test_default_zero_swap_is_rejected_with_recommendation() -> None:
    """13.1GB fp8 底模不开 swap 装不进 12GB —— 这正是「选中就开跑」的坑。"""
    result = _evaluate(blocks_to_swap=0)

    assert result.checked and not result.ok
    assert result.recommended is not None
    assert 0 < result.recommended <= _TOTAL_BLOCKS
    assert "blocks_to_swap" in result.message


def test_recommendation_leaves_training_headroom() -> None:
    """推荐值不能只够放权重：LoRA / 优化器状态 / 激活 / dequant 还要地方。

    否则用户照着推荐值改完，在第 3 分钟 OOM —— 比不给推荐更糟。
    """
    result = _evaluate(blocks_to_swap=0)
    weights_only = int(_FP8_BYTES * (1.0 - _ratio(result.recommended)))

    assert _FREE_VRAM_12G - weights_only >= preflight._RECOMMEND_HEADROOM_BYTES


def test_recommended_value_itself_passes_preflight() -> None:
    """推荐值必须自洽：照着改一遍就应当通过（不能推荐一个仍被拒的值）。"""
    recommended = _evaluate(blocks_to_swap=0).recommended

    assert _evaluate(blocks_to_swap=recommended).ok


def test_configured_swap_that_fits_passes() -> None:
    """fp8 + 换出 26 层是 12GB/32GB 的目标配置，必须直通。"""
    result = _evaluate(blocks_to_swap=26)

    assert result.checked and result.ok
    assert result.message == ""


# ---------------------------------------------------------------------------
# bf16 底模：12GB/32GB 上存在能跑的档位，但没有一个留得下训练余量
# ---------------------------------------------------------------------------


def test_bf16_on_32g_ram_rejects_26_blocks() -> None:
    """换出 26 层时锁定内存超上限（24.8GB > 19.2GB）——正是 32GB 机器的边界。"""
    result = _evaluate(file_bytes=_BF16_BYTES, blocks_to_swap=26)

    assert result.checked and not result.ok
    assert "锁定" in result.message


def test_bf16_on_32g_ram_recommends_a_tight_value_and_says_so() -> None:
    """bf16 在 12GB/32GB 上不是完全不可能，但余量为零 —— 文案必须说明。

    只给一个数字而不提「这只够放权重」，用户照做后中途 OOM 会以为预检骗了他。
    """
    result = _evaluate(file_bytes=_BF16_BYTES, blocks_to_swap=26)

    assert result.recommended is not None
    assert "刚好装下权重" in result.message
    assert "fp8" in result.message


def test_fp8_recommendation_is_not_flagged_tight() -> None:
    """对照组：fp8 底模有留够余量的档位，不该带「紧」的警告。"""
    result = _evaluate(blocks_to_swap=0)

    assert "刚好装下权重" not in result.message


def test_no_workable_swap_count_when_ram_is_small() -> None:
    """bf16 + 16GB 内存：锁定上限压到 8GB，任何档位都不成立。"""
    result = _evaluate(
        file_bytes=_BF16_BYTES, blocks_to_swap=26, avail_ram=int(12 * _GIB),
    )

    assert result.checked and not result.ok
    assert result.recommended is None
    assert "fp8" in result.message


def test_bf16_fits_comfortably_when_ram_is_large_enough() -> None:
    """同样的 bf16 底模，大内存机器上换出足够多的层就能跑。"""
    result = _evaluate(
        file_bytes=_BF16_BYTES, blocks_to_swap=28, avail_ram=int(56 * _GIB),
    )

    assert result.ok


# ---------------------------------------------------------------------------
# 防误拒：信息不全时一律不下判断
# ---------------------------------------------------------------------------


def test_unknown_free_vram_skips_judgement() -> None:
    result = _evaluate(blocks_to_swap=0, free_vram=None)

    assert not result.checked and result.ok


def test_missing_checkpoint_size_skips_judgement() -> None:
    result = _evaluate(file_bytes=0, blocks_to_swap=0)

    assert not result.checked


def test_unknown_ram_still_checks_vram_side() -> None:
    """内存查不到不影响显存侧判断（各信号独立降级）。"""
    result = _evaluate(blocks_to_swap=0, avail_ram=None)

    assert result.checked and not result.ok


def test_unknown_ram_never_blames_pinned_memory() -> None:
    """内存未知时不能因为「锁定内存超限」拒绝 —— 那是凭空捏造的判据。"""
    result = _evaluate(
        file_bytes=_BF16_BYTES, blocks_to_swap=28, avail_ram=None,
    )

    assert result.ok


def test_blocks_to_swap_above_total_is_clamped() -> None:
    """旧 yaml / 裸 CLI 可能写超过层数的值；按上限评估，不炸。"""
    result = _evaluate(blocks_to_swap=999)

    assert result.checked and result.ok


# ---------------------------------------------------------------------------
# 与 check_pinned_budget 共用同一条水位线（两处漂移 = 预检说行、护栏当场拒）
# ---------------------------------------------------------------------------


def test_pinned_limit_agrees_with_guard(monkeypatch) -> None:
    avail = int(24 * _GIB)
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: avail)
    limit = sysmem.pinned_safe_limit(avail)

    sysmem.check_pinned_budget(limit, blocks=26)  # 正好在线上：放行
    with pytest.raises(RuntimeError):
        sysmem.check_pinned_budget(limit + 1, blocks=26)


# ---------------------------------------------------------------------------
# run(ctx)：跳过条件
# ---------------------------------------------------------------------------


def _ctx(*, capabilities=frozenset({"block_swap"}), preflight_on=True):
    family = types.SimpleNamespace(
        spec=types.SimpleNamespace(capabilities=capabilities),
        swapped_param_ratio=_ratio,
        swappable_blocks=lambda: _TOTAL_BLOCKS,
    )
    args = types.SimpleNamespace(
        block_swap_preflight=preflight_on,
        blocks_to_swap=0,
        transformer_path="/nonexistent/model.safetensors",
    )
    return types.SimpleNamespace(args=args, family=family)


def test_run_is_noop_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(sysmem, "_file_bytes", lambda _p: _FP8_BYTES)
    monkeypatch.setattr(sysmem, "gpu_free_bytes_global", lambda: _FREE_VRAM_12G)
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: _AVAIL_RAM_32G)

    preflight.run(_ctx(preflight_on=False))  # 关掉就不该抛


def test_run_is_noop_for_families_without_block_swap(monkeypatch) -> None:
    monkeypatch.setattr(sysmem, "_file_bytes", lambda _p: _FP8_BYTES)
    monkeypatch.setattr(sysmem, "gpu_free_bytes_global", lambda: _FREE_VRAM_12G)
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: _AVAIL_RAM_32G)

    preflight.run(_ctx(capabilities=frozenset({"masked_loss"})))


def test_run_raises_with_recommendation(monkeypatch) -> None:
    monkeypatch.setattr(sysmem, "_file_bytes", lambda _p: _FP8_BYTES)
    monkeypatch.setattr(sysmem, "gpu_free_bytes_global", lambda: _FREE_VRAM_12G)
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: _AVAIL_RAM_32G)

    with pytest.raises(RuntimeError, match="建议"):
        preflight.run(_ctx())


def test_run_survives_broken_family_estimate(monkeypatch) -> None:
    """预检是辅助设施：自身出错要放行，不能挡住训练。"""
    monkeypatch.setattr(sysmem, "_file_bytes", lambda _p: _FP8_BYTES)
    monkeypatch.setattr(sysmem, "gpu_free_bytes_global", lambda: _FREE_VRAM_12G)
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: _AVAIL_RAM_32G)

    def _boom(_blocks):
        raise RuntimeError("meta 模型构造失败")

    ctx = _ctx()
    ctx.family.swapped_param_ratio = _boom

    preflight.run(ctx)
