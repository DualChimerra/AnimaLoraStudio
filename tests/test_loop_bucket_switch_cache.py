"""ARB 切桶时归还 allocator 缓存（runtime/training/loop.py::_BucketSwitchCacheRelease，issue #505）。

背景：BucketBatchSampler 逐桶连续产出；切桶后新形状塞不进旧桶的 cached block，
reserved ≈ 旧峰值 + 新峰值。Windows WDDM 下 cudaMalloc 不失败而是溢到共享内存，
allocator 的「OOM → 释放缓存 → 重试」自愈永不触发，训练速度永久掉一半以上。
修法：latent 空间形状变化时 empty_cache 一次；同桶连续 batch 不清；navit 路径不走。

torch.cuda.empty_cache 在 CPU 上是 no-op，这里 monkeypatch 成计数器验证调用时机。
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from runtime.training import loop as loop_mod  # noqa: E402
from runtime.training.context import TrainingContext  # noqa: E402


# ---------------------------------------------------------------------------
# loop.run() 的最小 harness（CPU、标准 rectified-flow 路径、真 SGD）。
# 上游把它放在 tests/test_loop_nonfinite_loss.py 里，本仓没有那条修复，
# 所以就地内联一份。
# ---------------------------------------------------------------------------

class _ScalarGainModel(nn.Module):
    """pred = 输入 × 标量参数：梯度直达唯一参数，更新与否一眼可判。"""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.w


class _FakeFamily:
    spec = types.SimpleNamespace(family_id="fake")

    def encode_text_for_batch(self, text_stack, model, captions, device, dtype, **kw):
        return torch.zeros(len(captions), 1, 8)

    def forward_train(self, model, noisy, t, cross, use_checkpoint=False):
        return model(noisy)


class _FakeTimestepSampler:
    def sample(self, bs, device):
        return torch.full((bs,), 0.5)

    def record(self, t, mse):
        pass

    def maybe_refresh(self, step):
        pass

    def status(self):
        return {"kind": "fake"}


class _FakeInjector:
    def on_step_begin(self, step_ctx):
        pass

    def regularization_loss(self, step_ctx):
        return None

    def save(self, path):
        pass


class _FakeLoss:
    def compute(self, pred, target, t):
        return (pred - target) ** 2


class _FakeWandb:
    def log(self, *a, **k):
        pass

    def upload_model(self, *a, **k):
        pass

    def upload_state_auto(self, *a, **k):
        pass

    def upload_state_manual(self, *a, **k):
        pass

    def finish(self):
        pass


def _make_args(**over):
    d = dict(
        epochs=1,
        grad_accum=2,
        grad_checkpoint=False,
        max_steps=0,
        sample_steps=0,
        sample_every=0,
        save_every_epochs=0,
        save_every_steps=0,
        save_state_every_epochs=0,
        save_state_every_steps=0,
        log_every=10,  # 0 会在既有 infonoise 可观测性分支里除零（真实默认 10）
        loss_curve_steps=100,
        output_name="t",
        navit_packing=False,
        leap_enabled=False,
        masked_loss=False,
        loss_weighting="none",
        noise_enhancement_type="none",
        timestep_shift_resolution_aware=False,
        caption_comfy_encoding=True,
        kv_trim=False,
    )
    d.update(over)
    return types.SimpleNamespace(**d)


def _make_ctx(tmp_path, batches, monkeypatch, **args_over):
    # epoch 末的周期 IO（auto_epoch_state 写盘 + event）与本测试无关，打掉
    monkeypatch.setattr(loop_mod, "save_training_state", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "write_config_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "emit_event", lambda *a, **k: None)

    ctx = TrainingContext(args=_make_args(**args_over))
    model = _ScalarGainModel()
    ctx.family = _FakeFamily()
    ctx.device = "cpu"
    ctx.dtype = torch.float32
    ctx.use_cached = True
    ctx.dataloader = batches
    ctx.model = model
    ctx.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ctx.optimizer_type = "sgd"
    ctx.trainable_params = list(model.parameters())
    ctx.timestep_sampler = _FakeTimestepSampler()
    ctx.injector = _FakeInjector()
    ctx.loss_fn = _FakeLoss()
    ctx.wandb_monitor = _FakeWandb()
    ctx.output_dir = tmp_path / "out"
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    ctx.sample_dir = tmp_path / "samples"
    ctx.grad_clip = 0.0
    ctx.total_steps = 10
    return ctx


class _EmptyCacheSpy:
    def __init__(self, monkeypatch):
        self.calls = 0
        monkeypatch.setattr(torch.cuda, "empty_cache", self)

    def __call__(self):
        self.calls += 1


def _lat(h, w, bs=1):
    return torch.randn(bs, 4, 1, h, w)


# ---------------------------------------------------------------- 类级语义

def test_same_shape_never_releases(monkeypatch):
    spy = _EmptyCacheSpy(monkeypatch)
    tracker = loop_mod._BucketSwitchCacheRelease("cpu")
    for _ in range(5):
        tracker.observe(_lat(8, 8))
    assert spy.calls == 0


def test_release_only_on_shape_change(monkeypatch):
    spy = _EmptyCacheSpy(monkeypatch)
    tracker = loop_mod._BucketSwitchCacheRelease("cpu")
    tracker.observe(_lat(8, 8))     # 首个 batch：没有「上一个桶」，不清
    tracker.observe(_lat(8, 8))
    assert spy.calls == 0
    tracker.observe(_lat(6, 10))    # 切桶
    assert spy.calls == 1
    tracker.observe(_lat(6, 10))    # 同桶
    assert spy.calls == 1
    tracker.observe(_lat(8, 8))     # 再切
    assert spy.calls == 2


def test_batch_size_change_within_bucket_does_not_release(monkeypatch):
    # 同桶尾批（drop_last=False）bs 变小：张量更小、塞得进 cached block，不需要清
    spy = _EmptyCacheSpy(monkeypatch)
    tracker = loop_mod._BucketSwitchCacheRelease("cpu")
    tracker.observe(_lat(8, 8, bs=2))
    tracker.observe(_lat(8, 8, bs=1))
    assert spy.calls == 0


def test_each_switch_logged_at_debug_only(monkeypatch, caplog):
    _EmptyCacheSpy(monkeypatch)
    tracker = loop_mod._BucketSwitchCacheRelease("cpu")
    with caplog.at_level("DEBUG", logger=loop_mod.logger.name):
        tracker.observe(_lat(8, 8))
        tracker.observe(_lat(6, 10))
        tracker.observe(_lat(8, 8))
        tracker.observe(_lat(6, 10))
    recs = [r for r in caplog.records if "ARB 切桶" in r.getMessage()]
    assert [r.levelname for r in recs] == ["DEBUG"] * 3
    assert "8x8→6x10" in recs[0].getMessage()
    assert "6x10→8x8" in recs[1].getMessage()

    caplog.clear()
    with caplog.at_level("INFO", logger=loop_mod.logger.name):
        tracker.observe(_lat(8, 8))
    assert not [r for r in caplog.records if "ARB 切桶" in r.getMessage()]


# ---------------------------------------------------------------- loop 接线

def _batch(h, w):
    return {"captions": ["c"], "latents": _lat(h, w)}


def test_loop_releases_between_buckets_not_within(tmp_path, monkeypatch):
    # 桶 A ×2 → 桶 B ×2 → 桶 A ×1：两次切桶 → 恰好 2 次 empty_cache
    spy = _EmptyCacheSpy(monkeypatch)
    batches = [_batch(8, 8), _batch(8, 8), _batch(6, 10), _batch(6, 10), _batch(8, 8)]
    ctx = _make_ctx(tmp_path, batches, monkeypatch, grad_accum=1)
    loop_mod.run(ctx)
    assert ctx.global_step == 5
    assert spy.calls == 2


def test_loop_single_bucket_never_releases(tmp_path, monkeypatch):
    spy = _EmptyCacheSpy(monkeypatch)
    ctx = _make_ctx(tmp_path, [_batch(8, 8)] * 4, monkeypatch, grad_accum=1)
    loop_mod.run(ctx)
    assert ctx.global_step == 4
    assert spy.calls == 0
