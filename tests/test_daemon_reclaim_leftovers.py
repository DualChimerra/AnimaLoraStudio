"""daemon 的显存残骸清扫（上游 #499）。

**「模型没加载」不等于「显存干净」**：加载 / 采样中途 OOM 后 ``ModelCache.model``
仍是 None，但异常 traceback 与 frame 互相引用，钉住 frame locals 里半上卡的
state_dict —— refcount 收不掉，必须 gc。真机表现是出图失败后点「释放缓存」
显示成功、20GB 占用纹丝不动，只能重启服务。

本文件测的是**接线**（哪些路径调到了清扫、哪些路径不该调），不是分配器行为。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _daemon():
    import anima_daemon

    return anima_daemon


def test_unload_reclaims_leftovers_even_when_model_absent(monkeypatch):
    """**回归**：早退分支此前只在「刚丢过 text_stack」时才清，手动「释放缓存」
    落到这里就成了空转。现在无条件清扫。"""
    d = _daemon()
    calls = []
    monkeypatch.setattr(d, "_reclaim_cuda_leftovers", lambda: calls.append(1))

    cache = d.ModelCache()
    assert not cache.loaded
    cache.unload()

    assert calls, "早退分支必须清扫（gc + empty_cache + pinned 归还）"


def test_generate_worker_reclaims_leftovers_on_failure(monkeypatch):
    """任务失败后 worker 清扫一次 —— OOM 残骸不该留到下一次加载。

    清扫必须发生在 except 块结束（异常对象被隐式 del）之后，traceback 钉住的
    frame locals 才收得掉；成功路径不清，别把有用的 allocator cache 也倒掉。
    """
    d = _daemon()
    calls = []
    monkeypatch.setattr(d, "_reclaim_cuda_leftovers", lambda: calls.append(1))
    monkeypatch.setattr(d, "_emit_for", lambda *a, **k: None)

    def _boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(d, "_run_generate", _boom)
    d._run_generate_worker("req-x", 1, {}, Path("."), threading.Event())
    assert calls, "失败路径必须清扫"

    calls.clear()
    monkeypatch.setattr(d, "_run_generate", lambda *a, **k: None)
    d._run_generate_worker("req-y", 2, {}, Path("."), threading.Event())
    assert not calls, "成功路径不清扫"
