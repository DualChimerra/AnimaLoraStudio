"""第三方缓存收进仓库文件夹（本 fork）—— studio/infrastructure/local_cache.py。

动机：项目自带的目录（venv / models / studio_data）都在仓库里，但依赖库的缓存
默认全落在系统盘，光 pip 的 CUDA torch 轮子就 2-3GB。把项目放在专用 SSD 上的
用户会发现系统盘照样被吃掉，且删仓库带不走它们。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from studio.infrastructure import local_cache


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """给本文件一份**独立的** os.environ，且不含任何缓存变量。

    不能用 `monkeypatch.delenv(raising=False)`：对本来就不存在的键它不记录任何
    还原信息，于是 `apply()` 之后写进去的值会泄漏到整个测试会话 —— 后面
    cli.main() 那些用例会看到「PIP_CACHE_DIR 已被设过」而跳过，测出来的是假绿。
    整个替换成副本则天然隔离，monkeypatch 退出时把真 environ 换回来。
    """
    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ.pop(local_cache.OPT_OUT_ENV, None)
    for var in local_cache._CACHE_ENV_DIRS:
        os.environ.pop(var, None)


def test_apply_points_every_cache_into_the_repo(
    tmp_path: Path, clean_env: None  # noqa: ARG001
) -> None:
    applied = local_cache.apply(tmp_path)
    assert set(applied) == set(local_cache._CACHE_ENV_DIRS)
    for var, value in applied.items():
        assert os.environ[var] == value
        assert Path(value).is_relative_to(tmp_path / local_cache.CACHE_DIR_NAME)


def test_pip_cache_is_the_one_we_precreate(
    tmp_path: Path, clean_env: None  # noqa: ARG001
) -> None:
    """pip 在某些版本上缓存目录不存在就**静默放弃缓存** —— 于是「重装一次 venv
    就重下 2.5GB torch」会毫无征兆地发生。其余库都会自建，不预建以免仓库根
    多出一片空目录。"""
    local_cache.apply(tmp_path)
    root = tmp_path / local_cache.CACHE_DIR_NAME
    assert (root / "pip").is_dir()
    assert not (root / "huggingface").exists()


def test_never_overrides_an_existing_value(
    tmp_path: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch  # noqa: ARG001
) -> None:
    """显式设过 HF_HOME 的人多半在几个项目间共享大缓存，背着他改就是错的。"""
    monkeypatch.setenv("HF_HOME", "/somewhere/shared/hf")
    applied = local_cache.apply(tmp_path)
    assert "HF_HOME" not in applied
    assert os.environ["HF_HOME"] == "/somewhere/shared/hf"
    # 其余项照常接管
    assert "PIP_CACHE_DIR" in applied


def test_opt_out_disables_everything(
    tmp_path: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch  # noqa: ARG001
) -> None:
    monkeypatch.setenv(local_cache.OPT_OUT_ENV, "1")
    assert local_cache.apply(tmp_path) == {}
    assert "PIP_CACHE_DIR" not in os.environ
    assert not (tmp_path / local_cache.CACHE_DIR_NAME).exists()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_opt_out_accepted_spellings(
    tmp_path: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch, value: str  # noqa: ARG001
) -> None:
    monkeypatch.setenv(local_cache.OPT_OUT_ENV, value)
    assert local_cache.apply(tmp_path) == {}


def test_apply_is_idempotent(tmp_path: Path, clean_env: None) -> None:  # noqa: ARG001
    """第二次调用什么也不写 —— 入口不止一个（cli.py import 期、api/main.py、
    launcher），重复调用必须无害。"""
    first = local_cache.apply(tmp_path)
    assert first
    assert local_cache.apply(tmp_path) == {}


def test_apply_can_target_a_separate_env_dict(
    tmp_path: Path, clean_env: None  # noqa: ARG001
) -> None:
    """传 env 时不碰 os.environ（给「只想给子进程设」的调用方）。"""
    env: dict[str, str] = {}
    applied = local_cache.apply(tmp_path, env=env)
    assert applied
    assert env["HF_HOME"] == applied["HF_HOME"]
    assert "HF_HOME" not in os.environ


def test_hf_home_is_the_only_huggingface_variable(clean_env: None) -> None:  # noqa: ARG001
    """不再单独设 HUGGINGFACE_HUB_CACHE / TRANSFORMERS_CACHE：新版 huggingface_hub
    里它们已弃用并从 HF_HOME 派生，两处都设只会在库升级时留下互相矛盾的配置。"""
    assert "HF_HOME" in local_cache._CACHE_ENV_DIRS
    assert "HUGGINGFACE_HUB_CACHE" not in local_cache._CACHE_ENV_DIRS
    assert "TRANSFORMERS_CACHE" not in local_cache._CACHE_ENV_DIRS


def test_apply_survives_a_patched_os_name(
    tmp_path: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch  # noqa: ARG001
) -> None:
    """**回归**：`Path(...)` 按 `os.name` 选 flavour，于是在 Linux 上打了
    `os.name="nt"` 的调用方会让路径构造抛 NotImplementedError。

    这不是假想场景：cli.main() 会调用本函数，而 test_studio_cli 的 npm 提示
    用例正是靠 patch `os.name` 来测 Windows 分支的 —— 之前用 pathlib 拼路径时
    整个 pytest 会话会直接 INTERNALERROR。故路径拼接一律走 os.path。
    """
    monkeypatch.setattr(os, "name", "nt")
    applied = local_cache.apply(tmp_path)
    assert applied
    assert applied["HF_HOME"].endswith("huggingface")


def test_describe_reports_current_values(
    tmp_path: Path, clean_env: None  # noqa: ARG001
) -> None:
    assert set(local_cache.describe(tmp_path)) == set(local_cache._CACHE_ENV_DIRS)
    # 还没 apply → 全空串（= 库自己的默认位置）
    assert all(v == "" for v in local_cache.describe(tmp_path).values())
    local_cache.apply(tmp_path)
    assert all(v for v in local_cache.describe(tmp_path).values())
