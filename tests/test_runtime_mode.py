"""运行模式（Colab / Local）—— 探测、解析优先级、持久化与端点契约。

本 fork 新增：`studio/infrastructure/runtime_mode.py` + `/api/runtime`。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import secrets, server
from studio.cli import build_parser, _apply_runtime_mode_defaults
from studio.infrastructure import runtime_mode


@pytest.fixture
def secrets_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    sf = tmp_path / "secrets.json"
    monkeypatch.setattr(secrets, "SECRETS_FILE", sf)
    return sf


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """清掉一切会让探测翻成 colab 的变量 —— CI 上不会有，本地 notebook 里跑
    测试时会有，不清就是 flaky。"""
    for key in ("ALS_RUNTIME_MODE", "COLAB_RELEASE_TAG", "COLAB_GPU",
                "COLAB_JUPYTER_IP", "COLAB_BACKEND_VERSION",
                "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE",
                "KAGGLE_DATA_PROXY_TOKEN"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def client(secrets_file: Path, clean_env: None) -> TestClient:  # noqa: ARG001
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# normalize / detect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("local", "local"),
        ("colab", "colab"),
        ("  COLAB  ", "colab"),
        # 同类云端环境归到 colab 这一档
        ("kaggle", "colab"),
        ("cloud", "colab"),
        ("notebook", "colab"),
        ("pc", "local"),
        ("desktop", "local"),
        # 未知一律当"没选过"，绝不猜
        ("bogus", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize(raw: object, expected: str) -> None:
    assert runtime_mode.normalize(raw) == expected


def test_detect_local_by_default(clean_env: None) -> None:  # noqa: ARG001
    assert runtime_mode.detect() == "local"


@pytest.mark.parametrize(
    "env_key",
    ["COLAB_RELEASE_TAG", "COLAB_GPU", "COLAB_JUPYTER_IP", "KAGGLE_KERNEL_RUN_TYPE"],
)
def test_detect_colab_from_env(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, env_key: str  # noqa: ARG001
) -> None:
    monkeypatch.setenv(env_key, "1")
    assert runtime_mode.detect() == "colab"


def test_content_dir_alone_does_not_flip_to_colab(
    clean_env: None, monkeypatch: pytest.MonkeyPatch  # noqa: ARG001
) -> None:
    """/content 存在但没有任何 colab/kaggle 变量 → 仍判 local。

    本地也可能有 /content（挂载点 / 别人的脚本建的），单凭它翻 colab 会让本机
    用户拿到 0.0.0.0 绑定。它只作为 signals 里的诊断项。
    """
    monkeypatch.setattr(
        runtime_mode.Path, "is_dir", lambda self: str(self) == "/content"
    )
    assert runtime_mode.detect_signals()["content_dir"] is True
    assert runtime_mode.detect() == "local"


# ---------------------------------------------------------------------------
# 解析优先级：env override > secrets > 未选
# ---------------------------------------------------------------------------


def test_resolve_unset_when_never_chosen(secrets_file: Path, clean_env: None) -> None:  # noqa: ARG001
    """没选过时 resolve() 返回空串（不被探测值顶掉）。

    这一条是选择框能弹出来的前提 —— 如果 resolve() 悄悄回落到 detect()，
    `mode` 永远非空，前端就永远不问了。
    """
    assert runtime_mode.resolve() == ""
    # 而需要"现在就要一个值"的调用方拿到探测兜底
    assert runtime_mode.effective() == "local"


def test_stored_choice_wins_over_detection(
    secrets_file: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch  # noqa: ARG001
) -> None:
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab-2026")
    secrets.update({"runtime": {"mode": "local"}})
    assert runtime_mode.detect() == "colab"
    assert runtime_mode.resolve() == "local"
    assert runtime_mode.effective() == "local"


def test_env_override_wins_over_stored(
    secrets_file: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch  # noqa: ARG001
) -> None:
    secrets.update({"runtime": {"mode": "local"}})
    monkeypatch.setenv("ALS_RUNTIME_MODE", "colab")
    assert runtime_mode.resolve() == "colab"
    assert runtime_mode.describe()["locked"] is True


def test_invalid_env_override_is_ignored(
    secrets_file: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch  # noqa: ARG001
) -> None:
    secrets.update({"runtime": {"mode": "local"}})
    monkeypatch.setenv("ALS_RUNTIME_MODE", "banana")
    assert runtime_mode.env_override() == ""
    assert runtime_mode.resolve() == "local"


def test_invalid_stored_mode_falls_back_to_unset(secrets_file: Path, clean_env: None) -> None:  # noqa: ARG001
    """手改坏的 secrets.json 不该让应用按错模式跑 —— validator 归零成"没选过"，
    宁可多问一次。"""
    secrets_file.write_text('{"runtime": {"mode": "banana", "asked": true}}', encoding="utf-8")
    assert secrets.load().runtime.mode == ""
    assert runtime_mode.resolve() == ""


def test_choosing_a_mode_sets_asked(secrets_file: Path, clean_env: None) -> None:  # noqa: ARG001
    s = secrets.update({"runtime": {"mode": "colab"}})
    assert s.runtime.mode == "colab"
    assert s.runtime.asked is True


# ---------------------------------------------------------------------------
# /api/runtime
# ---------------------------------------------------------------------------


def test_get_runtime_reports_unset(client: TestClient) -> None:
    body = client.get("/api/runtime").json()
    assert body["mode"] == ""
    assert body["stored"] == ""
    assert body["detected"] == "local"
    assert body["effective"] == "local"
    assert body["locked"] is False
    assert body["modes"] == ["local", "colab"]
    assert "studio_data" in body["environment"]


def test_put_runtime_persists(client: TestClient) -> None:
    body = client.put("/api/runtime", json={"mode": "colab"}).json()
    assert body["mode"] == "colab"
    assert body["stored"] == "colab"
    assert secrets.load().runtime.mode == "colab"
    # 再 GET 一次拿到同样的值（落盘生效，不只是响应体）
    assert client.get("/api/runtime").json()["mode"] == "colab"


def test_put_runtime_rejects_unknown_mode(client: TestClient) -> None:
    r = client.put("/api/runtime", json={"mode": "banana"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "runtime.invalid_mode"
    assert secrets.load().runtime.mode == ""


def test_put_runtime_conflicts_with_env_pin(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """env 钉死时 PUT 别的值要 409 而不是静默写 secrets。

    写下去会造成"设置里显示 local、实际跑 colab"的分裂状态 —— env 不落盘，
    UI 读的是 resolve()，用户改不动却看不出为什么。
    """
    monkeypatch.setenv("ALS_RUNTIME_MODE", "colab")
    r = client.put("/api/runtime", json={"mode": "local"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "runtime.mode_locked"
    assert secrets.load().runtime.mode == ""
    # 与 env 一致的 PUT 仍然放行（前端"确认当前模式"是无害操作）
    assert client.put("/api/runtime", json={"mode": "colab"}).status_code == 200


# ---------------------------------------------------------------------------
# CLI 默认值随模式变
# ---------------------------------------------------------------------------


def _parse(argv: list[str]):
    args = build_parser().parse_args(argv)
    _apply_runtime_mode_defaults(args)
    return args


def test_cli_local_binds_loopback_and_opens_browser(
    secrets_file: Path, clean_env: None  # noqa: ARG001
) -> None:
    args = _parse(["run", "--mode", "local"])
    assert args.host == "127.0.0.1"
    assert args.no_browser is False


def test_cli_colab_binds_all_and_skips_browser(
    secrets_file: Path, clean_env: None  # noqa: ARG001
) -> None:
    """容器里没有浏览器可开，且 notebook 的端口代理要从外面连进来。"""
    args = _parse(["run", "--mode", "colab"])
    assert args.host == "0.0.0.0"  # noqa: S104 — colab 下这正是要的
    assert args.no_browser is True


def test_cli_explicit_host_beats_mode_default(
    secrets_file: Path, clean_env: None  # noqa: ARG001
) -> None:
    args = _parse(["run", "--mode", "colab", "--host", "127.0.0.1"])
    assert args.host == "127.0.0.1"


def test_cli_mode_flag_propagates_via_env(
    secrets_file: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch  # noqa: ARG001
) -> None:
    """`--mode` 要写进 ALS_RUNTIME_MODE —— server 子进程和 UI 才看到同一个值。"""
    _parse(["run", "--mode", "colab"])
    assert runtime_mode.env_override() == "colab"


def test_cli_follows_stored_choice_without_flag(
    secrets_file: Path, clean_env: None  # noqa: ARG001
) -> None:
    secrets.update({"runtime": {"mode": "colab"}})
    args = _parse(["run"])
    assert args.host == "0.0.0.0"  # noqa: S104
    assert args.no_browser is True
