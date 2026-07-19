"""Studio FastAPI 守护进程的端点冒烟测试（P1 范围）。

测试只覆盖 server.py 暴露的 5 个端点。每个用例通过 monkeypatch 把
`studio.server` 模块里指向运行时数据的路径常量改写到 tmp_path，
避免污染仓库真实目录。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import server


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """把 server 模块里的路径全部指向 tmp_path 下的隔离目录。

    PR-6 commit 1：/samples / / 等 routes 搬到 api/routers/，监控 OUTPUT_DIR /
    WEB_DIST 的真实绑名在新模块。新位置和 server.py 同时 patch，保 old
    `server.OUTPUT_DIR` patch 不丢、新 handler 也能看到 fake 值。
    """
    from studio import db
    from studio.api.routers import root as _root_router
    from studio.api.routers import samples as _samples_router
    from studio.services.projects import projects
    output = tmp_path / "output"
    samples_dir = output / "samples"
    web_dist = tmp_path / "web_dist"  # 不创建即模拟未构建
    samples_dir.mkdir(parents=True)

    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server, "OUTPUT_DIR", output)
    monkeypatch.setattr(server, "WEB_DIST", web_dist)
    monkeypatch.setattr(_samples_router, "OUTPUT_DIR", output)
    monkeypatch.setattr(_root_router, "WEB_DIST", web_dist)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    return {
        "tmp": tmp_path,
        "db": dbfile,
        "output": output,
        "samples_dir": samples_dir,
        "web_dist": web_dist,
    }


@pytest.fixture
def client(isolated_paths: dict[str, Path]) -> TestClient:
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------

def test_health_returns_ok(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == server.app.version


def test_generate_sample_response_is_not_browser_cached(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """加密磁盘 cache 读路径：put PNG → GET /api/generate/{tid}/sample/{fn} →
    解密后 no-store 返回。

    本 fork：显式 init disk_cache —— 上游此处依赖其它测试触发过 lifespan init
    的执行顺序副作用（那些 system 路由测试已随 updater 移除）。
    """
    from studio.services.inference import disk_cache as generate_cache

    generate_cache.init(isolated_paths["tmp"] / ".cache" / "generate")

    try:
        generate_cache.cache_image(7, "sample.png", b"PNG", snapshot={"mode": "single"})
        resp = client.get("/api/generate/7/sample/sample.png")
        assert resp.status_code == 200
        assert resp.content == b"PNG"
        assert resp.headers["cache-control"] == "no-store"
    finally:
        generate_cache.drop_task(7)


# ---------------------------------------------------------------------------
# /api/state
# ---------------------------------------------------------------------------

def test_torch_status_proxies_service(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /api/torch/status 把 torch_setup.current_status() 透传给前端。"""
    from studio.services.runtime import torch as torch_setup
    monkeypatch.setattr(torch_setup, "current_status", lambda: {
        "installed": True,
        "version": "2.5.0+cpu",
        "cuda_build": "cpu",
        "cuda_available": False,
        "device_name": None,
        "cuda_detect": {"available": True, "driver_version": "555.86", "gpu_name": "RTX 5090"},
        "recommended_cu_tag": "cu128",
        "is_cpu_with_gpu": True,
        "is_cuda_build_unavailable": False,
    })
    resp = client.get("/api/torch/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_cpu_with_gpu"] is True
    assert body["recommended_cu_tag"] == "cu128"


def test_torch_reinstall_registers_marker_returns_pending(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """POST /api/torch/reinstall 不真装，写 marker 返回 pending。"""
    from studio.services.runtime import pending_install, torch as torch_setup
    monkeypatch.setattr(pending_install, "STUDIO_DATA", tmp_path)
    monkeypatch.setattr(pending_install, "PENDING_MARKER", tmp_path / ".pending-pip-install.json")
    monkeypatch.setattr(torch_setup, "_decide_target_tag", lambda _t: "cu128")

    resp = client.post("/api/torch/reinstall", json={"target": "auto"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["pending"] is True
    assert body["tag"] == "cu128"
    assert body["target"] == "auto"
    assert "studio.bat" in body["message"]
    # marker 文件已写
    assert (tmp_path / ".pending-pip-install.json").exists()


def test_torch_reinstall_invalid_target_returns_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.runtime import torch as torch_setup
    monkeypatch.setattr(
        torch_setup, "_decide_target_tag",
        lambda t: (_ for _ in ()).throw(ValueError(f"非法 target: {t!r}")),
    )
    resp = client.post("/api/torch/reinstall", json={"target": "xpu"})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "install.target_invalid"
    assert body["error"]["details"]["target"] == "xpu"


def test_flash_attention_status_returns_env_and_candidates(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /api/flash-attention/status 应返回 status + env + slim candidates + fetch_error。"""
    from studio.services.runtime import flash_attention as flash_attention_setup
    monkeypatch.setattr(flash_attention_setup, "current_status", lambda: {
        "installed": True, "version": "2.8.3"
    })
    monkeypatch.setattr(flash_attention_setup, "detect_env", lambda: {
        "python_tag": "cp311", "cuda_tag": "cu128", "cuda_ver": "12.8",
        "torch_tag": "torch2.5", "torch_ver": "2.5.0+cu128", "platform": "win_amd64",
    })
    monkeypatch.setattr(flash_attention_setup, "find_candidates", lambda _env: ([
        {
            "url": "https://x/wheel.whl",
            "name": "flash_attn-2.8.3+cu128torch2.5-cp311-cp311-win_amd64.whl",
            "score": 40,  # 应被剥掉
            "notes": [],
            "usable": True,
            "tags": {"cuda": "cu128"},  # 应被剥掉
        },
    ], None))

    resp = client.get("/api/flash-attention/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["installed"] is True
    assert body["version"] == "2.8.3"
    assert body["env"]["platform"] == "win_amd64"
    # candidates 只保留 url/name/notes/usable —— score / tags 不暴露给前端
    assert len(body["candidates"]) == 1
    c = body["candidates"][0]
    assert set(c.keys()) == {"url", "name", "notes", "usable"}
    assert body["fetch_error"] is None


def test_flash_attention_status_passes_fetch_error_through(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GitHub 限流 / 网络异常时 fetch_error 透传给 UI。"""
    from studio.services.runtime import flash_attention as flash_attention_setup
    monkeypatch.setattr(flash_attention_setup, "current_status", lambda: {
        "installed": False, "version": None,
    })
    monkeypatch.setattr(flash_attention_setup, "detect_env", lambda: {
        "python_tag": "cp311", "cuda_tag": None, "cuda_ver": None,
        "torch_tag": None, "torch_ver": None, "platform": "linux_x86_64",
    })
    monkeypatch.setattr(
        flash_attention_setup, "find_candidates",
        lambda _env: ([], "GitHub API 错误: API rate limit exceeded"),
    )
    resp = client.get("/api/flash-attention/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["candidates"] == []
    assert "rate limit" in body["fetch_error"]


def test_flash_attention_install_success(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.runtime import flash_attention as flash_attention_setup
    captured: dict = {}

    def fake_install(url):
        captured["url"] = url
        return {
            "installed": True, "version": "2.8.3",
            "url": url or "https://auto/wheel.whl",
            "stdout_tail": "Successfully installed",
            "restart_required": True,
        }

    monkeypatch.setattr(flash_attention_setup, "install", fake_install)
    resp = client.post("/api/flash-attention/install", json={"url": "https://x/manual.whl"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["installed"] is True
    assert body["restart_required"] is True
    assert captured["url"] == "https://x/manual.whl"


def test_flash_attention_install_url_null_uses_auto(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """前端不传 url（或显式 null）→ service 收到 None，走自动匹配。"""
    from studio.services.runtime import flash_attention as flash_attention_setup
    captured: dict = {}

    def fake_install(url):
        captured["url"] = url
        return {"installed": True, "version": "2.8.3", "url": "auto",
                "stdout_tail": "", "restart_required": True}

    monkeypatch.setattr(flash_attention_setup, "install", fake_install)
    resp = client.post("/api/flash-attention/install", json={"url": None})
    assert resp.status_code == 200
    assert captured["url"] is None


def test_flash_attention_install_failure_returns_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.runtime import flash_attention as flash_attention_setup

    def boom(_url):
        raise RuntimeError("pip install 失败:\nERROR: bad wheel")

    monkeypatch.setattr(flash_attention_setup, "install", boom)
    resp = client.post("/api/flash-attention/install", json={"url": "https://x/bad.whl"})
    assert resp.status_code == 500
    assert "bad wheel" in resp.json()["error"]["message"]


def test_state_missing_returns_empty(client: TestClient, isolated_paths: dict[str, Path]) -> None:
    """没有 task_id 也没有 running 任务时返回空状态。"""
    resp = client.get("/api/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["losses"] == []
    assert body["lr_history"] == []
    assert body["step"] == 0
    assert body["epoch"] == 0
    assert body["start_time"] is None


def _make_task_with_state(
    isolated_paths: dict[str, Path], payload: dict | str | None
) -> int:
    """建一个 task 并写 state 文件，返回 task_id。payload=None 表示不写文件。"""
    from studio import db as _db
    state_dir = isolated_paths["tmp"] / "states"
    state_dir.mkdir(exist_ok=True)
    state_file = state_dir / "state.json"
    if payload is not None:
        state_file.write_text(
            json.dumps(payload) if isinstance(payload, dict) else payload,
            encoding="utf-8",
        )
    with _db.connection_for(isolated_paths["db"]) as conn:
        tid = _db.create_task(conn, name="t", config_name="x")
        _db.update_task(conn, tid, monitor_state_path=str(state_file))
    return tid


def test_state_by_task_id_returns_parsed_json(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    payload = {
        "losses": [{"step": 1, "loss": 0.5, "time": 100.0}],
        "lr_history": [{"step": 1, "lr": 1e-4}],
        "epoch": 2,
        "step": 42,
        "total_steps": 1000,
        "speed": 1.23,
        "samples": [],
        "start_time": 1700000000.0,
        "config": {"lora_rank": 32},
    }
    tid = _make_task_with_state(isolated_paths, payload)
    resp = client.get(f"/api/state?task_id={tid}")
    assert resp.status_code == 200
    assert resp.json() == payload


def test_state_includes_eval_context_for_bound_task(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    from studio import db as _db
    from studio.services.projects import projects, versions

    payload = {
        "losses": [],
        "lr_history": [],
        "epoch": 1,
        "step": 10,
        "total_steps": 20,
        "speed": 1.0,
        "samples": [],
        "start_time": 1700000000.0,
        "config": {},
    }
    tid = _make_task_with_state(isolated_paths, payload)
    with _db.connection_for(isolated_paths["db"]) as conn:
        project = projects.create_project(conn, title="Eval Monitor")
        version = versions.create_version(conn, project_id=project["id"], label="v1")
        _db.update_task(
            conn,
            tid,
            project_id=project["id"],
            version_id=version["id"],
        )

    resp = client.get(f"/api/state?task_id={tid}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["project_id"] == project["id"]
    assert body["project_slug"] == project["slug"]
    assert body["version_id"] == version["id"]
    assert body["version_label"] == "v1"
    assert body["task_id"] == tid
    assert body["step"] == 10


def test_state_corrupt_returns_500(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    tid = _make_task_with_state(isolated_paths, "this is not json")
    resp = client.get(f"/api/state?task_id={tid}")
    assert resp.status_code == 500


def test_state_unknown_task_returns_empty(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    resp = client.get("/api/state?task_id=99999")
    assert resp.status_code == 200
    assert resp.json()["losses"] == []


def test_state_running_task_used_when_no_task_id(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """没给 task_id → 默认拉当前 running 的 task。"""
    payload = {"losses": [], "lr_history": [], "epoch": 0, "step": 7,
               "total_steps": 0, "speed": 0.0, "samples": [],
               "start_time": None, "config": {}}
    from studio import db as _db
    tid = _make_task_with_state(isolated_paths, payload)
    with _db.connection_for(isolated_paths["db"]) as conn:
        _db.update_task(conn, tid, status="running", started_at=1.0)
    resp = client.get("/api/state")
    assert resp.json()["step"] == 7


def test_state_max_points_downsamples_losses(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """PR #37：/api/state 兑现 max_points，losses/lr 长度超过时均匀降采样。"""
    losses = [{"step": i, "loss": 1.0 / (i + 1), "time": float(i)} for i in range(5000)]
    lr_history = [{"step": i, "lr": 1e-4} for i in range(5000)]
    optimizer_metrics_history = [{"step": i, "actual_lr": 1e-4, "d": 1e-4} for i in range(5000)]
    payload = {
        "losses": losses, "lr_history": lr_history,
        "optimizer_metrics_history": optimizer_metrics_history,
        "epoch": 0, "step": 4999,
        "total_steps": 5000, "speed": 0.0, "samples": [],
        "start_time": None, "config": {},
    }
    tid = _make_task_with_state(isolated_paths, payload)

    # max_points=500 → 都被压到 500
    resp = client.get(f"/api/state?task_id={tid}&max_points=500")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["losses"]) == 500
    assert len(body["lr_history"]) == 500
    assert len(body["optimizer_metrics_history"]) == 500
    # 首尾保留
    assert body["losses"][0]["step"] == 0
    assert body["losses"][-1]["step"] == 4999
    # 其他字段透传
    assert body["step"] == 4999
    assert body["total_steps"] == 5000


def test_state_max_points_zero_disables_downsample(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """max_points=0 (无穷) → 不降采样，原样返回。"""
    losses = [{"step": i, "loss": 0.0} for i in range(100)]
    payload = {"losses": losses, "lr_history": [], "epoch": 0, "step": 99,
               "total_steps": 100, "speed": 0.0, "samples": [],
               "start_time": None, "config": {}}
    tid = _make_task_with_state(isolated_paths, payload)
    resp = client.get(f"/api/state?task_id={tid}&max_points=0")
    assert resp.status_code == 200
    assert len(resp.json()["losses"]) == 100


def test_state_default_returns_full_payload(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """新默认（PR #43）：不传 max_points 等价于 max_points=0，返回全量历史。

    10k 步训练 cold start 时用户能拿到完整数据；想降采样的 caller 必须显式
    传具体数字。
    """
    losses = [{"step": i, "loss": 0.1} for i in range(10000)]
    payload = {
        "losses": losses, "lr_history": [], "epoch": 0, "step": 9999,
        "total_steps": 10000, "speed": 0.0, "samples": [],
        "start_time": None, "config": {},
    }
    tid = _make_task_with_state(isolated_paths, payload)
    # 不传 max_points 任何参数
    resp = client.get(f"/api/state?task_id={tid}")
    assert resp.status_code == 200
    assert len(resp.json()["losses"]) == 10000


# ---------------------------------------------------------------------------
# /samples/{filename}
# ---------------------------------------------------------------------------

def test_sample_404_for_missing(client: TestClient) -> None:
    resp = client.get("/samples/does_not_exist.png")
    assert resp.status_code == 404


def test_sample_returns_file(client: TestClient, isolated_paths: dict[str, Path]) -> None:
    img_path = isolated_paths["samples_dir"] / "step_42.png"
    img_path.write_bytes(b"fake-png-bytes")
    resp = client.get("/samples/step_42.png")
    assert resp.status_code == 200
    assert resp.content == b"fake-png-bytes"


@pytest.mark.parametrize("bad", ["../secret.txt", "..\\secret.txt", "sub/dir.png", "sub\\dir.png"])
def test_sample_blocks_traversal(client: TestClient, bad: str) -> None:
    """`/samples/{name}` 不允许斜杠 / 反斜杠 / 上级路径。"""
    resp = client.get(f"/samples/{bad}")
    # 含 `/` 或 `\` 的会被路由层拆成多段（404），含 `..` 的被显式 400 拒绝；
    # 任何一种都不应该 200。
    assert resp.status_code != 200


def test_sample_with_task_id_finds_in_output_samples(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """回归 Q4：anima_train 把 sample 写到 `output_dir/samples/`，端点应在
    `monitor_state_path 同级 output/samples/` 也能命中（之前只查了同级 samples/）。"""
    from studio import db as _db
    state_path = isolated_paths["tmp"] / "v1" / "monitor_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{}", encoding="utf-8")
    out_samples = state_path.parent / "output" / "samples"
    out_samples.mkdir(parents=True)
    (out_samples / "step_0_baseline_0.png").write_bytes(b"sample-bytes")

    with _db.connection_for(isolated_paths["db"]) as conn:
        tid = _db.create_task(conn, name="t", config_name="x")
        _db.update_task(conn, tid, monitor_state_path=str(state_path))

    resp = client.get(f"/samples/step_0_baseline_0.png?task_id={tid}")
    assert resp.status_code == 200, resp.text
    assert resp.content == b"sample-bytes"


def test_sample_with_task_id_finds_in_state_dir_samples(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """旧约定路径（monitor_state.json 同级 samples/）仍兼容。"""
    from studio import db as _db
    state_path = isolated_paths["tmp"] / "v2" / "monitor_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{}", encoding="utf-8")
    samples = state_path.parent / "samples"
    samples.mkdir()
    (samples / "step_5.png").write_bytes(b"old-layout")

    with _db.connection_for(isolated_paths["db"]) as conn:
        tid = _db.create_task(conn, name="t", config_name="x")
        _db.update_task(conn, tid, monitor_state_path=str(state_path))

    resp = client.get(f"/samples/step_5.png?task_id={tid}")
    assert resp.status_code == 200
    assert resp.content == b"old-layout"


# ---------------------------------------------------------------------------
# /
# ---------------------------------------------------------------------------

def test_root_serves_index_when_built(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """ADR 0012：前端 dist 存在时，/ 直接吐 index.html，不再重定向。"""
    web_dist = isolated_paths["web_dist"]
    web_dist.mkdir(parents=True, exist_ok=True)
    (web_dist / "index.html").write_text("<!doctype html><title>anima</title>", encoding="utf-8")
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<title>anima</title>" in resp.text


def test_root_fallback_when_no_dist(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """前端未构建时返回 JSON 提示，而不是 404 / 跳转。"""
    assert not isolated_paths["web_dist"].exists()
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 200
    body = resp.json()
    assert "AnimaStudio" in body["message"]


def test_legacy_studio_path_redirects_to_root(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """ADR 0012 legacy：老 /studio/... 链接一次性 307 跳到根路径，保留 query。"""
    resp = client.get("/studio/projects/1?tab=log", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/projects/1?tab=log"
    # 裸 /studio → /
    resp = client.get("/studio", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/"


# ---------------------------------------------------------------------------
# /api/system/restart (ADR 0002 / PR-A)
# ---------------------------------------------------------------------------

def test_uvicorn_run_bounds_graceful_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main() 必须给 uvicorn 传 timeout_graceful_shutdown 上限。

    浏览器开着时 /api/events 的 SSE 长连接不主动断，graceful shutdown 默认
    无限等 →「Waiting for connections to close」卡死；且 py3.12+ 的
    asyncio.Server.wait_closed() 等全部活跃连接，二次 Ctrl+C 设 force_exit
    也解不开。没有这个上限，终端 Ctrl+C 永远关不掉 server。
    """
    import uvicorn

    from studio.api import main as main_mod

    captured: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: captured.update(kw))
    monkeypatch.setattr("sys.argv", ["anima-studio"])
    main_mod.main()
    timeout = captured.get("timeout_graceful_shutdown")
    assert isinstance(timeout, (int, float)) and timeout > 0


def test_cancelled_asgi_noise_filter_scope() -> None:
    """shutdown 超时取消 SSE 连接时，uvicorn 把 CancelledError 按
    「Exception in ASGI application」打 ERROR traceback —— 主动取消不是
    应用错误，应被过滤；其它 ASGI 异常 / 其它 message 必须照常放行。"""
    import asyncio

    from studio.api.lifespan import _CancelledAsgiNoiseFilter

    f = _CancelledAsgiNoiseFilter()

    def record(msg: str, exc: BaseException | None) -> logging.LogRecord:
        return logging.LogRecord(
            name="uvicorn.error", level=logging.ERROR, pathname=__file__,
            lineno=1, msg=msg, args=(),
            exc_info=(type(exc), exc, None) if exc is not None else None,
        )

    # 目标噪声：吞
    assert not f.filter(
        record("Exception in ASGI application\n", asyncio.CancelledError())
    )
    # 真实应用异常：放行
    assert f.filter(
        record("Exception in ASGI application\n", RuntimeError("boom"))
    )
    # 无异常信息 / 其它 message：放行
    assert f.filter(record("Exception in ASGI application\n", None))
    assert f.filter(record("Cancel 2 running task(s)", asyncio.CancelledError()))


# ---------------------------------------------------------------------------
# /api/system/version (ADR 0002 / PR-B)
# ---------------------------------------------------------------------------

