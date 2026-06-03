"""ADR 0010 — `ensure_train_manifest` 隐式 fallback 重建。

设计见 docs/adr/0010-preprocess-train-scope.md + docs/design/preprocess-train-scope-plan.md §3.2。

覆盖：
- 目标已存在 → 直接返回（不动现有 manifest 内容）
- 老 manifest 不存在 → 写空 v2 manifest
- 老 manifest 存在 + train/ 完全匹配
- 老 manifest 存在 + train/ 部分图（不在老 manifest 的不迁）
- 老 manifest 存在 + 老 entry 在 train/ 不存在（不迁该 entry）
- multi-crop fan-out 派生匹配（Y_c0.png + Y_c1.png 共享 origin）
- duplicate_removed 老 entry 不进新 manifest
- 老 manifest 损坏 → 按不存在处理
- 幂等：调 2 次结果一致 + 不重写已有 manifest
- 仅识别图像后缀（.txt 等不算 train 图）
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from studio.services.preprocess import manifest as pm


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """模拟项目目录：download/ + preprocess/ 存在，versions/v1/ 可按需建。"""
    (tmp_path / "download").mkdir()
    (tmp_path / "preprocess").mkdir()
    return tmp_path


def _train_dir(project_dir: Path, label: str = "v1") -> Path:
    d = project_dir / "versions" / label / "train"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_legacy(project_dir: Path, images: dict) -> None:
    pm.manifest_path(project_dir).write_text(
        json.dumps({"images": images}, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_train_manifest(project_dir: Path, label: str = "v1") -> dict:
    return json.loads(
        pm.train_manifest_path(project_dir, label).read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# Case 1: 目标已存在 → 直接返回，不动内容
# ---------------------------------------------------------------------------


def test_ensure_returns_existing_without_rewrite(project_dir: Path) -> None:
    train_dir = _train_dir(project_dir)
    existing = {
        "version": 2,
        "images": {"keep.png": {"origin": "keep.jpg", "mtime": 999, "size": 42}},
    }
    target = pm.train_manifest_path(project_dir, "v1")
    target.write_text(json.dumps(existing), encoding="utf-8")
    # 老 manifest 也存在但内容不同 — 不应该污染 target
    _write_legacy(project_dir, {"keep.png": {"origin": "DIFFERENT.jpg"}})

    returned = pm.ensure_train_manifest(project_dir, "v1")

    assert returned == target
    assert _read_train_manifest(project_dir) == existing


# ---------------------------------------------------------------------------
# Case 2 / 3: 老 manifest 不存在
# ---------------------------------------------------------------------------


def test_no_legacy_no_train_writes_empty(project_dir: Path) -> None:
    """train/ 不存在 + 老 manifest 不存在 → 创建目录 + 写空 manifest。"""
    target = pm.ensure_train_manifest(project_dir, "v1")
    assert target.exists()
    assert _read_train_manifest(project_dir) == {"version": 2, "images": {}}
    # 目录被建出来
    assert target.parent.exists()


def test_no_legacy_with_train_files_writes_empty(project_dir: Path) -> None:
    """train/ 有图但老 manifest 不存在 → 写空 manifest（无 origin 信息可继承）。"""
    train_dir = _train_dir(project_dir)
    (train_dir / "foo.png").write_bytes(b"\x89PNG")

    pm.ensure_train_manifest(project_dir, "v1")
    assert _read_train_manifest(project_dir) == {"version": 2, "images": {}}


# ---------------------------------------------------------------------------
# Case 4 / 5: 老 manifest 存在 + train/ 部分匹配
# ---------------------------------------------------------------------------


def test_legacy_full_match_rebuilds(project_dir: Path) -> None:
    train_dir = _train_dir(project_dir)
    (train_dir / "X.png").write_bytes(b"x")
    (train_dir / "Y.png").write_bytes(b"y")
    _write_legacy(project_dir, {
        "X.png": {"origin": "X.jpg", "mtime": 100, "size": 1000},
        "Y.png": {"origin": "Y.jpg", "mtime": 200, "size": 2000},
    })

    pm.ensure_train_manifest(project_dir, "v1")
    data = _read_train_manifest(project_dir)
    assert data["version"] == 2
    assert data["images"] == {
        "X.png": {"origin": "X.jpg", "mtime": 100, "size": 1000},
        "Y.png": {"origin": "Y.jpg", "mtime": 200, "size": 2000},
    }


def test_legacy_entry_missing_in_train_is_skipped(project_dir: Path) -> None:
    """老 entry 在 train/ 没对应文件 → 不进新 manifest。"""
    train_dir = _train_dir(project_dir)
    (train_dir / "X.png").write_bytes(b"x")
    # Y.png 在老 manifest 但不在 train/
    _write_legacy(project_dir, {
        "X.png": {"origin": "X.jpg", "mtime": 100, "size": 1000},
        "Y.png": {"origin": "Y.jpg", "mtime": 200, "size": 2000},
    })

    pm.ensure_train_manifest(project_dir, "v1")
    data = _read_train_manifest(project_dir)
    assert set(data["images"].keys()) == {"X.png"}


def test_train_file_not_in_legacy_is_skipped(project_dir: Path) -> None:
    """train/ 里有图但老 manifest 没记 → 不写 entry（用户手动拖入 / curate 复制
    的原图；新模型下这种图默认 origin = name，但本 fallback 不假设这点）。"""
    train_dir = _train_dir(project_dir)
    (train_dir / "X.png").write_bytes(b"x")
    (train_dir / "Z.png").write_bytes(b"z")  # 老 manifest 没记
    _write_legacy(project_dir, {
        "X.png": {"origin": "X.jpg", "mtime": 100, "size": 1000},
    })

    pm.ensure_train_manifest(project_dir, "v1")
    data = _read_train_manifest(project_dir)
    assert set(data["images"].keys()) == {"X.png"}


# ---------------------------------------------------------------------------
# Case 6: multi-crop fan-out（共享 origin）
# ---------------------------------------------------------------------------


def test_multi_crop_fan_out_preserved(project_dir: Path) -> None:
    train_dir = _train_dir(project_dir)
    (train_dir / "Y_c0.png").write_bytes(b"c0")
    (train_dir / "Y_c1.png").write_bytes(b"c1")
    _write_legacy(project_dir, {
        "Y_c0.png": {"origin": "Y.jpg", "mtime": 200, "size": 800},
        "Y_c1.png": {"origin": "Y.jpg", "mtime": 200, "size": 850},
    })

    pm.ensure_train_manifest(project_dir, "v1")
    data = _read_train_manifest(project_dir)
    assert data["images"]["Y_c0.png"]["origin"] == "Y.jpg"
    assert data["images"]["Y_c1.png"]["origin"] == "Y.jpg"


# ---------------------------------------------------------------------------
# Case 7: duplicate_removed 老 entry
# ---------------------------------------------------------------------------


def test_duplicate_removed_legacy_entry_skipped(project_dir: Path) -> None:
    """duplicate_removed 是人工去重审核记录（kind="duplicate_removed"），
    跟 train ↔ download 关系无关，不应进新 train manifest。"""
    train_dir = _train_dir(project_dir)
    (train_dir / "X.png").write_bytes(b"x")
    # 模拟用户清过去重，dup.jpg 在老 manifest 标了 duplicate_removed
    _write_legacy(project_dir, {
        "X.png": {"origin": "X.jpg", "mtime": 100, "size": 1000},
        "dup.jpg": {"kind": pm.DUPLICATE_REMOVED_KIND, "origin": "dup.jpg"},
    })
    # 即使 dup.jpg 物理存在于 train/（罕见但可能），也不该被写进
    (train_dir / "dup.jpg").write_bytes(b"dup")

    pm.ensure_train_manifest(project_dir, "v1")
    data = _read_train_manifest(project_dir)
    assert set(data["images"].keys()) == {"X.png"}


# ---------------------------------------------------------------------------
# Case 8: 老 manifest 损坏
# ---------------------------------------------------------------------------


def test_corrupted_legacy_treated_as_missing(project_dir: Path) -> None:
    train_dir = _train_dir(project_dir)
    (train_dir / "X.png").write_bytes(b"x")
    pm.manifest_path(project_dir).write_text("not json", encoding="utf-8")

    pm.ensure_train_manifest(project_dir, "v1")
    data = _read_train_manifest(project_dir)
    assert data == {"version": 2, "images": {}}


def test_legacy_invalid_shape_treated_as_missing(project_dir: Path) -> None:
    """非 dict 形状 → 当损坏处理（跟现有 `load()` 一致）。"""
    train_dir = _train_dir(project_dir)
    (train_dir / "X.png").write_bytes(b"x")
    pm.manifest_path(project_dir).write_text(json.dumps(["array"]), encoding="utf-8")

    pm.ensure_train_manifest(project_dir, "v1")
    data = _read_train_manifest(project_dir)
    assert data == {"version": 2, "images": {}}


# ---------------------------------------------------------------------------
# Case 9: 幂等
# ---------------------------------------------------------------------------


def test_idempotent_second_call_no_rewrite(project_dir: Path) -> None:
    train_dir = _train_dir(project_dir)
    (train_dir / "X.png").write_bytes(b"x")
    _write_legacy(project_dir, {
        "X.png": {"origin": "X.jpg", "mtime": 100, "size": 1000},
    })

    pm.ensure_train_manifest(project_dir, "v1")
    target = pm.train_manifest_path(project_dir, "v1")
    first_mtime = target.stat().st_mtime_ns

    # 第二次调用应该 short-circuit（O(1) stat），不动文件
    pm.ensure_train_manifest(project_dir, "v1")
    assert target.stat().st_mtime_ns == first_mtime


# ---------------------------------------------------------------------------
# Case 10: 仅识别图像后缀
# ---------------------------------------------------------------------------


def test_non_image_files_in_train_ignored(project_dir: Path) -> None:
    """train/ 里 .txt / .json 不算图像，不影响重建逻辑。"""
    train_dir = _train_dir(project_dir)
    (train_dir / "X.png").write_bytes(b"x")
    (train_dir / "X.txt").write_text("caption")  # caption 不应被当作图
    (train_dir / "manifest.json.tmp").write_text("{}")  # tmp 文件残留
    _write_legacy(project_dir, {
        "X.png": {"origin": "X.jpg", "mtime": 100, "size": 1000},
        "X.txt": {"origin": "X.txt", "mtime": 100, "size": 5},  # 老 manifest 误存（不会发生但防御）
    })

    pm.ensure_train_manifest(project_dir, "v1")
    data = _read_train_manifest(project_dir)
    # X.txt 老 entry 跟 train_files 集合（仅图像）不匹配 → 不迁
    assert set(data["images"].keys()) == {"X.png"}


# ---------------------------------------------------------------------------
# Case 11: 并发安全
# ---------------------------------------------------------------------------


def test_concurrent_ensure_yields_single_manifest(project_dir: Path) -> None:
    """多线程同时调 → `_LOCK` 双检查保证只重建一次，结果一致。"""
    train_dir = _train_dir(project_dir)
    (train_dir / "X.png").write_bytes(b"x")
    _write_legacy(project_dir, {
        "X.png": {"origin": "X.jpg", "mtime": 100, "size": 1000},
    })

    results: list[Path] = []
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            results.append(pm.ensure_train_manifest(project_dir, "v1"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(set(map(str, results))) == 1
    data = _read_train_manifest(project_dir)
    assert data["version"] == 2
    assert data["images"]["X.png"]["origin"] == "X.jpg"


# ---------------------------------------------------------------------------
# Case 12: 多 version 隔离
# ---------------------------------------------------------------------------


def test_multiple_versions_independent(project_dir: Path) -> None:
    """两个 version 各自重建独立 manifest，互不干扰。"""
    v1_train = _train_dir(project_dir, "v1")
    v2_train = _train_dir(project_dir, "v2")
    (v1_train / "X.png").write_bytes(b"x")
    (v2_train / "Y.png").write_bytes(b"y")
    _write_legacy(project_dir, {
        "X.png": {"origin": "X.jpg", "mtime": 100, "size": 1000},
        "Y.png": {"origin": "Y.jpg", "mtime": 200, "size": 2000},
    })

    pm.ensure_train_manifest(project_dir, "v1")
    pm.ensure_train_manifest(project_dir, "v2")

    v1_data = _read_train_manifest(project_dir, "v1")
    v2_data = _read_train_manifest(project_dir, "v2")
    assert set(v1_data["images"].keys()) == {"X.png"}
    assert set(v2_data["images"].keys()) == {"Y.png"}
