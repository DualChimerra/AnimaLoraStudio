"""自定义 VAE / 文本编码器：注册、选中、解析与失效回退。

主模型（transformer）的本地权重早有 `anima` / `krea2` 两个候选 domain；本组
测试覆盖同一条链路新增的两类权重：

    vae                 族无关共享 VAE 单文件 → models.selected_vae
    anima_te / krea2_te 按族的文本编码器目录  → models.selected_te[family]

不变量与主模型一致：注册只登记路径（不下载不复制）、移除不动磁盘、移除当前
选中回退官方默认、选中值指向的文件/目录失效时解析回落官方落点。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import secrets, server
from studio.services import models as model_downloader


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """secrets.json 与 models_root 都隔离到 tmp_path。"""
    sf = tmp_path / "secrets.json"
    monkeypatch.setattr(secrets, "SECRETS_FILE", sf)
    sf.write_text(
        json.dumps({"models": {"root": str(tmp_path / "models")}}),
        encoding="utf-8",
    )
    return TestClient(server.app)


def _rows(catalog: dict, domain: str) -> list[dict]:
    return catalog["model_sources"][domain]


def _make_te_dir(root: Path, name: str) -> Path:
    """最小可用的 transformers 目录（就绪判据只看 config.json）。"""
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"w")
    return d


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------


def test_vae_catalog_lists_official_row_selected_by_default(
    client: TestClient,
) -> None:
    rows = _rows(client.get("/api/models/catalog").json(), "vae")
    assert [r["kind"] for r in rows] == ["preset"]
    official = rows[0]
    # 官方行的 value 是空串 = "跟随官方落点"（selected_vae 的默认语义）
    assert official["value"] == ""
    assert official["is_current"] is True
    assert official["removable"] is False
    assert official["download_id"] == "anima_vae"


def test_register_and_select_local_vae_drives_all_families(
    client: TestClient, tmp_path: Path
) -> None:
    vae = tmp_path / "my-vae.safetensors"
    vae.write_bytes(b"weights")

    res = client.post(
        "/api/model-sources/vae", json={"kind": "local", "path": str(vae)},
    )
    assert res.status_code == 200
    local = [r for r in _rows(res.json(), "vae") if r["kind"] == "local"]
    assert [r["value"] for r in local] == [str(vae)]
    assert local[0]["exists"] is True
    assert local[0]["deletable"] is False   # 本地文件永不从 UI 删除

    secrets.update({"models": {"selected_vae": str(vae)}})
    # VAE 是族无关共享资产：两族的新建 version 都跟随这一个选择
    for family in ("anima", "krea2"):
        paths = model_downloader.default_paths_for_new_version(family=family)
        assert paths["vae_path"] == str(vae)

    rows = _rows(client.get("/api/models/catalog").json(), "vae")
    assert [r["is_current"] for r in rows] == [False, True]


def test_local_vae_must_be_safetensors_file(
    client: TestClient, tmp_path: Path
) -> None:
    wrong = tmp_path / "vae.ckpt"
    wrong.write_bytes(b"w")
    assert client.post(
        "/api/model-sources/vae", json={"kind": "local", "path": str(wrong)},
    ).status_code == 400
    assert client.post(
        "/api/model-sources/vae",
        json={"kind": "local", "path": str(tmp_path / "nope.safetensors")},
    ).status_code == 400


def test_vae_download_candidate_rejected_with_actionable_error(
    client: TestClient,
) -> None:
    """VAE / TE 目前只支持选本地文件——下载型候选明确报错而非静默存下。"""
    res = client.post(
        "/api/model-sources/vae",
        json={"kind": "download", "repo": "some/repo", "filename": "v.safetensors"},
    )
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "model_source.download_unsupported"


def test_removing_selected_vae_falls_back_to_official(
    client: TestClient, tmp_path: Path
) -> None:
    vae = tmp_path / "my-vae.safetensors"
    vae.write_bytes(b"weights")
    client.post(
        "/api/model-sources/vae", json={"kind": "local", "path": str(vae)},
    )
    secrets.update({"models": {"selected_vae": str(vae)}})

    res = client.request(
        "DELETE", "/api/model-sources/vae",
        json={"kind": "local", "path": str(vae)},
    )
    assert res.status_code == 200
    assert secrets.load().models.selected_vae == ""
    assert vae.exists()                      # 移除 ≠ 删除
    paths = model_downloader.default_paths_for_new_version()
    assert paths["vae_path"].endswith("qwen_image_vae.safetensors")


def test_missing_local_vae_falls_back_instead_of_dead_path(
    client: TestClient, tmp_path: Path
) -> None:
    vae = tmp_path / "gone.safetensors"
    vae.write_bytes(b"weights")
    client.post(
        "/api/model-sources/vae", json={"kind": "local", "path": str(vae)},
    )
    secrets.update({"models": {"selected_vae": str(vae)}})
    vae.unlink()

    paths = model_downloader.default_paths_for_new_version()
    assert paths["vae_path"].endswith("qwen_image_vae.safetensors")


# ---------------------------------------------------------------------------
# 文本编码器（按族）
# ---------------------------------------------------------------------------


def test_register_and_select_local_text_encoder_per_family(
    client: TestClient, tmp_path: Path
) -> None:
    anima_te = _make_te_dir(tmp_path, "my-qwen3")
    krea2_te = _make_te_dir(tmp_path, "my-qwen3-vl")

    assert client.post(
        "/api/model-sources/anima_te",
        json={"kind": "local", "path": str(anima_te)},
    ).status_code == 200
    assert client.post(
        "/api/model-sources/krea2_te",
        json={"kind": "local", "path": str(krea2_te)},
    ).status_code == 200

    secrets.update({"models": {"selected_te": {
        "anima": str(anima_te), "krea2": str(krea2_te),
    }}})
    assert model_downloader.default_paths_for_new_version(
        family="anima")["text_encoder_path"] == str(anima_te)
    assert model_downloader.default_paths_for_new_version(
        family="krea2")["text_encoder_path"] == str(krea2_te)

    catalog = client.get("/api/models/catalog").json()
    # 官方 variant 行与本地行同构，选中的只有本地那条
    anima_rows = _rows(catalog, "anima_te")
    assert [r["kind"] for r in anima_rows] == ["preset", "local"]
    assert [r["is_current"] for r in anima_rows] == [False, True]
    krea2_rows = _rows(catalog, "krea2_te")
    assert [r["kind"] for r in krea2_rows] == ["preset", "preset", "local"]
    assert [r["is_current"] for r in krea2_rows] == [False, False, True]
    # krea2 TE 卡回显选中值原样（前端据此显示「自定义」）
    assert catalog["krea2_text_encoder"]["selected"] == str(krea2_te)


def test_local_text_encoder_requires_config_json(
    client: TestClient, tmp_path: Path
) -> None:
    bare = tmp_path / "no-config"
    bare.mkdir()
    (bare / "model.safetensors").write_bytes(b"w")
    res = client.post(
        "/api/model-sources/krea2_te", json={"kind": "local", "path": str(bare)},
    )
    assert res.status_code == 400
    assert res.json()["error"]["details"]["missing"] == ["config.json"]


def test_removing_selected_text_encoder_falls_back_to_family_default(
    client: TestClient, tmp_path: Path
) -> None:
    te = _make_te_dir(tmp_path, "my-qwen3-vl")
    client.post(
        "/api/model-sources/krea2_te", json={"kind": "local", "path": str(te)},
    )
    secrets.update({"models": {"selected_te": {"krea2": str(te)}}})

    client.request(
        "DELETE", "/api/model-sources/krea2_te",
        json={"kind": "local", "path": str(te)},
    )
    # krea2 的官方默认 = bf16 目录；anima 侧默认是空串（只有一份官方目录）
    assert secrets.load().models.selected_te["krea2"] == "bf16"
    assert te.is_dir()
    paths = model_downloader.default_paths_for_new_version(family="krea2")
    assert paths["text_encoder_path"].endswith("Qwen_Qwen3-VL-4B-Instruct")


def test_missing_local_text_encoder_falls_back(
    client: TestClient, tmp_path: Path
) -> None:
    te = _make_te_dir(tmp_path, "vanishing")
    client.post(
        "/api/model-sources/anima_te", json={"kind": "local", "path": str(te)},
    )
    secrets.update({"models": {"selected_te": {"anima": str(te)}}})
    (te / "config.json").unlink()

    paths = model_downloader.default_paths_for_new_version(family="anima")
    assert paths["text_encoder_path"].endswith("text_encoders")


def test_text_encoder_local_paths_stay_out_of_models_custom(
    client: TestClient, tmp_path: Path
) -> None:
    """models.custom 是「族本地主模型」的兼容读面——VAE / TE 不能混进去，
    否则底模下拉会把编码器目录当 transformer 列出来。"""
    te = _make_te_dir(tmp_path, "my-qwen3")
    vae = tmp_path / "my-vae.safetensors"
    vae.write_bytes(b"w")
    client.post(
        "/api/model-sources/anima_te", json={"kind": "local", "path": str(te)},
    )
    client.post(
        "/api/model-sources/vae", json={"kind": "local", "path": str(vae)},
    )
    custom = secrets.load().models.custom
    assert "anima_te" not in custom and "vae" not in custom
    assert custom.get("anima", []) == []


# ---------------------------------------------------------------------------
# 工作模式（本地主模型挂在哪个族）
# ---------------------------------------------------------------------------


def test_local_main_model_family_switch_moves_between_domains(
    client: TestClient, tmp_path: Path
) -> None:
    """前端「工作模式」下拉 = 换个 domain 重登记同一条路径：族决定训练配置
    默认值与配套 VAE / TE 解析。"""
    weights = tmp_path / "community-ft.safetensors"
    weights.write_bytes(b"w")
    client.post(
        "/api/model-sources/anima", json={"kind": "local", "path": str(weights)},
    )
    secrets.update({"models": {"selected": {"anima": str(weights)}}})

    client.post(
        "/api/model-sources/krea2", json={"kind": "local", "path": str(weights)},
    )
    catalog = client.request(
        "DELETE", "/api/model-sources/anima",
        json={"kind": "local", "path": str(weights)},
    ).json()

    s = secrets.load()
    assert s.models.custom.get("anima", []) == []
    assert s.models.custom["krea2"] == [str(weights)]
    # 原族的选中值回退官方最新 variant（权重已不在该族名下）
    assert s.models.selected["anima"] == "1.0"
    krea2_local = [r for r in _rows(catalog, "krea2") if r["kind"] == "local"]
    assert [r["value"] for r in krea2_local] == [str(weights)]
