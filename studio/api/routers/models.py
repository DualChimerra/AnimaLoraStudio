"""模型 catalog / 下载（PR-6 commit 2 从 server.py 抽出）。

3 routes（PP7 第一刀域）：
    GET  /api/models/catalog         列已知模型 + 各自磁盘状态 + 当前下载状态
    GET  /api/models/path-defaults   当前 Settings 算出的 4 个模型字段绝对路径
    POST /api/models/download        启动后台下载，返回 status key
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from ..schemas.models import (
    AnimaCustomDownloadRequest,
    AnimaSelectRequest,
    ModelDownloadRequest,
)
from ... import secrets
from ...services import models as model_downloader

router = APIRouter()


@router.get("/api/models/catalog")
def get_models_catalog() -> dict[str, Any]:
    """前端设置页 Models 区块用：列已知模型 + 各自磁盘状态 + 当前下载状态。"""
    return model_downloader.build_catalog()


@router.get("/api/models/path-defaults")
def get_models_path_defaults() -> dict[str, str]:
    """当前 Settings 算出的 4 个模型字段绝对路径。

    给预设页 reset 按钮和「新建预设」初始填充用——这两个场景没有 project
    上下文，拿不到 /api/projects/{pid}/versions/{vid}/config 里的
    project_specific_defaults，所以单独开一个端点。
    """
    return model_downloader.default_paths_for_new_version()


@router.post("/api/models/download")
def start_model_download(body: ModelDownloadRequest) -> dict[str, Any]:
    """启动后台下载，立即返回 status key；前端通过 SSE
    (`model_download_changed`) 或轮询 catalog 看进度。"""
    try:
        key = model_downloader.trigger(body.model_id, body.variant)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    snap = model_downloader.get_status_snapshot()
    return {"key": key, "status": snap.get(key, {}).get("status", "running")}


@router.post("/api/models/select")
def select_anima(body: AnimaSelectRequest) -> dict[str, Any]:
    """切换默认 base 模型（写入 secrets.models.selected_anima）。

    接受预设 variant label 或已落盘的自定义本地文件名（在 diffusion_models/）；
    非法值（既非预设也不在 diffusion_models/ 找到）返回 404。逻辑镜像
    /api/upscalers/select。
    """
    variant = body.variant.strip()
    if not variant:
        raise HTTPException(400, "variant 不能为空")
    valid = variant in model_downloader.ANIMA_VARIANTS or variant == "latest"
    if not valid and not model_downloader.is_custom_anima(variant):
        raise HTTPException(
            404,
            f"base 模型不存在: {variant}（既非预设也未在 diffusion_models/ 找到）",
        )
    cur = secrets.load()
    new_models = cur.models.model_copy(update={"selected_anima": variant})
    new = cur.model_copy(update={"models": new_models})
    secrets.save(new)
    return {"selected": variant}


@router.post("/api/models/download_custom")
def start_anima_custom_download(body: AnimaCustomDownloadRequest) -> dict[str, Any]:
    """自定义 base 模型下载：用户填 HF/MS repo + 文件名，落到
    `{diffusion_models}/{filename}`。落地后可用 /api/models/select 注册为默认 base。

    复用通用 start_download_async；key 形如 `anima_main:custom:foo.safetensors`
    便于前端 SSE 过滤 + catalog 状态匹配。镜像 /api/upscalers/download_custom。
    """
    if body.source not in ("hf", "ms"):
        raise HTTPException(400, f"未知下载源: {body.source}")
    if not body.repo_id.strip() or not body.filename.strip():
        raise HTTPException(400, "repo_id / filename 不能为空")
    save_name = Path(body.filename).name
    if not save_name.lower().endswith(model_downloader.ANIMA_EXTS):
        raise HTTPException(400, f"仅支持 {model_downloader.ANIMA_EXTS} 扩展名")
    key = f"anima_main:custom:{save_name}"
    model_downloader.start_download_async(
        key,
        lambda log: model_downloader.download_anima_custom(
            body.source, body.repo_id, body.filename, on_log=log
        ),
    )
    snap = model_downloader.get_status_snapshot()
    return {"key": key, "status": snap.get(key, {}).get("status", "running")}
