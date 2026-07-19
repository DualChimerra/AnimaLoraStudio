"""图片获取 + 预处理（PR-6.5 commit 3 从 server.py 抽出）。

14 routes：

  下载 / 上传 (5)
    POST /api/projects/{pid}/download/estimate    booru count API 估算
    POST /api/projects/{pid}/download             启动 booru 下载 job
    POST /api/projects/{pid}/upload               多文件本地上传 (单图 / zip)
    POST /api/projects/{pid}/upload-from-path     服务端可见路径导入单图 / zip
    GET  /api/projects/{pid}/download/status      最近 download job + log_tail

  预处理 (13)
    POST /api/projects/{pid}/preprocess/start
    GET  /api/projects/{pid}/preprocess/status
    GET  /api/projects/{pid}/preprocess/files
    GET  /api/projects/{pid}/preprocess/duplicates/removed
    GET  /api/projects/{pid}/preprocess/crop/workspace
    POST /api/projects/{pid}/preprocess/crop
    POST /api/projects/{pid}/versions/{vid}/preprocess/inpaint/save
    GET  /api/projects/{pid}/versions/{vid}/preprocess/mask
    PUT  /api/projects/{pid}/versions/{vid}/preprocess/mask
    DELETE /api/projects/{pid}/versions/{vid}/preprocess/mask
    POST /api/projects/{pid}/preprocess/files/reset
    POST /api/projects/{pid}/preprocess/files/restore
    GET  /api/projects/{pid}/preprocess/thumb     [Deprecated] 兼容旧 URL

注：duplicates scan / apply（preprocess 子域）属于 commit 4（curation），不在本文件。
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, BinaryIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from ...errors import _validate_component_or_400  # noqa: F401  reserved for future use
from ....domain.errors import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
from ...schemas.ingestion import (
    DownloadRequest,
    EstimateRequest,
    PreprocessCropRequest,
    PreprocessRestoreRequest,
    PreprocessStartRequest,
    UploadFromPathBody,
)
from ._shared import _publish_job_state, _publish_project_state
from ....infrastructure.event_bus import bus
from .... import db, secrets
from ....services.projects import jobs as project_jobs, projects, versions
from ....paths import REPO_ROOT
from ....services.preprocess import core as preprocess_svc
from ....services import model_downloader
from ....services.booru import downloader
from ....services.preprocess import manifest as preprocess_manifest

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# /api/projects/{pid}/download + /api/projects/{pid}/files + /api/jobs/*  (PP2)
# ---------------------------------------------------------------------------


@router.post("/api/projects/{pid}/download/estimate")
def estimate_download(pid: int, body: EstimateRequest) -> dict[str, Any]:
    """先调 booru 的 count API 估算命中数，再让用户决定 count。

    返回 -1 表示未知（API 不支持精确计数）；前端按「下载全部」处理。
    """
    if body.api_source not in {"gelbooru", "danbooru"}:
        raise ValidationError(
            f"Unsupported image source: {body.api_source}",
            code="download.source_unsupported",
            details={"source": body.api_source}, http_status=400,
        )
    if not body.tag.strip():
        raise ValidationError(
            "Tag is required", code="download.tag_required", http_status=400,
        )
    if not secrets.has_credentials_for(body.api_source):
        raise ValidationError(
            f"No {body.api_source} credentials configured; add them on the Settings page",
            code="download.credentials_missing",
            details={"source": body.api_source}, http_status=400,
        )
    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise NotFoundError(
                "Project not found", code="project.not_found", details={"id": pid},
            )
    sec = secrets.load()
    if body.api_source == "danbooru":
        opts = downloader.DownloadOptions(
            tag=body.tag.strip(),
            count=1,
            api_source="danbooru",
            username=sec.danbooru.username,
            api_key=sec.danbooru.api_key,
            exclude_tags=list(sec.download.exclude_tags),
        )
    else:
        opts = downloader.DownloadOptions(
            tag=body.tag.strip(),
            count=1,
            api_source="gelbooru",
            user_id=sec.gelbooru.user_id,
            api_key=sec.gelbooru.api_key,
            exclude_tags=list(sec.download.exclude_tags),
        )
    count = downloader.estimate(opts)
    return {
        "tag": body.tag.strip(),
        "api_source": body.api_source,
        "exclude_tags": list(sec.download.exclude_tags),
        "effective_query": opts.effective_tag_query(),
        "count": count,
    }


@router.post("/api/projects/{pid}/download")
def start_download(pid: int, body: DownloadRequest) -> dict[str, Any]:
    if not body.tag.strip():
        raise ValidationError(
            "Tag is required", code="download.tag_required", http_status=400,
        )
    if body.count < 1:
        raise ValidationError(
            "Download count must be at least 1",
            code="download.count_invalid", http_status=400,
        )
    if body.api_source not in {"gelbooru", "danbooru"}:
        raise ValidationError(
            f"Unsupported image source: {body.api_source}",
            code="download.source_unsupported",
            details={"source": body.api_source}, http_status=400,
        )
    if not secrets.has_credentials_for(body.api_source):
        raise ValidationError(
            f"No {body.api_source} credentials configured; add them on the Settings page",
            code="download.credentials_missing",
            details={"source": body.api_source}, http_status=400,
        )

    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise NotFoundError(
                "Project not found", code="project.not_found", details={"id": pid},
            )
        job = project_jobs.create_job(
            conn,
            project_id=pid,
            kind="download",
            params={
                "tag": body.tag.strip(),
                "count": body.count,
                "api_source": body.api_source,
            },
        )
    _publish_job_state(job)
    return job


_UPLOAD_CHUNK = 1 << 20  # 1 MiB 流式落盘块


def _staging_root() -> Path:
    """上传暂存根目录 = STUDIO_DATA/uploads。

    用 ``project_jobs.JOB_LOGS_DIR.parent`` 派生（而非直接 STUDIO_DATA 常量），
    这样测试 monkeypatch JOB_LOGS_DIR 时暂存目录也跟着进 tmp，不污染仓库。
    """
    return project_jobs.JOB_LOGS_DIR.parent / "uploads"


def _safe_name(name: str) -> str:
    """剥掉路径段，只留 basename（防穿越）。"""
    return (name or "").replace("\\", "/").rsplit("/", 1)[-1]


async def _stage_upload_files(files: list[UploadFile], staging_dir: Path) -> int:
    """把上传文件流式落到 staging_dir（分块，不整包进内存）。返回落盘文件数。"""
    staging_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files:
        name = _safe_name(f.filename or "")
        if not name:
            continue
        dest = staging_dir / name
        with dest.open("wb") as out:
            while True:
                chunk = await f.read(_UPLOAD_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
        n += 1
    return n


def _publish_upload_log(pid: int, line: str) -> None:
    """推一行上传阶段日志给 SSE 订阅者（前端 TaskLogDrawer 显示）。

    `accept_many` 的 on_log 回调每 25 张 / 5s / 慢图触发一次（节流，不刷屏）。
    跟 logger.info 并存：前者给用户看，后者落 studio.log 给 debug 用。
    """
    bus.publish({"type": "project_upload_log", "project_id": pid, "line": line})


def _publish_upload_state(pid: int, status: str) -> None:
    """推 upload 状态转换（running / done / failed）给 SSE 订阅者。

    LogSource.status 用这个驱动 TaskLogDrawer 的徽标 + 自动展开（live 进入
    automatic open；终态保持展开但不再 auto-open）。
    """
    bus.publish({"type": "project_upload_state", "project_id": pid, "status": status})


@router.post("/api/projects/{pid}/upload")
async def upload_local_files(
    pid: int, files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """本地上传：单图 / zip 包 / 同名 .txt caption → 后台 job 处理。

    端点只把上传文件**流式落到 staging 目录**就立刻创建 upload job 返回（秒级），
    真正的解压 / convert_to_png / caption 配对在 upload_worker 后台跑。这样大 zip
    不会卡在同步请求里触发 Cloudflare 100s 超时（524）。前端轮询
    `upload/status` 看进度 + 结果。
    """
    if not files:
        raise ValidationError(
            "No files uploaded", code="dataset.no_files", http_status=400,
        )
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")

    staging_dir = _staging_root() / f"{pid}_{uuid.uuid4().hex}"
    try:
        staged = await _stage_upload_files(files, staging_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    if staged == 0:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise HTTPException(400, "没有有效文件名")

    with db.connection_for() as conn:
        job = project_jobs.create_job(
            conn,
            project_id=pid,
            kind="upload",
            params={"staging_dir": str(staging_dir), "source": "upload"},
        )
    _publish_job_state(job)
    return job


@router.post("/api/projects/{pid}/upload-from-path")
def upload_local_file_from_path(pid: int, body: UploadFromPathBody) -> dict[str, Any]:
    """从 server 可见路径导入单图 / zip → 后台 upload job（不拷贝、不删原文件）。"""
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise NotFoundError(
            "Project not found", code="project.not_found", details={"id": pid},
        )
    src = Path(body.path)
    if not src.is_absolute():
        src = (REPO_ROOT / src).resolve()
    else:
        src = src.resolve()
    if not src.exists():
        raise NotFoundError(
            f"Path not found: {body.path}",
            code="path.not_found", details={"path": body.path},
        )
    if not src.is_file():
        raise HTTPException(400, "请选择文件")

    with db.connection_for() as conn:
        job = project_jobs.create_job(
            conn,
            project_id=pid,
            kind="upload",
            params={"paths": [str(src)], "source": "path"},
        )
    _publish_job_state(job)
    return job


@router.get("/api/projects/{pid}/upload/status")
def upload_status(pid: int) -> dict[str, Any]:
    """最近一条 upload job + log_tail + 结果（added/skipped）。

    前端上传完字节后轮询这里：job 终态前显示「处理中」，done 后用 result 弹
    added/skipped 汇总并刷新图库；failed 显示 error_msg。
    """
    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise HTTPException(404, f"项目不存在: id={pid}")
        job = project_jobs.latest_for(conn, project_id=pid, kind="upload")
    if not job:
        return {"job": None, "log_tail": "", "result": None}
    log_path = Path(job.get("log_path") or "")
    tail = ""
    if log_path.exists():
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            tail = "\n".join(text.splitlines()[-50:])
        except Exception:
            tail = ""
    result = (
        project_jobs.read_result(job["id"])
        if job.get("status") == "done"
        else None
    )
    return {"job": job, "log_tail": tail, "result": result}


@router.get("/api/projects/{pid}/download/status")
def download_status(pid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise NotFoundError(
                "Project not found", code="project.not_found", details={"id": pid},
            )
        job = project_jobs.latest_for(conn, project_id=pid, kind="download")
    if not job:
        return {"job": None, "log_tail": ""}
    log_path = Path(job.get("log_path") or "")
    tail = ""
    if log_path.exists():
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            tail = "\n".join(text.splitlines()[-50:])
        except Exception:
            tail = ""
    return {"job": job, "log_tail": tail}


# ---------------------------------------------------------------------------
# ADR 0010 — train-scope preprocess endpoint 群
#
# `/api/projects/{pid}/versions/{vid}/preprocess/*` —— scope 收窄到 train 集合，
# 调 *_train 服务函数。
# ---------------------------------------------------------------------------


def _resolve_pv_or_404(pid: int, vid: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """拿 (project, version) 校验项目+版本存在且 vid 属于 pid。"""
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
        if not p:
            raise NotFoundError(
                "Project not found", code="project.not_found", details={"id": pid},
            )
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise NotFoundError(
                "Version not found", code="version.not_found", details={"id": vid},
            )
    return p, v


@router.post("/api/projects/{pid}/versions/{vid}/preprocess/start")
def start_preprocess_train(
    pid: int, vid: int, body: PreprocessStartRequest,
) -> dict[str, Any]:
    """ADR 0010 train scope: 对 versions/{label}/train/{folder}/ 跑 upscale。

    跟老 `start_preprocess` 同样的 body schema + validation；worker 看
    job.version_id 派发到 _run_upscale_train。
    """
    if body.mode not in ("all", "selected", "all_force"):
        raise ValidationError(
            f"Invalid preprocess mode: {body.mode}",
            code="preprocess.mode_invalid",
            details={"mode": body.mode}, http_status=400,
        )
    if body.tile_size <= 0:
        raise ValidationError(
            "Tile size must be greater than 0",
            code="preprocess.tile_size_invalid", http_status=400,
        )
    if body.device not in ("auto", "cuda", "cpu"):
        raise ValidationError(
            f"Invalid device: {body.device}",
            code="preprocess.device_invalid",
            details={"device": body.device}, http_status=400,
        )
    if body.target_area is not None and (
        body.target_area < 256 * 256 or body.target_area > 4096 * 4096
    ):
        raise ValidationError(
            f"Target area is out of range: {body.target_area}",
            code="preprocess.target_area_out_of_range",
            details={"value": body.target_area}, http_status=400,
        )
    try:
        target = model_downloader.upscaler_target(body.model)
    except ValueError as exc:
        raise NotFoundError(
            f'Upscaler "{body.model}" not found',
            code="upscaler.not_found", details={"name": body.model},
        ) from exc
    if not target.exists():
        raise ConflictError(
            f'Upscaler weights for "{body.model}" are not downloaded; '
            "download them under Settings → Preprocess",
            code="upscaler.not_downloaded", details={"name": body.model},
        )

    p, v = _resolve_pv_or_404(pid, vid)
    with db.connection_for() as conn:
        job = preprocess_svc.start_job_train(
            conn,
            project_id=pid,
            version_id=vid,
            mode=body.mode,
            names=body.names,
            model=body.model,
            tile_size=body.tile_size,
            tile_pad=body.tile_pad,
            device=body.device,
            target_area=body.target_area,
        )
    _publish_job_state(job)
    return job


@router.get("/api/projects/{pid}/versions/{vid}/preprocess/status")
def preprocess_status_train(pid: int, vid: int) -> dict[str, Any]:
    """最新 train-scope preprocess job + 日志尾 + train summary。"""
    p, v = _resolve_pv_or_404(pid, vid)
    with db.connection_for() as conn:
        job = project_jobs.latest_for(
            conn, project_id=pid, version_id=vid,
            kind=preprocess_svc.PREPROCESS_KIND,
        )
    log_tail = ""
    if job:
        log_path = Path(job.get("log_path") or "")
        if log_path.exists():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                log_tail = "\n".join(text.splitlines()[-50:])
            except Exception:  # noqa: BLE001
                log_tail = ""
    return {
        "job": job,
        "log_tail": log_tail,
        "summary": preprocess_svc.summary_train(p, v["label"]),
    }


@router.get("/api/projects/{pid}/versions/{vid}/preprocess/files")
def list_preprocess_files_train(pid: int, vid: int) -> dict[str, Any]:
    """train scope: 列 versions/{label}/train/ 全部图 + manifest 元数据。

    新模型下 list_pending / list_processed 二元概念消失（详 ADR 0010
    §Manifest schema v2）；统一返回 `images` 列表，前端按 entry 字段差异
    渲染状态徽章。response 仍含 `summary` 跟老 endpoint 一致。
    """
    p, v = _resolve_pv_or_404(pid, vid)
    return {
        "images": preprocess_svc.list_train_images(p, v["label"]),
        "summary": preprocess_svc.summary_train(p, v["label"]),
    }


@router.get("/api/projects/{pid}/versions/{vid}/preprocess/duplicates/removed")
def list_duplicate_removed_train(pid: int, vid: int) -> dict[str, Any]:
    """train scope: 「已删除」tab 列被去重审核标记的 manifest entries。"""
    p, v = _resolve_pv_or_404(pid, vid)
    return {
        "images": preprocess_svc.list_duplicate_removed_workspace_train(
            p, v["label"]
        ),
    }


@router.get("/api/projects/{pid}/versions/{vid}/preprocess/crop/workspace")
def list_crop_workspace_train_endpoint(pid: int, vid: int) -> dict[str, Any]:
    """train scope: 裁剪页工作集 = train/{folder}/{image} 全部 + 像素尺寸 +
    processed 标记。"""
    p, v = _resolve_pv_or_404(pid, vid)
    return {"images": preprocess_svc.list_crop_workspace_train(p, v["label"])}


@router.post("/api/projects/{pid}/versions/{vid}/preprocess/crop")
def start_preprocess_crop_train(
    pid: int, vid: int, body: PreprocessCropRequest,
) -> dict[str, Any]:
    """train scope: 创建 crop job。`crops` 的源文件名是 train rel path
    （`"1_data/X.png"`，跟 list_crop_workspace_train 返回 `name` 一致）。"""
    if not body.crops:
        raise ValidationError(
            "No crop regions provided",
            code="preprocess.crops_required", http_status=400,
        )
    _resolve_pv_or_404(pid, vid)
    crops_payload: dict[str, list[dict[str, Any]]] = {
        name: [r.model_dump() for r in rects]
        for name, rects in body.crops.items()
    }
    with db.connection_for() as conn:
        job = preprocess_svc.start_crop_job_train(
            conn, project_id=pid, version_id=vid, crops=crops_payload,
        )
    _publish_job_state(job)
    return job


@router.post("/api/projects/{pid}/versions/{vid}/preprocess/inpaint/save")
async def inpaint_save_train_endpoint(
    pid: int, vid: int,
    name: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """train scope 涂抹保存：前端 canvas 整图导出（PNG）覆盖 `train/{name}`。

    同步写盘（无 job）；产物统一 `{folder}/{stem}.png`、manifest 标
    processed=True（详 core.inpaint_save_train）。PIL 编码大图有秒级耗时，
    走 threadpool 不堵 event loop。
    """
    p, v = _resolve_pv_or_404(pid, vid)
    data = await file.read()
    res = await run_in_threadpool(
        preprocess_svc.inpaint_save_train, p, v["label"], name=name, data=data,
    )
    _publish_project_state(p)
    return res


@router.get("/api/projects/{pid}/versions/{vid}/preprocess/mask")
def get_mask_train_endpoint(pid: int, vid: int, name: str) -> FileResponse:
    """训练 mask sidecar（灰度 PNG，尺寸=源图）。无 mask → 404（前端以此
    区分「从未画过」）。前端带 mask_mtime cache-buster，这里只挂 no-cache。"""
    p, v = _resolve_pv_or_404(pid, vid)
    path = preprocess_svc.mask_file_train(p, v["label"], name=name)
    if path is None:
        raise NotFoundError(
            "Mask not found", code="preprocess.mask_not_found",
            details={"name": name},
        )
    return FileResponse(
        path, media_type="image/png",
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


@router.put("/api/projects/{pid}/versions/{vid}/preprocess/mask")
async def put_mask_train_endpoint(
    pid: int, vid: int,
    name: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """写入训练 mask（前端 mask 层导出的灰度 PNG）。同步写盘无 job。"""
    p, v = _resolve_pv_or_404(pid, vid)
    data = await file.read()
    res = await run_in_threadpool(
        preprocess_svc.mask_save_train, p, v["label"], name=name, data=data,
    )
    _publish_project_state(p)
    return res


@router.delete("/api/projects/{pid}/versions/{vid}/preprocess/mask")
def delete_mask_train_endpoint(pid: int, vid: int, name: str) -> dict[str, Any]:
    """删除训练 mask（= 该图恢复全图正常学习）。"""
    p, v = _resolve_pv_or_404(pid, vid)
    res = preprocess_svc.mask_delete_train(p, v["label"], name=name)
    _publish_project_state(p)
    return res


@router.post("/api/projects/{pid}/versions/{vid}/preprocess/files/reset")
def reset_preprocess_files_train(pid: int, vid: int) -> dict[str, Any]:
    """train scope: 清空 train manifest 状态（**不动** train/ 物理文件，详
    ADR 0010 §train_clear_all 决策）。下游 list_train_images 仍能列物理图，
    只是 entry 元数据没了；UI 走未处理状态徽章。
    """
    p, v = _resolve_pv_or_404(pid, vid)
    pdir = projects.project_dir(p["id"], p["slug"])
    preprocess_manifest.train_clear_all(pdir, v["label"])
    _publish_project_state(p)
    return {"ok": True}


@router.post("/api/projects/{pid}/versions/{vid}/preprocess/files/restore")
def restore_preprocess_files_train(
    pid: int, vid: int, body: PreprocessRestoreRequest,
) -> dict[str, Any]:
    """train scope restore: 从 `download/{entry.origin}` 复制覆盖回
    `train/{name}`。返回 `{restored, missing, no_origin}` 三组（详 ADR 0010
    §Restore 语义）；`no_origin` 给前端三选项 UI [拖入替换 / 保留 / 移除] 用。
    """
    if not body.names:
        return {"restored": [], "missing": [], "no_origin": []}
    p, v = _resolve_pv_or_404(pid, vid)
    res = preprocess_svc.restore_products_train(p, v["label"], body.names)
    if res["restored"]:
        _publish_project_state(p)
    return res


