"""上传 worker 子进程入口（fix 524）。

由 supervisor 启动：`python -m studio.workers.upload_worker --job-id N`。

为什么要 worker：`/upload` 同步解 zip + convert_to_png 重编码每张图，184MB
的包轻松 >100s，Cloudflare 100s 超时直接回 524。改成后台 job 后端点秒回，
真正的解压 / 转码在这里跑，前端轮询 `upload/status` 看进度 + 结果。

job.params：
- ``staging_dir``: 端点把上传的原始文件落到这个临时目录；处理完整目录后删掉它。
- ``paths``: 直接给定服务端可见路径列表（upload-from-path 用），处理后**不删**。
至少要有其一。

结果（added / skipped）写到 `jobs/{id}.result.json`，status endpoint 读回。
日志只走 stdout（见 download_worker 的说明）。
"""
from __future__ import annotations

import logging
from pathlib import Path

from studio import db, secrets

logger = logging.getLogger(__name__)
from studio.services.projects import jobs as project_jobs, projects
from studio.services.dataset import uploads as uploads_svc


def _gather_sources(params: dict) -> tuple[list[Path], Path | None]:
    """从 params 解出待处理文件列表 + 需清理的 staging 目录（无则 None）。"""
    staging = params.get("staging_dir")
    if staging:
        staging_dir = Path(staging)
        if not staging_dir.is_dir():
            return [], staging_dir
        sources = sorted(p for p in staging_dir.iterdir() if p.is_file())
        return sources, staging_dir
    paths = params.get("paths") or []
    return [Path(p) for p in paths], None


def run(job_id: int) -> int:
    """主体：返回退出码（0 成功 / 1 失败）。"""
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    if not job:
        print(f"[error] job {job_id} not found", flush=True)
        return 1
    if job["kind"] != "upload":
        print(f"[error] wrong kind: {job['kind']}", flush=True)
        return 1

    params = job.get("params_decoded") or {}

    def progress(line: str) -> None:
        print(line, flush=True)

    staging_dir: Path | None = None
    try:
        with db.connection_for() as conn:
            project = projects.get_project(conn, job["project_id"])
        if not project:
            progress(f"[error] project {job['project_id']} missing")
            return 1
        dest = projects.project_dir(project["id"], project["slug"]) / "download"

        sources, staging_dir = _gather_sources(params)
        if not sources:
            progress("[error] 没有待处理文件")
            project_jobs.write_result(job_id, {"added": [], "skipped": []})
            return 1

        sec = secrets.load()
        progress(
            f"[start] files={len(sources)} convert_to_png={sec.download.convert_to_png}"
        )
        result = uploads_svc.ingest_paths(
            sources, dest,
            convert_to_png=sec.download.convert_to_png,
            remove_alpha_channel=sec.download.remove_alpha_channel,
            on_progress=progress,
        )
        project_jobs.write_result(job_id, result.as_dict())
        progress(
            f"[done] added={len(result.added)} skipped={len(result.skipped)}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001 — 同 download_worker
        logger.exception("upload worker crashed (job_id=%s)", job_id)
        progress(f"[error] {exc}")
        project_jobs.write_result(job_id, {"added": [], "skipped": []})
        return 1
    finally:
        # staging 是端点为本次上传创建的临时目录，处理完无论成败都清掉。
        if staging_dir is not None and staging_dir.is_dir():
            import shutil
            shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    from ._base import worker_main
    worker_main(run)
