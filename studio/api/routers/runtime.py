"""运行模式（Colab / Local）—— 本 fork 新增。

2 routes：
    GET /api/runtime         当前模式 + 探测结果 + 该模式下的环境事实
    PUT /api/runtime         持久化用户选择（前端首次进应用的选择框）

前端在启动时 GET 一次：`mode` 为空串 = 用户还没选过 → 弹选择框；选完 PUT。
`locked=true`（`ALS_RUNTIME_MODE` 已注入，Colab notebook 的启动 cell 会设）时
前端不弹框也不允许改 —— 环境已经替用户答过了。
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ...domain.errors import DomainError
from ...infrastructure import runtime_mode
from ...infrastructure.paths import STUDIO_DATA
from ... import secrets

logger = logging.getLogger(__name__)
router = APIRouter()


class RuntimeModePatch(BaseModel):
    mode: str


def _environment() -> dict[str, Any]:
    """选择框和设置区展示的环境事实。

    刻意都是「用户能据此判断自己选对没有」的东西：磁盘在哪、还剩多少、有没有
    GPU。GPU 名走 nvidia-smi 而不是 import torch —— 这个端点在 UI 启动路径上，
    不该为了一行字付 torch 的 import 成本（首次 import 秒级）。
    """
    total = free = None
    try:
        usage = shutil.disk_usage(STUDIO_DATA if STUDIO_DATA.exists() else Path.cwd())
        total, free = usage.total, usage.free
    except OSError:
        logger.debug("disk_usage failed for studio_data", exc_info=True)

    gpu = ""
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            import subprocess

            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                gpu = out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ""
        except Exception:
            logger.debug("nvidia-smi probe failed", exc_info=True)

    return {
        "platform": platform.system(),
        "python": sys.version.split()[0],
        "studio_data": str(STUDIO_DATA),
        "studio_data_env": os.environ.get("ALS_STUDIO_DATA", ""),
        "disk_total": total,
        "disk_free": free,
        "gpu": gpu,
    }


def _payload() -> dict[str, Any]:
    data = runtime_mode.describe()
    data["environment"] = _environment()
    return data


@router.get("/api/runtime")
def get_runtime() -> dict[str, Any]:
    return _payload()


@router.put("/api/runtime")
def put_runtime(body: RuntimeModePatch) -> dict[str, Any]:
    mode = runtime_mode.normalize(body.mode)
    if mode not in runtime_mode.MODES:
        raise DomainError(
            f"unknown runtime mode {body.mode!r}; expected one of {list(runtime_mode.MODES)}",
            code="runtime.invalid_mode",
            details={"mode": body.mode, "allowed": list(runtime_mode.MODES)},
            http_status=400,
        )
    override = runtime_mode.env_override()
    if override and override != mode:
        # env 是权威且不落盘：写 secrets 会造成"设置里显示 local、实际跑 colab"
        # 的分裂状态，不如直接拒绝并把原因说清楚。
        raise DomainError(
            f"runtime mode is pinned to {override!r} by the "
            f"{runtime_mode.ENV_OVERRIDE} environment variable",
            code="runtime.mode_locked",
            details={"mode": override, "env": runtime_mode.ENV_OVERRIDE},
            http_status=409,
        )
    secrets.update({"runtime": {"mode": mode, "asked": True}})
    return _payload()
