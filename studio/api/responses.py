"""共享响应常量 / 响应工厂（PR-5 起从 server.py 抽出）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.responses import FileResponse

from ..services.dataset import thumb_cache

# /api/state 在 task_id 不存在 / 没 task / state.json 缺失时返回的空 state，
# 保持前端 monitor 页能稳定渲染（不报错也不显示 "loading"）。
EMPTY_STATE: dict[str, Any] = {
    "losses": [],
    "lr_history": [],
    "epoch": 0,
    "total_epochs": 0,
    "step": 0,
    "total_steps": 0,
    "speed": 0.0,
    "samples": [],
    "start_time": None,
    "config": {},
}


def _thumb_response(src: Path, size: int, immutable: bool = False) -> FileResponse:
    """统一 thumb 响应：弱 etag（基于 src mtime+size）。

    默认 `no-cache 强制重验`：早先用 `max-age=86400` 会让浏览器记住所有响应
    24h，包括重启过渡期的失败响应；用户视角就是「重启后图片加载不了」。改用
    etag + no-cache 后，浏览器每次发条件请求，命中走 304 几 ms。

    `immutable=True`：内容一旦写就不会变（且 URL 已是稳定唯一 key，如训练
    采样图 `?task_id=N` + 带 epoch/step 的文件名）。此时用
    `max-age=1y, immutable` 让浏览器**完全不再回源**——云端走 cloudflared
    隧道时，每张图省掉一次 RTT 的 304 往返，监控页采样图条秒开、重开不再
    重新加载。重启过渡期的隐患不适用：只有 `.exists()` 命中、真返回图片时
    才发这个头，404 / 失败响应不带缓存头。

    PR-6：从 server.py 抽到 api/responses.py 给 samples router 和 server.py 内的
    project_thumb（PR-6.5 之前还留 server.py）共用。
    """
    out = thumb_cache.get_or_make_thumb(src, size)
    try:
        mtime_ns = out.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    etag = f'W/"{mtime_ns}-{size}"'
    cache_control = (
        "public, max-age=31536000, immutable"
        if immutable
        else "no-cache, must-revalidate"
    )
    return FileResponse(
        out,
        headers={
            "Cache-Control": cache_control,
            "ETag": etag,
        },
    )
