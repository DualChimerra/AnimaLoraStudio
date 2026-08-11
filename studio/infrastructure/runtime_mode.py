"""运行模式（Colab / Local）—— 探测、解析与生效值。

本 fork 同时服务两类用户：

- **Colab / Kaggle 等云端 notebook**：进程跑在临时容器里，浏览器在另一台机器
  上，端口要经 notebook 的代理暴露；磁盘随时回收，`studio_data` 常被
  `ALS_STUDIO_DATA` 指到本机快速盘再单独同步到 Drive。
- **本地机器（Windows / Linux / macOS）**：浏览器和进程同机，监听 127.0.0.1
  即可，起完自动开浏览器；磁盘是持久的，`studio_data/` 就在仓库旁边。

两者的合理默认值互相冲突（bind host、是否开浏览器、是否提示 Drive 同步等），
以前只能靠 CLI flag 各自记住。这里把它抽成一个显式的一等设置：

    secrets.runtime.mode = "" | "local" | "colab"

`""` = 用户还没选过 —— 前端进应用时弹一次选择框（`RuntimeModePicker`），选完
落盘就不再问。探测结果（`detect()`）只用来**预选**，不替用户做决定，因为探测
永远有误判空间（自建 JupyterHub、docker 里跑本地训练等）。

环境变量 `ALS_RUNTIME_MODE` 优先级最高且不落盘：Colab notebook 的启动 cell 注
入它，云端用户开箱即用不会被问；本地用户不设它，走 secrets 里的选择。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

Mode = Literal["local", "colab"]

MODES: tuple[str, ...] = ("local", "colab")

#: 未选择时的存储值。刻意不用 None —— pydantic 字段保持 str，前端判空即可。
UNSET = ""

ENV_OVERRIDE = "ALS_RUNTIME_MODE"


def normalize(value: object) -> str:
    """把任意入参规整成 `"local"` / `"colab"` / `""`（未知一律当未选）。"""
    text = str(value or "").strip().lower()
    if text in ("kaggle", "cloud", "notebook"):
        # 同类云端 notebook 环境统一归到 colab 这一档（行为需求完全一致）。
        return "colab"
    if text in ("pc", "desktop", "localhost"):
        return "local"
    return text if text in MODES else UNSET


def detect_signals() -> dict[str, bool]:
    """收集云端 notebook 的判据，供 `detect()` 与 `/api/runtime` 诊断展示。

    每条都只读环境 / 文件系统，不 import 重模块（`google.colab` 只在 Colab
    里存在，用 sys.modules 查而不是真 import —— 真 import 在非 Colab 上是几
    十毫秒的失败路径，且某些镜像里会打印告警）。
    """
    env = os.environ
    return {
        # Colab 运行时自己注入的变量，最可靠的一条。
        "colab_env": any(
            key in env
            for key in ("COLAB_RELEASE_TAG", "COLAB_GPU", "COLAB_JUPYTER_IP",
                        "COLAB_BACKEND_VERSION")
        ),
        # 已经被 import 过的 colab 模块（notebook 里 `from google.colab import
        # drive` 几乎是必经步骤）。
        "colab_module": "google.colab" in sys.modules,
        # Colab 的工作目录约定。单独一条不足以判定（有人本地也建 /content），
        # 所以下面 detect() 要求它跟别的信号叠加。
        "content_dir": Path("/content").is_dir(),
        # Kaggle notebook。
        "kaggle_env": any(
            key in env
            for key in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE",
                        "KAGGLE_DATA_PROXY_TOKEN")
        ),
    }


def detect() -> Mode:
    """猜当前环境是云端 notebook 还是本地机器。

    宁可猜 `local`：猜错成 colab 会让本地用户拿到 0.0.0.0 绑定和"不开浏览器"，
    比反过来更容易让人卡住；而且选择框会把探测结果摆出来让用户改。
    """
    signals = detect_signals()
    if signals["colab_env"] or signals["colab_module"] or signals["kaggle_env"]:
        return "colab"
    # `content_dir` 只进 signals 供用户自查，不单独作判据 —— 本地也可能有
    # /content（挂载点、别人的脚本建的），单凭它翻成 colab 误伤面太大。
    return "local"


def env_override() -> str:
    """`ALS_RUNTIME_MODE` 的规整值（未设或非法 → `""`）。"""
    return normalize(os.environ.get(ENV_OVERRIDE))


def stored() -> str:
    """secrets 里用户选过的模式（未选 → `""`）。

    secrets 依赖 STUDIO_DATA 路径，import 放在函数内避免 infrastructure 内部
    出现 paths → secrets → paths 的循环。
    """
    try:
        from . import secrets as secrets_mod

        return normalize(secrets_mod.load().runtime.mode)
    except Exception:
        # secrets.json 损坏 / 尚未创建时不该阻断启动，当作"没选过"。
        return UNSET


def resolve() -> str:
    """最终生效的模式：env override → 用户选择 → `""`（未选）。

    注意这里**不**回落到 `detect()` —— 「没选过」是一个前端要能看见的状态，
    被探测值悄悄顶掉的话选择框就永远不弹了。需要一个可用值的调用方（如
    `effective()`）自己决定怎么兜底。
    """
    return env_override() or stored()


def effective() -> Mode:
    """给需要"现在就要一个值"的调用方（CLI / 后端默认值）用的兜底解析。

    未选时用探测结果，保证 CLI 在用户还没进过 UI 时也有合理行为。
    """
    resolved = resolve()
    if resolved in MODES:
        return resolved  # type: ignore[return-value]
    return detect()


def is_colab() -> bool:
    return effective() == "colab"


def describe() -> dict[str, object]:
    """`/api/runtime` 的载荷：既给前端渲染，也给用户自查为什么被判成某模式。"""
    return {
        "mode": resolve(),
        "stored": stored(),
        "detected": detect(),
        "effective": effective(),
        "env_override": env_override(),
        "locked": bool(env_override()),
        "signals": detect_signals(),
        "modes": list(MODES),
    }
