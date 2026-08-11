"""把第三方库的缓存目录收进仓库文件夹（本 fork 新增）。

**为什么需要**：这个项目自带的东西已经都在仓库里了 —— `venv/`、`models/`、
`studio_data/`。但它依赖的库各自还有自己的缓存，默认全落在用户主目录 /
系统盘：

| 缓存 | 默认位置 | 量级 |
|---|---|---|
| pip 轮子缓存 | `%LOCALAPPDATA%\\pip\\Cache` / `~/.cache/pip` | **数 GB**（光 CUDA torch 轮子就 2-3GB） |
| HuggingFace hub | `~/.cache/huggingface` | 数 GB（eval 模型、tokenizer、下载中转） |
| torch hub | `~/.cache/torch` | 数百 MB |
| ModelScope | `~/.cache/modelscope` | 数 GB |
| npm | `~/.npm` / `%AppData%\\npm-cache` | 数百 MB |
| triton | `~/.triton` | 数百 MB |

把整个项目放在专用 SSD 上的用户（本 fork 的典型本机场景）会发现系统盘照样
被吃掉几个 GB，而且删仓库不会带走它们。本模块在进程启动最早期把这些库的
缓存环境变量指到 `<仓库>/.cache/<name>`，于是「整个项目 = 一个文件夹」真正
成立：拷走它就是全部，删掉它就干净。

**三条纪律**：

1. **绝不覆盖用户已经设过的值。** 显式设了 `HF_HOME` 的人多半是在几个项目
   之间共享大缓存，这里再改就是背着他改行为。
2. **可整体关掉**：`ALS_SYSTEM_CACHES=1` → 本模块 no-op，全部回到库的默认
   位置（想让多个 checkout 共享 pip 缓存时用）。
3. **只用标准库**，且只碰环境变量：它要在 venv 还没建好、依赖一个都没装的
   时候就能跑（`tools/launcher.py` 在第一次 `pip install` 之前调它 —— pip
   缓存正是最大的那一份，晚设一步就白设了）。
"""
from __future__ import annotations

import os
from pathlib import Path

#: 设成 1/true/yes/on 时本模块不做任何事（缓存回到库的默认位置）。
OPT_OUT_ENV = "ALS_SYSTEM_CACHES"

#: 仓库内缓存根目录名。已在 .gitignore 里。
CACHE_DIR_NAME = ".cache"

#: 环境变量 → `<cache_root>/` 下的子目录名。
#:
#: `HF_HOME` 一条就覆盖了 hub 缓存与 transformers 缓存（新版
#: huggingface_hub 里 `HUGGINGFACE_HUB_CACHE` / `TRANSFORMERS_CACHE` 都已弃用
#: 并从它派生），所以不再单独设那两个 —— 设了反而会在库升级时留下互相矛盾
#: 的双份配置。
#:
#: `XDG_CACHE_HOME` 是 Linux 上的兜底：没有专用变量的库（matplotlib、fontconfig
#: 等）都读它。Windows 上无害（读它的库本来就不在那儿跑）。
_CACHE_ENV_DIRS: dict[str, str] = {
    "PIP_CACHE_DIR": "pip",
    "HF_HOME": "huggingface",
    "TORCH_HOME": "torch",
    "MODELSCOPE_CACHE": "modelscope",
    "TRITON_CACHE_DIR": "triton",
    "WANDB_CACHE_DIR": "wandb",
    # npm 读小写的 `npm_config_<key>` 形式；前端构建的缓存同样是几百 MB。
    "npm_config_cache": "npm",
    "XDG_CACHE_HOME": "xdg",
}


def opted_out() -> bool:
    return str(os.environ.get(OPT_OUT_ENV, "")).strip().lower() in {
        "1", "true", "yes", "on",
    }


def cache_root(repo_root: Path) -> Path:
    """展示用。注意**不要**在这里 `Path(repo_root)` 重新包一层 —— 见 apply()。"""
    return repo_root / CACHE_DIR_NAME


def apply(repo_root: Path, *, env: dict[str, str] | None = None) -> dict[str, str]:
    """把缓存环境变量指进 `<repo_root>/.cache/`；返回本次**实际写入**的项。

    `env` 缺省就地改 `os.environ`（当前进程 + 它之后 spawn 的子进程都继承）；
    传入 dict 则只改那份（给「只想给子进程设」的调用方，如 supervisor）。

    路径拼接走 `os.path` 而不是 `pathlib`：产出的本来就是环境变量字符串，用不上
    Path；而且 `Path("...")` 会**按 `os.name` 选 flavour**，于是任何 monkeypatch
    了 `os.name` 的调用方（cli 的 npm 提示测试就是）会让这里在 Linux 上构造
    `WindowsPath` 直接抛 NotImplementedError。字符串拼接没有这个耦合。

    目录**不在这里创建** —— 各库自己会建，而提前建一堆空目录会让仓库根多出
    一片噪声（用户开了 `ALS_SYSTEM_CACHES` 之后更是徒留空壳）。唯一的例外是
    pip：它在某些版本上不会自建缓存目录，见下方。

    返回空 dict 的两种情况：用户整体关掉了（`ALS_SYSTEM_CACHES`），或每一项
    都已经被显式设过。
    """
    target = env if env is not None else os.environ
    if opted_out():
        return {}

    root = os.path.join(os.fspath(repo_root), CACHE_DIR_NAME)
    applied: dict[str, str] = {}
    for var, subdir in _CACHE_ENV_DIRS.items():
        # 已有值 = 用户（或外层 launcher）的显式选择，不动。
        if str(target.get(var, "")).strip():
            continue
        value = os.path.join(root, subdir)
        target[var] = value
        applied[var] = value

    # pip 是唯一需要预建的：部分版本在缓存目录不存在时直接放弃缓存而不报错，
    # 于是「重装一次 venv 就重下 2.5GB torch」这件事会静默发生。
    if "PIP_CACHE_DIR" in applied:
        try:
            os.makedirs(applied["PIP_CACHE_DIR"], exist_ok=True)
        except OSError:
            # 只读挂载 / 权限不足：让 pip 走它自己的默认位置，不阻断启动。
            del target["PIP_CACHE_DIR"]
            del applied["PIP_CACHE_DIR"]

    return applied


def describe(repo_root: Path) -> dict[str, str]:
    """当前生效的缓存位置（诊断用，不改任何环境变量）。

    值取 `os.environ` 的实际值 —— 用户自己设过的、本模块设的、以及关掉之后
    库的默认位置（显示为空串），一眼能分清。
    """
    return {var: str(os.environ.get(var, "")) for var in _CACHE_ENV_DIRS}
