#!/usr/bin/env python
"""本机一键启动器 —— `AnimaLoraStudio.exe` 的源码（本 fork 新增）。

面向的是「下载了仓库、双击就想用」的本地用户：Windows 上 `studio.bat` 要先会
开终端、知道 PowerShell 得写 `.\\`、遇到没装 Python 时只能看到一行英文报错就
消失。这个启动器把同一套 bootstrap 用 Python 重写，编译成单文件 exe 后双击即
可，出错时留在屏幕上讲人话。

它**不是**把整个应用打进 exe：torch / CUDA 轮子有好几 GB，塞进 PyInstaller 既
不现实也没法按用户显卡选对版本。exe 只是引导程序（几 MB，纯 stdlib），干的事
和 `studio.bat` 一样：

    找仓库 → 建/复用 venv → 按 GPU 装 torch → 装 requirements → 起 studio

因此本文件**必须只用标准库**：它要在 venv 还不存在、requirements 一个都没装的
时候跑起来。

用法（源码形式与 exe 形式等价）：
    python tools/launcher.py                本机模式启动
    python tools/launcher.py --mode colab   传给 studio 的模式（云端一般直接用
                                            notebook，这里主要给测试用）
    python tools/launcher.py --port 8800    透传给 `python -m studio run`
    python tools/launcher.py --reinstall    删掉 venv 重建（studio_data 不动）
    python tools/launcher.py --repo D:\\path 手动指定仓库位置

未识别的参数原样透传给 `python -m studio run`。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any, NoReturn, Optional, Sequence

APP_NAME = "AnimaLora Studio"

#: 判定「这个目录是不是仓库根」的标记。两个都要有 —— 只看 requirements.txt 会
#: 命中一堆无关的 Python 项目。
REPO_MARKERS = ("requirements.txt", "studio/__init__.py")

#: cli.py 的 installer 自更新协议：cli 退出码 42 = 启动器文件本身被改过，要重新
#: 加载自己。见 docs/adr/0002-webui-self-update.md 与 studio.bat 的同名分支。
INSTALLER_RELOAD_EXIT_CODE = 42

PYPI_MIRROR = "https://mirrors.cloud.tencent.com/pypi/simple/"

MIN_PYTHON = (3, 10)


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def say(msg: str) -> None:
    print(f"[studio] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[studio] WARNING: {msg}", file=sys.stderr, flush=True)


def die(msg: str, *hints: str) -> NoReturn:
    """报错 + 可操作建议，然后按住窗口。

    双击启动的场景里控制台会随进程退出立刻关掉，用户根本读不到报错 —— 这是
    「exe 一闪而过」类问题的全部成因，所以出错路径一律 pause。
    """
    print(f"\n[studio] ERROR: {msg}", file=sys.stderr, flush=True)
    for hint in hints:
        print(f"         → {hint}", file=sys.stderr, flush=True)
    pause()
    raise SystemExit(1)


def pause() -> None:
    """只在真的连着终端时等回车（CI / 管道里不要挂住）。"""
    if not sys.stdin or not sys.stdin.isatty():
        return
    try:
        input("\n[studio] Press Enter to close...")
    except (EOFError, KeyboardInterrupt):
        pass


# ---------------------------------------------------------------------------
# 定位仓库
# ---------------------------------------------------------------------------


def launcher_dir() -> Path:
    """exe / 脚本自身所在目录。

    PyInstaller onefile 会把内容解到临时目录再执行，`__file__` 指向那个临时目
    录 —— 找仓库必须用 `sys.executable`（真正的 exe 位置）。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def looks_like_repo(path: Path) -> bool:
    return all((path / marker).exists() for marker in REPO_MARKERS)


def find_repo(explicit: Optional[str]) -> Path:
    """按「显式指定 → exe 所在目录及其上级 → 当前工作目录」的顺序找仓库根。

    上溯是为了让 exe 放在 `tools/`、`dist/` 之类子目录里也能用；层数给到 4，
    再深就不是误放而是放错地方了，应当报错而不是继续猜。
    """
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not looks_like_repo(path):
            die(
                f"{path} is not an AnimaLoraStudio checkout",
                f"expected to find {' and '.join(REPO_MARKERS)} there",
            )
        return path

    candidates: list[Path] = []
    start = launcher_dir()
    candidates.extend([start, *list(start.parents)[:4]])
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *list(cwd.parents)[:2]])

    for candidate in candidates:
        if looks_like_repo(candidate):
            return candidate

    die(
        "could not find the AnimaLoraStudio files next to this launcher",
        f"put {APP_NAME} in the folder that contains requirements.txt and studio/",
        "or pass --repo <path-to-the-folder>",
    )


# ---------------------------------------------------------------------------
# 缓存目录
# ---------------------------------------------------------------------------


def load_local_cache(repo: Path) -> Optional[Any]:
    """按路径 import `studio/infrastructure/local_cache.py`。

    按路径 import 而不是把环境变量清单复制一份进来：清单只该有一处权威源，
    抄成两份早晚会漂。这里能这么做是因为此刻仓库已经找到、而那个模块是纯
    标准库的（venv 还不存在也 import 得动）。

    失败返回 None —— 缓存位置是优化不是正确性，加载不了就让各库用自己的默认
    位置照常启动。但要 warn：静默会让「以为收进来了、其实没有」无从察觉。
    """
    module_path = repo / "studio" / "infrastructure" / "local_cache.py"
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_als_local_cache", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:  # noqa: BLE001
        warn(f"could not redirect caches into the project folder ({exc});"
             " they will use their default system locations")
        return None


def apply_local_caches(repo: Path) -> dict[str, str]:
    """把第三方缓存指进 `<仓库>/.cache/`。

    **必须在第一次 `pip install` 之前调用** —— pip 的轮子缓存是所有缓存里最大
    的一份（CUDA torch 单个轮子 2-3GB），晚一步设就已经落到系统盘了。
    """
    module = load_local_cache(repo)
    return dict(module.apply(repo)) if module is not None else {}


# ---------------------------------------------------------------------------
# venv
# ---------------------------------------------------------------------------


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def find_existing_venv(repo: Path) -> Optional[Path]:
    """`venv/` 优先、`.venv/` 兜底 —— 与 studio.bat / studio.sh 的顺序一致。"""
    for name in ("venv", ".venv"):
        candidate = repo / name
        if venv_python(candidate).exists():
            return candidate
    return None


def bootstrap_python() -> list[str]:
    """挑一个用来**创建** venv 的解释器。

    frozen 的 exe 里 `sys.executable` 是 exe 自己 —— 它没有 venv 模块也没法
    `-m venv`，所以必须去系统上找真 Python。Windows 上优先 `py -3`：很多机器
    为了兼容老项目在 PATH 上留着一个旧 `python`，而 py launcher 会挑最新的
    3.x。非 frozen 运行时直接用当前解释器。
    """
    if not getattr(sys, "frozen", False):
        return [sys.executable]

    if os.name == "nt" and shutil.which("py"):
        return ["py", "-3"]
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            return [found]
    die(
        "Python 3.10+ is not installed (or not on PATH)",
        "install it from https://www.python.org/downloads/",
        'tick "Add python.exe to PATH" in the installer',
    )


def check_version(python_argv: Sequence[str], *, label: str) -> None:
    """低于 3.10 只警告不拦：部分依赖会装不上，但用户可能只是想跑已装好的
    环境，硬拦住反而堵死自救路径。"""
    code = f"import sys;sys.exit(0 if sys.version_info>={MIN_PYTHON} else 1)"
    try:
        rc = subprocess.call([*python_argv, "-c", code])
    except OSError:
        return
    if rc != 0:
        want = ".".join(str(p) for p in MIN_PYTHON)
        warn(f"{label} is older than Python {want}; some dependencies may fail to install")


def create_venv(repo: Path) -> Path:
    """建 venv。frozen 时用系统 Python 起子进程，否则用内置 venv 模块。"""
    venv_dir = repo / "venv"
    say(f"no virtual environment found; creating {venv_dir} (first run takes a few minutes)")
    if getattr(sys, "frozen", False):
        argv = bootstrap_python()
        check_version(argv, label="python")
        rc = subprocess.call([*argv, "-m", "venv", str(venv_dir)])
        if rc != 0:
            die(
                "failed to create the virtual environment",
                "make sure Python 3.10+ is installed and you can write to this folder",
            )
    else:
        check_version([sys.executable], label=sys.executable)
        venv.EnvBuilder(with_pip=True).create(venv_dir)
    return venv_dir


# ---------------------------------------------------------------------------
# 依赖
# ---------------------------------------------------------------------------


def pip_install(py: Path, args: Sequence[str], *, what: str) -> bool:
    """先走官方 PyPI，失败再退到腾讯镜像（国内直连常年超时）。返回是否成功。"""
    base = [str(py), "-m", "pip", "install", *args]
    if subprocess.call(base) == 0:
        return True
    say(f"{what}: pip failed, retrying via mirror...")
    return subprocess.call([*base, "-i", PYPI_MIRROR]) == 0


def install_torch(py: Path, repo: Path, forced_tag: Optional[str]) -> None:
    """先按 GPU 装对的 torch，再让 requirements.txt 跑。

    顺序是关键：requirements.txt 里只写 `torch>=2.0.0`，先跑它的话 pip 会拉
    PyPI 上的 **CPU** 轮子，之后一切训练都跑在 CPU 上而且没有任何报错。先从
    PyTorch 官方 index 装好 CUDA 版，约束已满足，pip 不会再替换。
    """
    if forced_tag:
        index = f"https://download.pytorch.org/whl/{forced_tag}"
        say(f"installing torch from {index} (--torch={forced_tag})")
    else:
        index = ""
        helper = repo / "tools" / "select_torch_index.py"
        if helper.exists():
            try:
                out = subprocess.run(
                    [str(py), str(helper)], capture_output=True, text=True, timeout=60
                )
                index = out.stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                index = ""
        if not index:
            say("no NVIDIA GPU detected; using the default PyTorch build from PyPI")
            return
        say(f"NVIDIA GPU detected; installing torch from {index}")

    if not pip_install(py, ["torch", "torchvision", "--index-url", index], what="torch"):
        warn("CUDA torch install failed; falling back to the PyPI default")
        warn("you can fix this later in Studio → Settings → PyTorch → Reinstall")


def requirements_marker(venv_dir: Path) -> Path:
    return venv_dir / ".studio-requirements.sha256"


def requirements_state(py: Path, repo: Path, marker: Path, *, update: bool = False) -> str:
    """复用 `tools/check_requirements_changed.py`（内容 hash 而非 mtime）。"""
    helper = repo / "tools" / "check_requirements_changed.py"
    if not helper.exists():
        return "stale" if not marker.exists() else "current"
    argv = [str(py), str(helper), "--marker", str(marker)]
    if update:
        argv.append("--update-marker")
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        return out.stdout.strip() or "stale"
    except (OSError, subprocess.TimeoutExpired):
        return "stale"


def ensure_deps(py: Path, repo: Path, venv_dir: Path, *, fresh: bool,
                forced_tag: Optional[str]) -> None:
    marker = requirements_marker(venv_dir)
    req = repo / "requirements.txt"

    if fresh:
        pip_install(py, ["--upgrade", "pip"], what="pip")
        install_torch(py, repo, forced_tag)
        if not req.exists():
            warn("requirements.txt not found, skipping dependency install")
            return
        say("installing Python dependencies (this takes a while on first run)...")
        if not pip_install(py, ["-r", str(req)], what="requirements"):
            die(
                "failed to install dependencies",
                "check your internet connection and run the launcher again",
                "or run it with --reinstall to rebuild the environment from scratch",
            )
        requirements_state(py, repo, marker, update=True)
        return

    if forced_tag:
        # 已有 venv 但用户显式换 CUDA 版本 —— 照办，不然 --torch 在第二次运行
        # 起就静默无效了。
        install_torch(py, repo, forced_tag)

    if requirements_state(py, repo, marker) == "stale":
        say("requirements.txt changed since the last sync; installing new dependencies...")
        if pip_install(py, ["-r", str(req)], what="requirements"):
            requirements_state(py, repo, marker, update=True)
            say("dependency sync complete")
        else:
            warn("dependency sync failed; the existing environment still works "
                 "but may be missing new packages")
            warn("run the launcher with --reinstall if you hit import errors")


def reinstall_venv(repo: Path) -> None:
    venv_dir = find_existing_venv(repo)
    if not venv_dir:
        return
    say(f"--reinstall: {venv_dir} will be DELETED and rebuilt.")
    say("  - studio_data/ (your projects and LoRA weights) is NOT touched")
    say("  - pip packages you installed outside requirements.txt will be lost")
    if sys.stdin and sys.stdin.isatty():
        answer = input("Continue? [y/N] ").strip().lower()
        if not answer.startswith("y"):
            say("--reinstall aborted")
            raise SystemExit(0)
    say(f"removing {venv_dir}...")
    shutil.rmtree(venv_dir, ignore_errors=True)
    if venv_python(venv_dir).exists():
        die(f"could not remove {venv_dir}", "close any running Studio window and retry")


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------


def run_studio(py: Path, repo: Path, mode: Optional[str], passthrough: Sequence[str]) -> int:
    """起 `python -m studio run` 并守 restart 协议。

    与 studio.bat 的外层 loop 同构：`tmp/restart` 还在就再起一轮（server 端
    `/api/system/restart` 写的标志）；退出码 42 表示启动器文件自己被更新过，
    exe 形态下没法像 POSIX `exec` 那样原地换壳，所以让用户重开一次 —— 这条
    路径只在自更新后出现一次，比起悄悄跑着旧逻辑更安全。
    """
    restart_flag = repo / "tmp" / "restart"
    env = dict(os.environ)
    # 只有用户显式传了 --mode 才钉死模式。不传时留给 studio 自己解析（用户在
    # UI 里选过的值 → 探测兜底）—— 启动器无条件注入 "local" 的话，UI 里的模式
    # 开关会永远显示成"被环境变量锁定"，等于用一个便利换掉了另一个功能。
    if mode:
        env["ALS_RUNTIME_MODE"] = mode
    # 非 UTF-8 系统区域（日文 cp932 等）下 studio 的中文输出会崩在 print 上。
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    argv = [str(py), "-m", "studio", "run", *passthrough]
    while True:
        try:
            rc = subprocess.call(argv, cwd=str(repo), env=env)
        except KeyboardInterrupt:
            say("stopped (Ctrl+C)")
            return 130

        if not restart_flag.exists():
            return rc

        try:
            restart_flag.unlink()
        except OSError:
            pass

        if rc == INSTALLER_RELOAD_EXIT_CODE:
            say("the launcher itself was updated — please start it again")
            return 0
        say("restart requested, starting again...")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="AnimaLoraStudio",
        description=f"{APP_NAME} launcher — sets up the environment and starts the app.",
    )
    p.add_argument("--repo", metavar="PATH",
                   help="path to the AnimaLoraStudio folder (default: next to this launcher)")
    p.add_argument("--mode", choices=("local", "colab"), default=None,
                   help="pin the runtime mode instead of letting Studio ask/remember")
    p.add_argument("--reinstall", action="store_true",
                   help="delete venv/ and rebuild it (studio_data/ is kept)")
    p.add_argument("--torch", metavar="TAG", dest="torch_tag",
                   help="force a PyTorch CUDA build (cu128/cu126/cu124/cu118/cpu)")
    p.add_argument("--check", action="store_true",
                   help="report what the launcher found (folder, Python, venv, GPU) and exit")
    return p


def run_check(repo: Path) -> int:
    """「为什么起不来」的自查报告。

    远程帮人排查时，"把 exe 拖进终端加 --check 再把输出发我"比来回问十句有效
    得多 —— 装没装 Python、venv 在不在、认不认得出显卡，一屏说清。
    """
    print()
    say(f"folder      : {repo}")
    say(f"launcher    : {'frozen exe' if getattr(sys, 'frozen', False) else 'python script'}")

    venv_dir = find_existing_venv(repo)
    if venv_dir:
        py = venv_python(venv_dir)
        try:
            out = subprocess.run([str(py), "-V"], capture_output=True, text=True, timeout=30)
            say(f"venv        : {venv_dir} ({out.stdout.strip() or out.stderr.strip()})")
        except (OSError, subprocess.TimeoutExpired):
            say(f"venv        : {venv_dir} (broken — cannot run {py.name})")
        state = requirements_state(py, repo, requirements_marker(venv_dir))
        say(f"dependencies: {state}")
    else:
        say("venv        : not created yet (first run will build it)")
        argv = bootstrap_python()
        try:
            out = subprocess.run([*argv, "-V"], capture_output=True, text=True, timeout=30)
            say(f"system python: {' '.join(argv)} ({out.stdout.strip() or out.stderr.strip()})")
        except (OSError, subprocess.TimeoutExpired):
            say(f"system python: {' '.join(argv)} (cannot run)")

    # 「东西到底装哪去了」是这个报告最常被用来回答的问题，缓存位置要在里面。
    module = load_local_cache(repo)
    if module is not None:
        current = module.describe(repo)
        outside = {
            var: value for var, value in current.items()
            if not str(value).startswith(str(repo))
        }
        say(f"caches      : {len(current) - len(outside)}/{len(current)} "
            f"inside {repo / module.CACHE_DIR_NAME}")
        for var, value in outside.items():
            say(f"              {var}={value or '(library default, outside this folder)'}")

    smi = shutil.which("nvidia-smi")
    if not smi:
        say("gpu         : nvidia-smi not found (CPU-only, or drivers not installed)")
    else:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30,
            )
            say(f"gpu         : {out.stdout.strip() or 'nvidia-smi returned nothing'}")
        except (OSError, subprocess.TimeoutExpired):
            say("gpu         : nvidia-smi found but did not respond")

    print()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)

    say(f"{APP_NAME} launcher")
    repo = find_repo(args.repo)
    say(f"folder: {repo}")
    # 在任何 pip 调用之前 —— 见 apply_local_caches 的说明。
    cached = apply_local_caches(repo)
    if cached:
        say(f"caches → {repo / '.cache'} (set ALS_SYSTEM_CACHES=1 to use system locations)")

    if args.check:
        return run_check(repo)

    if args.reinstall:
        reinstall_venv(repo)

    venv_dir = find_existing_venv(repo)
    fresh = venv_dir is None
    if venv_dir is None:
        venv_dir = create_venv(repo)
    py = venv_python(venv_dir)
    if not py.exists():
        die(f"the virtual environment at {venv_dir} looks broken (no {py.name})",
            "run the launcher with --reinstall to rebuild it")
    if not fresh:
        check_version([str(py)], label=str(venv_dir))

    ensure_deps(py, repo, venv_dir, fresh=fresh, forced_tag=args.torch_tag)

    if args.mode == "colab":
        say("starting Studio in colab mode (bound to 0.0.0.0, no browser)")
    else:
        say("starting Studio — your browser will open automatically")
    rc = run_studio(py, repo, args.mode, passthrough)
    if rc not in (0, 130):
        print(f"\n[studio] Studio exited with code {rc}.", file=sys.stderr, flush=True)
        pause()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
