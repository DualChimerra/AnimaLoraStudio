#!/usr/bin/env python
"""把 `tools/launcher.py` 编译成单文件可执行程序（Windows 上即 AnimaLoraStudio.exe）。

    python tools/build_launcher.py                 → dist/AnimaLoraStudio(.exe)
    python tools/build_launcher.py --output-dir X   自定义输出目录
    python tools/build_launcher.py --console        保留控制台窗口（默认就保留）

产物只有几 MB：打进去的是引导逻辑（纯 stdlib），**不含** torch / 前端 /
studio 本身 —— 那些留在仓库里，由 exe 在用户机器上按显卡装对的版本。见
`tools/launcher.py` 的模块 docstring。

PyInstaller 只在构建机上需要，故不进 requirements.txt；本脚本会在缺失时给出
安装命令而不是直接崩。交叉编译不可行（PyInstaller 产出的是宿主平台的可执行
文件），所以 Windows 的 exe 必须在 Windows 上构建 —— CI 走
`.github/workflows/build-launcher.yml`。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRY = REPO_ROOT / "tools" / "launcher.py"
NAME = "AnimaLoraStudio"


def ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401  — 只探测可用性
    except ImportError:
        print(
            "PyInstaller is not installed.\n"
            f"  {sys.executable} -m pip install pyinstaller\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


def build(output_dir: Path, *, windowed: bool, clean: bool) -> Path:
    work = output_dir / "_build"
    argv = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", NAME,
        "--distpath", str(output_dir),
        "--workpath", str(work),
        "--specpath", str(work),
        # 引导程序纯 stdlib：显式排掉这些包，免得构建机上恰好装了 torch /
        # numpy 时 PyInstaller 的依赖分析把几百 MB 一起打进去。
        "--exclude-module", "torch",
        "--exclude-module", "numpy",
        "--exclude-module", "PIL",
        "--exclude-module", "fastapi",
        "--exclude-module", "pydantic",
        "--exclude-module", "tkinter",
    ]
    if clean:
        argv.append("--clean")
    # 默认保留控制台：首次运行要装几 GB 依赖，装到哪一步、报什么错全靠这个窗口。
    # 藏掉窗口只会把「装了十分钟没反应」变成用户唯一能描述的现象。
    argv.append("--windowed" if windowed else "--console")
    icon = REPO_ROOT / "docs" / "images" / "launcher.ico"
    if icon.exists():
        argv += ["--icon", str(icon)]
    argv.append(str(ENTRY))

    print("[build]", " ".join(argv), flush=True)
    rc = subprocess.call(argv, cwd=str(REPO_ROOT))
    if rc != 0:
        raise SystemExit(rc)

    produced = output_dir / (f"{NAME}.exe" if sys.platform == "win32" else NAME)
    if not produced.exists():
        print(f"[build] expected {produced} but it was not created", file=sys.stderr)
        raise SystemExit(1)
    return produced


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", default=str(REPO_ROOT / "dist"),
                   help="where to put the executable (default: dist/)")
    p.add_argument("--windowed", action="store_true",
                   help="hide the console window (not recommended — see module docstring)")
    p.add_argument("--no-clean", action="store_true",
                   help="reuse PyInstaller's build cache")
    args = p.parse_args(argv)

    if not ENTRY.exists():
        print(f"entry point missing: {ENTRY}", file=sys.stderr)
        return 1
    ensure_pyinstaller()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    produced = build(output_dir, windowed=args.windowed, clean=not args.no_clean)

    size_mb = produced.stat().st_size / (1024 * 1024)
    print(f"\n[build] {produced}  ({size_mb:.1f} MB)")
    print("[build] put it in the repository root (next to requirements.txt) and double-click.")
    shutil.rmtree(output_dir / "_build", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
