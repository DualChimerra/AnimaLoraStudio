"""一键启动器（tools/launcher.py → AnimaLoraStudio.exe）。

只测**不碰网络、不装包**的部分：仓库定位、venv 路径推导、参数分流、以及
`--check` 自查报告。真正的 bootstrap（建 venv + pip install）由
`.github/workflows/build-launcher.yml` 的冒烟步骤在两个平台上跑。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import launcher  # noqa: E402


def make_repo(root: Path) -> Path:
    """造一个能被 `looks_like_repo` 认出来的最小目录。"""
    (root / "studio").mkdir(parents=True, exist_ok=True)
    (root / "studio" / "__init__.py").touch()
    (root / "requirements.txt").write_text("packaging\n", encoding="utf-8")
    return root


@pytest.fixture
def isolate_lookup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """把 launcher 的两个搜索起点（自身目录 / cwd）都挪到 tmp。

    不隔离的话 `find_repo` 会顺着 cwd 找到**真的**仓库，"找不到时报错"这条
    根本测不了 —— 测试自己就跑在仓库里。
    """
    empty = tmp_path / "elsewhere"
    empty.mkdir()

    def _set(where: Path) -> None:
        monkeypatch.setattr(launcher, "launcher_dir", lambda: where)
        monkeypatch.chdir(where)

    _set(empty)
    return _set


# ---------------------------------------------------------------------------
# 仓库定位
# ---------------------------------------------------------------------------


def test_looks_like_repo_needs_both_markers(tmp_path: Path) -> None:
    """只有 requirements.txt 不算 —— 否则随便一个 Python 项目都会被当成仓库。"""
    (tmp_path / "requirements.txt").touch()
    assert launcher.looks_like_repo(tmp_path) is False
    (tmp_path / "studio").mkdir()
    (tmp_path / "studio" / "__init__.py").touch()
    assert launcher.looks_like_repo(tmp_path) is True


def test_find_repo_next_to_launcher(tmp_path: Path, isolate_lookup) -> None:
    repo = make_repo(tmp_path / "AnimaLoraStudio")
    isolate_lookup(repo)
    assert launcher.find_repo(None) == repo


def test_find_repo_walks_up_from_subdirectory(tmp_path: Path, isolate_lookup) -> None:
    """exe 被放进 tools/ 或 dist/ 也要能用。"""
    repo = make_repo(tmp_path / "AnimaLoraStudio")
    nested = repo / "tools" / "bin"
    nested.mkdir(parents=True)
    isolate_lookup(nested)
    assert launcher.find_repo(None) == repo


def test_find_repo_explicit_path(tmp_path: Path, isolate_lookup) -> None:  # noqa: ARG001
    repo = make_repo(tmp_path / "somewhere")
    assert launcher.find_repo(str(repo)) == repo


def test_find_repo_explicit_path_rejects_wrong_folder(
    tmp_path: Path, isolate_lookup, capsys: pytest.CaptureFixture[str]  # noqa: ARG001
) -> None:
    with pytest.raises(SystemExit) as exc:
        launcher.find_repo(str(tmp_path))
    assert exc.value.code == 1
    assert "is not an AnimaLoraStudio checkout" in capsys.readouterr().err


def test_find_repo_reports_actionable_error_when_missing(
    isolate_lookup, capsys: pytest.CaptureFixture[str]  # noqa: ARG001
) -> None:
    """找不到时要讲清楚怎么办 —— 这是双击 exe 最常见的失败，报错就是全部 UI。"""
    with pytest.raises(SystemExit):
        launcher.find_repo(None)
    err = capsys.readouterr().err
    assert "could not find the AnimaLoraStudio files" in err
    assert "--repo" in err


# ---------------------------------------------------------------------------
# venv 路径
# ---------------------------------------------------------------------------


def test_venv_python_path_is_platform_correct(tmp_path: Path) -> None:
    py = launcher.venv_python(tmp_path / "venv")
    if launcher.os.name == "nt":
        assert py == tmp_path / "venv" / "Scripts" / "python.exe"
    else:
        assert py == tmp_path / "venv" / "bin" / "python"


def test_find_existing_venv_prefers_venv_over_dotvenv(tmp_path: Path) -> None:
    """与 studio.bat / studio.sh 的顺序一致，免得三个入口挑到不同环境。"""
    for name in (".venv", "venv"):
        py = launcher.venv_python(tmp_path / name)
        py.parent.mkdir(parents=True)
        py.touch()
    assert launcher.find_existing_venv(tmp_path) == tmp_path / "venv"


def test_find_existing_venv_none_when_absent(tmp_path: Path) -> None:
    assert launcher.find_existing_venv(tmp_path) is None


def test_find_existing_venv_ignores_empty_directory(tmp_path: Path) -> None:
    """只有目录、没有解释器 = 半坏的环境，不该被当成可用 venv 直接拿去跑。"""
    (tmp_path / "venv").mkdir()
    assert launcher.find_existing_venv(tmp_path) is None


# ---------------------------------------------------------------------------
# 参数分流
# ---------------------------------------------------------------------------


def test_unknown_args_are_passed_through() -> None:
    """`--port` 之类不是启动器的选项，要原样转给 `python -m studio run`。"""
    args, passthrough = launcher.build_parser().parse_known_args(
        ["--mode", "colab", "--port", "8800", "--no-browser"]
    )
    assert args.mode == "colab"
    assert passthrough == ["--port", "8800", "--no-browser"]


def test_mode_defaults_to_none() -> None:
    """不传 --mode 时启动器不能钉死模式 —— 否则 UI 里的模式开关会永远显示成
    被环境变量锁定。"""
    args, _ = launcher.build_parser().parse_known_args([])
    assert args.mode is None


# ---------------------------------------------------------------------------
# 输出必须是纯 ASCII
# ---------------------------------------------------------------------------


def test_printed_strings_are_pure_ascii() -> None:
    """**回归**：启动器打印出去的字符串里一个非 ASCII 字符都不能有。

    Windows 控制台按系统 ANSI 代码页解码（俄语 cp866 / 中文 cp936 / 日语 cp932），
    frozen exe 往里写一个 `→` 就是 UnicodeEncodeError 当场崩。真事：`caches → ...`
    这一行让 Windows CI 的冒烟步骤挂掉，而同一个字符也在 `die()` 的提示行里 ——
    也就是说**报错路径自己会崩**，用户看到的不是「Python 没装」而是一段
    PyInstaller traceback。studio.bat 顶部早有同一条纪律，这里补上机器校验。

    只查会被打印的字面量（say/warn/die/print/input 的参数 + argparse 的
    help/description）；注释和 docstring 不受限，那里中文说明更清楚。
    """
    import ast

    source = (Path(__file__).resolve().parent.parent / "tools" / "launcher.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))

    printed: list[tuple[int, str]] = []

    def collect(node: ast.AST) -> None:
        """把一个表达式里的字符串字面量都收进来（含 f-string 的固定片段）。"""
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                printed.append((sub.lineno, sub.value))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name in {"say", "warn", "die", "print", "input"}:
            for arg in node.args:
                collect(arg)
        # argparse 的 help= / description= 同样会打到终端
        for kw in node.keywords:
            if kw.arg in {"help", "description"}:
                collect(kw.value)

    offenders = [
        (line, text) for line, text in printed if not text.isascii()
    ]
    assert not offenders, (
        "non-ASCII in launcher output (crashes frozen exe on non-UTF-8 consoles):\n"
        + "\n".join(f"  line {line}: {text!r}" for line, text in offenders)
    )


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------


def test_check_reports_missing_venv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = make_repo(tmp_path / "repo")
    assert launcher.run_check(repo) == 0
    out = capsys.readouterr().out
    assert str(repo) in out
    assert "not created yet" in out
    assert "gpu" in out
