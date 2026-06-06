"""本地上传：单图 / zip 压缩包（自动解压拍平）/ 同名 .txt caption，落盘到 download/。

用户通过浏览器 file picker / 拖拽上传文件 → server 端解析 → 写入项目的
`download/` 目录，与 booru 下载共享同一份「全量备份」。

约束：
- 接受 IMAGE_EXTS 里所有格式的单图（png / jpg / jpeg / webp / bmp / gif）；
  zip 包内同样按这套白名单提取。
- zip 内的子目录结构会被拍平（取 basename）。
- zip 损坏时整包跳过并报告，不影响其它文件。

caption 配对（kohya_ss / sd-scripts 风格）：
- 与图片同 stem 的 `.txt` 视为该图的 caption / tags（`1.png` ↔ `1.txt`）；
- 一次上传里（zip 内、或同批多文件拖拽）出现的 `.txt` 会按 stem 配到对应图，
  随图一起落盘到 download/。后续 curation 会把同名 .txt 一并带进 train/，
  于是 tagging 阶段直接看到这些 caption（caption 覆盖率即时满足）；
- 没有对应图片的孤立 `.txt` 被跳过并报告；
- convert_to_png 把图重命名 / 加后缀时（见下），caption 跟随**落盘后的实际
  stem**，所以 `1.jpg` → `1.png` 的 caption 落 `1.png` 同 stem 的 `1.txt`，后缀化
  的 `1_1.png` 不会与 `1.png` 抢同一个 caption。

`convert_to_png` 模式（与 booru 下载共用的 `gelbooru.convert_to_png` 设置）：
- 所有图片经 PIL 解码后统一重编码为 .png，文件名 stem 不变后缀改 .png；
- 同 stem 冲突（含 `1.jpg` + `1.png` 同上传一次的场景）改加 `_1`/`_2` 后缀
  落盘，避免 caption `1.txt` 被两张不同图共用；
- `remove_alpha_channel=True` 时按白底压平 alpha，与 booru 下载一致；
- PIL 解码失败按 skipped 上报「图片损坏」。

`convert_to_png=False`（默认）保持历史行为：原扩展名拷贝、目标已存在则跳过。
"""
from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable, Optional

from PIL import Image

from ..booru.api import flatten_alpha, has_alpha
from .scan import IMAGE_EXTS

# PP10 起复用全链路白名单：上传 / 下载 / curation / 训练共用一份。
ALLOWED_IMAGE_EXTS = IMAGE_EXTS
ZIP_EXT = ".zip"
# kohya_ss / sd-scripts 风格 caption：与图同 stem 的 .txt。curation 的 _META_EXTS
# 也是 (.txt, .json)，所以落到 download/ 的 .txt 会随图带进 train/。
CAPTION_EXT = ".txt"


@dataclass
class UploadResult:
    """单次上传调用的汇总结果。

    ``added`` 含落盘的图片名，也含成功配对的 caption（`*.txt`）名。
    """

    added: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, list]:
        return {"added": self.added, "skipped": self.skipped}

    def merge(self, other: "UploadResult") -> None:
        self.added.extend(other.added)
        self.skipped.extend(other.skipped)


@dataclass
class _Entry:
    """解包后的叶子文件（图片或 caption），统一全量读入内存再处理。"""

    report: str  # 给用户看的名字（zip 内 entry 带 `pack.zip:` 前缀）
    base: str    # 落盘 basename
    data: bytes


def _safe_basename(name: str) -> str:
    """剥掉 zip 内嵌套子目录 / Windows 反斜杠，只留 basename。"""
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def _is_image_ext(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_IMAGE_EXTS


def _is_caption_ext(name: str) -> bool:
    return Path(name).suffix.lower() == CAPTION_EXT


def _unique_target(dest_dir: Path, name: str) -> Path:
    """目标已存在时加 `_1` / `_2` ... 后缀直到不冲突。

    用于 convert_to_png 模式：用户的本意是「这张图也进来」，文件名冲突时
    用后缀保住第二张，而不是丢弃。caption 文件按落盘后的实际 stem 配对，
    所以后缀化的 `1_1.png` 会拿到独立的 `1_1.txt`，不再与 `1.png` 共用。
    """
    target = dest_dir / name
    if not target.exists():
        return target
    stem = Path(name).stem
    suffix = Path(name).suffix
    i = 1
    while True:
        cand = dest_dir / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def _write_image_entry(
    src_name: str,
    src_stream: BinaryIO,
    dest_dir: Path,
    *,
    convert_to_png: bool,
    remove_alpha_channel: bool,
    report_name: str,
    result: UploadResult,
) -> Optional[Path]:
    """落盘单张图。返回落盘后的实际路径；跳过 / 损坏返回 ``None``。"""
    if not convert_to_png:
        target = dest_dir / src_name
        if target.exists():
            result.skipped.append(
                {"name": report_name, "reason": "已存在，跳过"}
            )
            return None
        with target.open("wb") as fh:
            shutil.copyfileobj(src_stream, fh)
        result.added.append(src_name)
        return target

    raw = src_stream.read()
    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception as exc:  # noqa: BLE001 — PIL 抛多种类型，整体当损坏
        result.skipped.append(
            {"name": report_name, "reason": f"图片损坏: {exc}"}
        )
        return None
    target = _unique_target(dest_dir, Path(src_name).stem + ".png")
    if remove_alpha_channel and has_alpha(img):
        img = flatten_alpha(img)
    out = (
        img.convert("RGBA")
        if has_alpha(img) and not remove_alpha_channel
        else img.convert("RGB")
    )
    out.save(target, "PNG", optimize=True)
    result.added.append(target.name)
    return target


def _expand_input(
    src_name: str, src_stream: BinaryIO
) -> tuple[list[_Entry], list[dict[str, str]]]:
    """把单个上传项展开成叶子 entry（图片 / caption）+ 顶层跳过项。

    - 单图 / 单 .txt → 一个 entry
    - zip → 解压所有图片 + .txt（拍平、内层 entry 同样进 entry 列表）
    - 其他 / 没扩展名 / 损坏 zip → 不产 entry，记 skipped
    """
    base = _safe_basename(src_name or "")
    if not base:
        return [], [{"name": src_name, "reason": "文件名为空"}]

    suffix = Path(base).suffix.lower()

    if suffix in ALLOWED_IMAGE_EXTS or suffix == CAPTION_EXT:
        return [_Entry(base, base, src_stream.read())], []

    if suffix == ZIP_EXT:
        entries: list[_Entry] = []
        skipped: list[dict[str, str]] = []
        try:
            with zipfile.ZipFile(src_stream) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    inner_base = _safe_basename(info.filename)
                    label = f"{base}:{info.filename}"
                    if not inner_base:
                        continue
                    inner_suffix = Path(inner_base).suffix.lower()
                    if inner_suffix in ALLOWED_IMAGE_EXTS or inner_suffix == CAPTION_EXT:
                        entries.append(_Entry(label, inner_base, zf.read(info)))
                    else:
                        skipped.append({"name": label, "reason": "格式不支持"})
        except zipfile.BadZipFile:
            skipped.append({"name": base, "reason": "zip 损坏"})
        return entries, skipped

    allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTS)) + ", .txt, .zip"
    return [], [{"name": base, "reason": f"格式不支持（仅 {allowed}）"}]


def _accept_entries(
    entries: Iterable[_Entry],
    dest_dir: Path,
    *,
    convert_to_png: bool,
    remove_alpha_channel: bool,
) -> UploadResult:
    """落盘一批已展开的 entry，并按 stem 把 caption 配到图片。

    配对规则：caption 按 **图片落盘后的实际 stem** 命名（`X.png` → `X.txt`）。
    同 stem 多张图时，caption 给第一张成功落盘的图（FIFO），后缀化的副本不抢；
    没有对应图片的孤立 caption 跳过并报告。
    """
    result = UploadResult()
    dest_dir.mkdir(parents=True, exist_ok=True)

    items = list(entries)
    images = [e for e in items if _is_image_ext(e.base)]
    captions = [e for e in items if _is_caption_ext(e.base)]

    # caption 按上传名 stem 入队，供同 stem 的图片按 FIFO 领取。
    cap_by_stem: dict[str, list[_Entry]] = {}
    for cap in captions:
        cap_by_stem.setdefault(Path(cap.base).stem, []).append(cap)

    for img in images:
        final = _write_image_entry(
            img.base, BytesIO(img.data), dest_dir,
            convert_to_png=convert_to_png,
            remove_alpha_channel=remove_alpha_channel,
            report_name=img.report,
            result=result,
        )
        if final is None:
            continue
        queue = cap_by_stem.get(Path(img.base).stem)
        if queue:
            cap = queue.pop(0)
            cap_path = dest_dir / (final.stem + CAPTION_EXT)
            cap_path.write_bytes(cap.data)
            result.added.append(cap_path.name)

    # 没配上图的 caption（孤立 .txt / stem 对不上）→ 跳过并报告。
    for queue in cap_by_stem.values():
        for cap in queue:
            result.skipped.append(
                {"name": cap.report, "reason": "无对应图片，已忽略 caption"}
            )

    return result


def accept_one(
    src_name: str,
    src_stream: BinaryIO,
    dest_dir: Path,
    *,
    convert_to_png: bool = False,
    remove_alpha_channel: bool = False,
) -> UploadResult:
    """处理单个上传文件。

    - IMAGE_EXTS 内任一格式 → 落盘（按 convert_to_png 决定是否重编码 PNG）
    - `.txt` → 当 caption；单独上传无对应图片时跳过
    - zip → 解压所有图片 + .txt（拍平、按 stem 配对 caption）
    - 其他 / 没扩展名 → 拒绝
    """
    entries, skipped = _expand_input(src_name, src_stream)
    result = _accept_entries(
        entries, dest_dir,
        convert_to_png=convert_to_png,
        remove_alpha_channel=remove_alpha_channel,
    )
    result.skipped = skipped + result.skipped
    return result


def accept_many(
    files: Iterable[tuple[str, BinaryIO]],
    dest_dir: Path,
    *,
    convert_to_png: bool = False,
    remove_alpha_channel: bool = False,
) -> UploadResult:
    """批量处理；先把每个输入展开成叶子 entry 汇到一起，再统一落盘 + 配对。

    汇总后再配对的好处：同批拖拽的 `1.png` + `1.txt`（不在同一个 zip 里）也能
    按 stem 配上 caption，而不只是 zip 内部。
    """
    all_entries: list[_Entry] = []
    pre_skipped: list[dict[str, str]] = []
    for name, stream in files:
        entries, skipped = _expand_input(name, stream)
        all_entries.extend(entries)
        pre_skipped.extend(skipped)

    result = _accept_entries(
        all_entries, dest_dir,
        convert_to_png=convert_to_png,
        remove_alpha_channel=remove_alpha_channel,
    )
    result.skipped = pre_skipped + result.skipped
    return result
