"""模型路径常量 + 本地路径解析（PR-3.8 从 model_downloader 1068 行拆出 4-way 第 1 个）。

只做"模型在本地哪儿"的回答：常量目录、各模型类型的 target Path、用户选定的
variant 读取（selected_anima / selected_upscaler）。不做下载、不读 endpoint /
mirror（那些在 sources.py）。

注意：`download_taeflux` 等 download_* 函数都搬到 downloader.py 了；这里只留
`taeflux_dir` / `taeflux_available` 这种"是否就绪"的查询。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from ... import secrets
from ...paths import REPO_ROOT

# ---------------------------------------------------------------------------
# 模型清单常量（新版本发布时改这里）
# ---------------------------------------------------------------------------

ANIMA_REPO = "circlestone-labs/Anima"
# 顺序：最新在前。`find_anima_main` 的 fallback 查找按本 dict 序遍历，
# `build_catalog` 给 UI 的 variants 列表也直接复用本顺序——所以新版本
# 加在最前，老版本往下排。
ANIMA_VARIANTS: dict[str, str] = {
    "1.0":           "split_files/diffusion_models/anima-base-v1.0.safetensors",
    "preview3-base": "split_files/diffusion_models/anima-preview3-base.safetensors",
    "preview2":      "split_files/diffusion_models/anima-preview2.safetensors",
    "preview":       "split_files/diffusion_models/anima-preview.safetensors",
}
LATEST_ANIMA = "1.0"
ANIMA_VAE_PATH = "split_files/vae/qwen_image_vae.safetensors"

QWEN_REPO = "Qwen/Qwen3-0.6B-Base"
# 注：Qwen3 把 special tokens 直接塞进 tokenizer.json，所以 repo 里没有
# `special_tokens_map.json`（旧 Qwen 版本有，照搬就 404）。
QWEN_FILES = [
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "config.json",
]

T5_REPO = "google/t5-v1_1-xxl"
T5_FILES = [
    "spiece.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

# TAEFlux：1.6MB 的 tiny autoencoder for Flux/Anima，daemon 预览中间步用。
# 用 diffusers.AutoencoderTiny.from_pretrained 加载 → 需要同时拿 config.json
# + safetensors 两个文件。
TAEFLUX_REPO = "madebyollin/taef1"
TAEFLUX_FILES = [
    "diffusion_pytorch_model.safetensors",
    "config.json",
]

# CLTagger 子目录布局：仓库内 cl_tagger_1_02/model.onnx 等。新版本（1.03 等）
# 出现时往这里加一行；UI 自动作为 radio 选项暴露。
# label → (model_path, tag_mapping_path)
CLTAGGER_VERSIONS: dict[str, tuple[str, str]] = {
    "cl_tagger_1_02": (
        "cl_tagger_1_02/model.onnx",
        "cl_tagger_1_02/tag_mapping.json",
    ),
}

# WD14 模型常驻文件名（HF SmilingWolf/* 仓库顶层都是这两个）。
WD14_FILES = ("model.onnx", "selected_tags.csv")

# 预处理放大器预设清单。
#
# label → 元数据 dict：
#   filename      落地文件名（也是 `selected_upscaler` 持久化的 key 之一）
#   hf            (repo_id, repo_subpath) HuggingFace 源；None 表示该模型在 HF 上无稳定镜像
#   ms            (repo_id, repo_subpath) ModelScope 源；None 表示无镜像
#   size_mb       近似下载体积，前端展示用
#   description   一句话用途描述（前端展示）
#
# 路由：download_upscaler 先按 _get_download_source() 取偏好源，对应 None 时透明
# fallback 到另一个源。两个源都 None 视为非法预设。
#
# 选源参考：libfishopen/upscaler 在魔搭上聚合了一批 A1111 时代主流权重，文件名 +
# 字节大小与 HF 原仓库一致；HF 一侧则使用各上游作者的官方仓库（更权威）。
UPSCALER_VARIANTS: dict[str, dict[str, Any]] = {
    "4x-AnimeSharp": {
        "filename": "4x-AnimeSharp.pth",
        "hf": ("Kim2091/AnimeSharp", "4x-AnimeSharp.pth"),
        "ms": ("libfishopen/upscaler", "4x-AnimeSharp.pth"),
        "size_mb": 64,
        "description": "二次元线稿/扁色友好（Kim2091, ESRGAN-RRDB）",
    },
    "R-ESRGAN_4x+Anime6B": {
        "filename": "R-ESRGAN_4x+Anime6B.pth",
        "hf": None,  # 上游 RealESRGAN 仓库未直接发 .pth，先只走 MS
        "ms": ("libfishopen/upscaler", "R-ESRGAN_4x+Anime6B.pth"),
        "size_mb": 18,
        "description": "动漫专用小模型（Real-ESRGAN，A1111 默认）",
    },
    "4x_foolhardy_Remacri": {
        "filename": "4x_foolhardy_Remacri.pth",
        "hf": None,
        "ms": ("libfishopen/upscaler", "4x_foolhardy_Remacri.pth"),
        "size_mb": 64,
        "description": "写实风格（口碑模型）",
    },
    "ESRGAN_4x": {
        "filename": "ESRGAN_4x.pth",
        "hf": None,
        "ms": ("libfishopen/upscaler", "ESRGAN_4x.pth"),
        "size_mb": 64,
        "description": "通用 ESRGAN baseline",
    },
}
DEFAULT_UPSCALER = "4x-AnimeSharp"
# 允许的自定义/上传放大器扩展名（白名单防写错路径 / 误传可执行）。
UPSCALER_EXTS = (".pth", ".safetensors")
# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def safe_dir_name(model_id: str) -> str:
    """把 HF/MS repo id 转成本地目录名（替换路径分隔符为 _）。

    通用 path-sanitization 工具，曾在 tagging.onnx_base 内（PR-3.8 移到这里
    打破循环：models/paths.py ← tagging/onnx_base.py ← models/downloader.py）。
    """
    return model_id.replace("/", "_").replace("\\", "_")


def models_root() -> Path:
    """模型根目录（所有训练 / WD14 模型共用）。

    解析优先级：
      1. `secrets.models.root`（用户在设置页配置的绝对路径）
      2. 环境变量 `ALS_MODELS_ROOT`（云端 notebook 注入；让模型一次性下载到
         持久盘 / Google Drive 后跨会话复用，无需用户手动进 Settings 配置）
      3. `{REPO_ROOT}/models/`（默认）

    云端机系统盘是临时盘 —— 不设 1/2 时模型每次新连接都会重新下载（Anima
    主模型 + Qwen3 + T5 等好几个 GB）。把根指到 Drive 即可「下载一次，永久复用」。

    注意目录命名：与 schema.py 里的 `transformer_path` 默认值（同 `models/`）
    + WD14 的 `models/wd14/` 对齐；HF repo 内部命名 `diffusion_models/`，本地
    扁平化时也用同名子目录。
    """
    try:
        cfg_root = secrets.load().models.root
    except Exception:
        cfg_root = None
    if cfg_root and str(cfg_root).strip():
        return Path(str(cfg_root).strip()).expanduser()
    env_root = os.environ.get("ALS_MODELS_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser()
    return REPO_ROOT / "models"


def anima_main_target(root: Path, variant: str) -> Path:
    if variant == "latest":
        variant = LATEST_ANIMA
    if variant not in ANIMA_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    return root / "diffusion_models" / Path(ANIMA_VARIANTS[variant]).name


# 基础模型（Anima 主模型）扩展名白名单 —— 注册本地 checkpoint 为 base 时校验。
ANIMA_EXTS: tuple[str, ...] = (".safetensors",)


def diffusion_models_dir(root: Optional[Path] = None) -> Path:
    """本地主模型（.safetensors）存放目录 —— 预设与自定义 base 都落在这里。"""
    return (root or models_root()) / "diffusion_models"


def _anima_preset_filenames() -> set[str]:
    """预设 variant 落地后的纯文件名集合（把自定义文件从预设中区分开）。"""
    return {Path(sp).name for sp in ANIMA_VARIANTS.values()}


def is_custom_anima(name: Optional[str], root: Optional[Path] = None) -> bool:
    """`name` 是否指向一个已落盘的自定义 base checkpoint（非预设 variant）。

    判据同 `selected_upscaler` 的 custom 分支：带白名单扩展名、纯文件名（拒绝
    路径分隔符 / 穿越）、不属于任何预设 variant、且在 diffusion_models/ 实际存在。
    """
    if not name or name in ANIMA_VARIANTS:
        return False
    safe = Path(name).name
    if safe != name:  # 带目录前缀 / .. → 拒绝
        return False
    if not safe.lower().endswith(ANIMA_EXTS):
        return False
    if safe in _anima_preset_filenames():
        return False
    return (diffusion_models_dir(root) / safe).exists()


def resolve_anima_main_path(variant: str, root: Optional[Path] = None) -> Path:
    """把 variant（预设 label 或自定义本地文件名）解析成主模型绝对路径。

    - 预设 label（含 "latest"）→ anima_main_target（行为不变）
    - 已落盘的自定义文件名 → diffusion_models/{filename}
    未知值回退到 LATEST_ANIMA 预设路径（与 selected_anima_variant 兜底一致）。
    """
    r = root or models_root()
    if variant == "latest":
        variant = LATEST_ANIMA
    if variant in ANIMA_VARIANTS:
        return anima_main_target(r, variant)
    if is_custom_anima(variant, r):
        return diffusion_models_dir(r) / Path(variant).name
    return anima_main_target(r, LATEST_ANIMA)


def anima_vae_target(root: Path) -> Path:
    return root / "vae" / Path(ANIMA_VAE_PATH).name


def qwen_dir(root: Path) -> Path:
    return root / "text_encoders"


def t5_tokenizer_dir(root: Path) -> Path:
    return root / "t5_tokenizer"


def taeflux_dir(root: Optional[Path] = None) -> Path:
    """TAEFlux 本地目录。daemon 用 AutoencoderTiny.from_pretrained 加载。"""
    r = root or models_root()
    return r / "taeflux"


def taeflux_available(root: Optional[Path] = None) -> bool:
    """两个文件都到位才算就绪。"""
    d = taeflux_dir(root)
    return all((d / f).exists() for f in TAEFLUX_FILES)



def wd14_target_dir(root: Path, model_id: str) -> Path:
    """WD14 单个 model_id 的本地目录。同 wd14_tagger 的 _resolve_model_dir 路径布局。"""
    return root / "wd14" / safe_dir_name(model_id)


def cltagger_target_root(root: Path, model_id: str) -> Path:
    """CLTagger repo 的本地根目录。子目录布局来自 CLTAGGER_VERSIONS。"""
    return root / "cltagger" / safe_dir_name(model_id)


def upscaler_dir(root: Optional[Path] = None) -> Path:
    """放大器权重根目录 `{models_root}/upscalers/`。"""
    r = root or models_root()
    return r / "upscalers"


def upscaler_target(label: str, root: Optional[Path] = None) -> Path:
    """单个放大器权重的目标路径。

    label 可以是：
      - 预设 key（在 UPSCALER_VARIANTS 中）→ 用预设里的 filename
      - 直接的文件名（带 .pth/.safetensors 扩展名）→ 视为自定义/已上传模型

    路径穿越保护：禁止 label 含 `/`、`\\` 或 `..`，避免落到 upscalers/ 之外。
    """
    if "/" in label or "\\" in label or ".." in label:
        raise ValueError(f"invalid upscaler label {label!r}")
    if label in UPSCALER_VARIANTS:
        fname = UPSCALER_VARIANTS[label]["filename"]
    else:
        if not label.lower().endswith(UPSCALER_EXTS):
            raise ValueError(f"unknown upscaler {label!r}")
        fname = label
    return upscaler_dir(root) / fname


def find_upscaler(label: str, root: Optional[Path] = None) -> Optional[Path]:
    """已下载返回本地路径，没下载返回 None。"""
    target = upscaler_target(label, root)
    return target if target.exists() else None


def find_anima_main(root: Optional[Path] = None) -> Optional[Path]:
    """按 ANIMA_VARIANTS 优先级（latest 在前）找第一个磁盘上存在的主模型。

    仅做兜底（裸 CLI / yaml 缺失时）；Studio 创建 version 时优先用
    `selected_anima_path()` 拿用户在 settings 里选定的 variant。
    """
    r = root or models_root()
    order = [LATEST_ANIMA] + [v for v in ANIMA_VARIANTS if v != LATEST_ANIMA]
    for v in order:
        target = anima_main_target(r, v)
        if target.exists():
            return target
    return None


def selected_anima_variant() -> str:
    """读 `secrets.models.selected_anima`，回退 LATEST_ANIMA。

    返回值可能是：
      - 预设 variant label（在 ANIMA_VARIANTS 中）
      - 已注册的自定义本地 checkpoint 文件名（在 diffusion_models/ 存在）
    两者都不匹配时回退 LATEST_ANIMA（与 selected_upscaler 的 custom 逻辑一致）。
    """
    try:
        v = secrets.load().models.selected_anima
    except Exception:
        v = None
    if v and v in ANIMA_VARIANTS:
        return v
    if v and is_custom_anima(v):
        return v
    return LATEST_ANIMA


def selected_upscaler() -> str:
    """读 `secrets.models.selected_upscaler`，回退 DEFAULT_UPSCALER。

    返回值可能是：
      - 预设 label（在 UPSCALER_VARIANTS 中）
      - 已存在的 custom filename（带扩展名）
    都未匹配时回退 DEFAULT_UPSCALER（预设 4x-AnimeSharp）。
    """
    try:
        v = secrets.load().models.selected_upscaler
    except Exception:
        v = None
    if not v:
        return DEFAULT_UPSCALER
    if v in UPSCALER_VARIANTS:
        return v
    # custom：扫盘看文件存不存在
    if v.lower().endswith(UPSCALER_EXTS) and (upscaler_dir() / v).exists():
        return v
    return DEFAULT_UPSCALER


def default_paths_for_new_version() -> dict[str, str]:
    """Studio 创建新 version 时用：返回 4 项路径的**绝对路径字符串**。

    根据当前 `secrets.models.root` 和 `secrets.models.selected_anima` 计算。
    用户在 settings 切了 selected_anima → 之后新建的 version 自动用新选择；
    已存在 version 的 yaml 不动（重现性）。
    """
    root = models_root()
    variant = selected_anima_variant()
    return {
        "transformer_path": str(resolve_anima_main_path(variant, root)),
        "vae_path": str(anima_vae_target(root)),
        "text_encoder_path": str(qwen_dir(root)),
        "t5_tokenizer_path": str(t5_tokenizer_dir(root)),
    }
