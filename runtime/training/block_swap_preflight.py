"""block swap 预检：在**任何权重加载之前**判断当前 blocks_to_swap 能不能跑，
不能就带着推荐值 fail-fast。

为什么需要它：``blocks_to_swap`` 默认 0，而 Krea 2 的 DiT 就算 fp8 也有
13GB —— 12GB 卡上「选中模型直接开跑」必然 OOM，而 OOM 发生在数据集扫描、
latent 缓存、文本编码全跑完之后，用户已经等了几分钟，报错还只是一句
CUDA out of memory，不告诉他该把哪个数字调到多少。

既有的两道护栏解决不了这件事：
- ``check_load_budget``：只在**加载那一刻**按文件大小预算，且训练侧默认关闭
  （误拒率高，见 secrets.TrainingSecretsConfig.ram_guard）；
- ``check_pinned_budget``：只管**内存**侧，且在 loader 内部、换出层要落 pinned
  的那一刻才触发。

本模块把两侧算术合到一起、提前到加载前，并且**给出推荐值**而不只是拒绝。
纯算术，不碰 CUDA、不读权重（只 stat 文件大小），失败一律静默放行。

族无关：只要族实现了 ``swapped_param_ratio`` / ``swappable_blocks``（目前只有
krea2，也只有它有 block_swap 能力位）就参与，否则整块跳过。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional


logger = logging.getLogger(__name__)

_GIB = 1024 ** 3

#: 推荐值要预留的显存余量。比 ``_VRAM_BASE_BYTES``（3GB，标定给「权重装得下吗」
#: 那道护栏）更宽，因为训练期显存不止权重：LoRA 参数 + 梯度 + 优化器状态、
#: gradient checkpointing 存下的层间激活、fp8 底模逐层 dequant 的临时权重、
#: 分配器碎片。推荐一个「刚好装下权重」的数字然后让用户在第 3 分钟 OOM，
#: 比不推荐更糟，所以这里刻意保守。
_RECOMMEND_HEADROOM_BYTES = 5 * _GIB


@dataclass(frozen=True)
class SwapVerdict:
    """某个 blocks_to_swap 候选值的预算结论。"""

    blocks: int
    vram_need: int
    pinned_need: int
    vram_ok: bool
    ram_ok: bool

    @property
    def ok(self) -> bool:
        return self.vram_ok and self.ram_ok


@dataclass(frozen=True)
class PreflightResult:
    """预检结论。``checked=False`` 表示信息不足、本次不下判断。"""

    checked: bool
    ok: bool
    current: Optional[SwapVerdict] = None
    recommended: Optional[int] = None
    message: str = ""


def evaluate(
    *,
    file_bytes: int,
    blocks_to_swap: int,
    total_blocks: int,
    ratio_fn: Callable[[int], float],
    free_vram_bytes: Optional[int],
    avail_ram_bytes: Optional[int],
    vram_base_bytes: int,
    pinned_limit_bytes: Optional[int],
) -> PreflightResult:
    """纯算术核心：给定预算输入，判断当前设置并搜推荐值。

    调用方负责把系统查询（空闲显存 / 可用内存 / 文件大小）取好传进来 ——
    这样本函数零副作用、可直接用数字测。

    ``free_vram_bytes is None``（非 CUDA / 查询失败）→ 不下判断。
    ``pinned_limit_bytes is None``（内存查询失败）→ 只查显存侧，内存侧一律放行。
    """
    if file_bytes <= 0 or total_blocks <= 0 or free_vram_bytes is None:
        return PreflightResult(checked=False, ok=True)

    def verdict(blocks: int) -> SwapVerdict:
        ratio = min(max(float(ratio_fn(blocks)), 0.0), 1.0)
        # 显存侧与 check_load_budget 同一条算术（need × (1-ratio) + 基底），
        # 两道护栏才不会出现「预检说行、加载时拒」。
        vram_need = int(file_bytes * (1.0 - ratio)) + vram_base_bytes
        # 内存侧是近似：loader 的 check_pinned_budget 读 header 数换出层的
        # **实际**字节，这里按参数比例 × 文件大小估。单一 dtype 的 checkpoint
        # 两者几乎相等（fp8_scaled 多出的 F32 scale 是零头）。权威判定仍归
        # loader，本预检只负责提前提醒和给推荐值。
        pinned_need = int(file_bytes * ratio)
        return SwapVerdict(
            blocks=blocks,
            vram_need=vram_need,
            pinned_need=pinned_need,
            vram_ok=free_vram_bytes >= vram_need,
            ram_ok=pinned_limit_bytes is None or pinned_need <= pinned_limit_bytes,
        )

    current = verdict(min(max(int(blocks_to_swap), 0), total_blocks))
    if current.ok:
        return PreflightResult(checked=True, ok=True, current=current)

    # 推荐：能满足「更宽的训练余量」的最小换出层数（换得越少越快）。
    # 没有这样的档位时退到「至少装得下权重」的最小值，并在文案里标注为紧。
    candidates = [verdict(n) for n in range(total_blocks + 1)]

    def has_training_headroom(v: SwapVerdict) -> bool:
        weights_only = v.vram_need - vram_base_bytes
        return free_vram_bytes >= weights_only + _RECOMMEND_HEADROOM_BYTES

    roomy = [v for v in candidates if v.ram_ok and has_training_headroom(v)]
    workable = [v for v in candidates if v.ok]
    if roomy:
        recommended, tight = roomy[0].blocks, False
    elif workable:
        # 有能装下权重的值，但没有一个留够训练余量。仍然推荐它（那是本机能
        # 给的最好结果），但必须在文案里说清是紧的——照着改完在第 3 分钟
        # OOM 而事先没被提醒，比不推荐更伤。
        recommended, tight = workable[0].blocks, True
    else:
        recommended, tight = None, False

    return PreflightResult(
        checked=True,
        ok=False,
        current=current,
        recommended=recommended,
        message=_message(
            current=current,
            recommended=recommended,
            tight=tight,
            candidates=candidates,
            total_blocks=total_blocks,
            free_vram_bytes=free_vram_bytes,
            avail_ram_bytes=avail_ram_bytes,
            pinned_limit_bytes=pinned_limit_bytes,
        ),
    )


def _gb(value: Optional[int]) -> str:
    return "未知" if value is None else f"{value / _GIB:.1f}GB"


def _message(
    *,
    current: SwapVerdict,
    recommended: Optional[int],
    tight: bool,
    candidates: list,
    total_blocks: int,
    free_vram_bytes: int,
    avail_ram_bytes: Optional[int],
    pinned_limit_bytes: Optional[int],
) -> str:
    head = (
        f"block swap 预检未通过：blocks_to_swap={current.blocks} 时底模约需 "
        f"{_gb(current.vram_need)} 显存、{_gb(current.pinned_need)} 锁定内存，"
        f"当前空闲显存 {_gb(free_vram_bytes)}"
    )
    if pinned_limit_bytes is not None:
        head += f"、可用内存 {_gb(avail_ram_bytes)}（可锁定上限 {_gb(pinned_limit_bytes)}）"
    head += "。"

    if recommended is not None:
        body = (
            f"{head}\n"
            f"  建议：把 blocks_to_swap 设为 {recommended}（共 {total_blocks} 层）。"
            f"换出的层常驻内存、算到才搬进显存，不影响训练结果，只慢一点。"
        )
        if tight:
            body += (
                f"\n  注意：这个值只是**刚好装下权重**，本机没有能同时留够"
                f"训练余量（LoRA + 优化器状态 + 激活 + dequant 临时权重）的档位。"
                f"训练中途仍可能 OOM —— 更稳的出路是换 fp8 底模（体积减半），"
                f"或降低训练分辨率。"
            )
        return (
            f"{body}\n"
            f"  确认估算不准可在「block swap 预检」开关处关掉本检查。"
        )

    # 一个候选都不成立：区分是显存还是内存卡住，给不同出路。
    vram_reachable = any(v.vram_ok for v in candidates)
    if not vram_reachable:
        why = (
            f"即使换出全部 {total_blocks} 层，不可换出的部分（嵌入层 / 输出层等）"
            f"仍需约 {_gb(candidates[total_blocks].vram_need)} 显存。"
        )
        fix = "换用 fp8 底模（体积减半），或关掉占着显存的其他程序（ComfyUI / 出图任务）。"
    else:
        why = (
            f"显存侧可以满足，但换出层要锁定的内存超过上限"
            f"（最少需锁定 {_gb(min(v.pinned_need for v in candidates if v.vram_ok))}）。"
            f"锁定内存不可换页，超限会拖垮整机。"
        )
        fix = "换用 fp8 底模（锁定内存同样减半），或关闭占内存的其他程序后重试。"
    return f"{head}\n  {why}\n  出路：{fix}"


def run(ctx) -> None:
    """在 models_phase 里调用：不通过就抛 RuntimeError（带推荐值）。

    以下情况静默跳过（宁可不判也不误拒）：开关关闭、族没有 block_swap 能力、
    族未实现比例/层数估算、权重文件读不到大小、显存查询失败。
    """
    args = ctx.args
    if not bool(getattr(args, "block_swap_preflight", True)):
        logger.info("block swap 预检已关闭（block_swap_preflight=false）")
        return
    family = ctx.family
    if family is None or "block_swap" not in family.spec.capabilities:
        return
    ratio_fn = getattr(family, "swapped_param_ratio", None)
    blocks_fn = getattr(family, "swappable_blocks", None)
    if ratio_fn is None or blocks_fn is None:
        return

    from training import sysmem

    try:
        total_blocks = int(blocks_fn())
        file_bytes = sysmem._file_bytes([getattr(args, "transformer_path", "")])
        avail_ram = sysmem.available_ram_bytes()
        result = evaluate(
            file_bytes=file_bytes,
            blocks_to_swap=int(getattr(args, "blocks_to_swap", 0) or 0),
            total_blocks=total_blocks,
            ratio_fn=ratio_fn,
            free_vram_bytes=sysmem.gpu_free_bytes_global(),
            avail_ram_bytes=avail_ram,
            vram_base_bytes=sysmem._VRAM_BASE_BYTES,
            pinned_limit_bytes=(
                None if avail_ram is None else sysmem.pinned_safe_limit(avail_ram)
            ),
        )
    except Exception:  # noqa: BLE001
        # 预检是辅助设施，自身出错绝不能挡住训练
        logger.warning("block swap 预检执行失败，跳过", exc_info=True)
        return

    if not result.checked:
        return
    if not result.ok:
        raise RuntimeError(result.message)
    if result.current is not None and result.current.blocks > 0:
        logger.info(
            "block swap 预检通过：换出 %d 层，底模约占显存 %s、锁定内存 %s",
            result.current.blocks,
            _gb(result.current.vram_need),
            _gb(result.current.pinned_need),
        )
