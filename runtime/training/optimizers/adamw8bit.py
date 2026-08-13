"""AdamW8bit optimizer build wrapper（ADR 0003 PR-C）。

bitsandbytes 的 8-bit AdamW：把两个动量状态（exp_avg / exp_avg_sq）分块量化到
int8，state 显存 ≈ fp32 AdamW 的 25%。更新数学与 AdamW 一致，超参（lr /
betas / weight_decay）**照搬 AdamW 不用换算** —— 与 Lion / Automagic 那种需要
重新标定 lr 的替代品不同。实现在 bitsandbytes，本文件只是 registry 的
build/validate 壳，与 adamw.py 同形。

`min_8bit_size`（utils 侧默认 4096）以下的小张量保持 fp32：小张量量化收益
小、精度损失相对大。LoRA 的 A/B 矩阵远超这个阈值，实际全部走 8-bit。

bitsandbytes 是**可选依赖**（requirements.txt 里注释掉：Windows 上并非总能
装上），所以 validate 在启动期把「没装」翻译成可操作错误，而不是等 build
时从 utils 里抛一句英文 ImportError。
"""

from __future__ import annotations


def validate(args) -> None:
    """启动期检查可选依赖 bitsandbytes 是否可用。"""
    from utils.optimizer_utils import BITSANDBYTES_AVAILABLE

    if not BITSANDBYTES_AVAILABLE:
        raise SystemExit(
            "optimizer_type=adamw8bit 需要 bitsandbytes，当前环境没有安装。\n"
            "  安装：pip install bitsandbytes\n"
            "  装不上时的替代：lion（state 显存 ≈ AdamW fp32 的 50%，但 lr 要"
            "改成 AdamW 的 1/3）或 came（state 更省，lr 同 AdamW 量级）。"
        )


def build(args, params, lr: float, weight_decay: float):
    """实例化 8-bit AdamW。

    不读 args.* 之外的额外参数；保持签名跟其他 builder 一致以便 registry
    统一派发（同 adamw.py）。
    """
    from utils.optimizer_utils import create_optimizer

    return create_optimizer(
        optimizer_type="adamw8bit",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
    )
