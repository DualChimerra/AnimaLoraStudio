"""CAME optimizer build wrapper（ADR 0003 PR-C）。

CAME = Confidence-guided Adaptive Memory Efficient（Luo et al., ACL 2023,
arxiv 2307.02047）。外部 lr + scheduler 系（同 AdamW / Lion）：不自适应 lr、非
schedule-free，所以不需要 validate()（无 lr=1.0 / lr_scheduler=none 约束）。

读 6 个 came_* 参数组装成 CAME 的 betas 三元组 + eps 二元组 + clip_threshold。
"""

from __future__ import annotations


def build(args, params, lr: float, weight_decay: float):
    """实例化 CAME，读 came_beta1/2/3、came_eps1/2、came_clip_threshold。"""
    from utils.optimizer_utils import create_optimizer

    return create_optimizer(
        optimizer_type="came",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
        betas=(
            float(getattr(args, "came_beta1", 0.9)),
            float(getattr(args, "came_beta2", 0.999)),
            float(getattr(args, "came_beta3", 0.9999)),
        ),
        eps=(
            float(getattr(args, "came_eps1", 1e-30)),
            float(getattr(args, "came_eps2", 1e-16)),
        ),
        clip_threshold=float(getattr(args, "came_clip_threshold", 1.0)),
    )
