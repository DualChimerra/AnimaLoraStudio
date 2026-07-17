"""数据集与 collate：ARB 分桶 + ImageDataset + 正则集 merge + cached latent。

抽自原 runtime/anima_train.py L1144-1675 + L1939-1962（ADR 0003 PR-A）。

公开：
- BucketManager / ImageDataset / RepeatDataset / MergedDataset
- BucketBatchSampler / CachedLatentDataset
- collate_fn / collate_fn_cached — DataLoader collate
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


# ===========================================================================
# NaViT / Patch-n-Pack 原生定尺寸规划（navit_native_resolution，opt-in）
# ===========================================================================


@dataclass
class NativeFitImagePlan:
    """NaViT 原生定尺寸的规划结果（``navit_native_resolution``）。

    ``width``/``height`` 是最终喂给 VAE 的像素尺寸（``align`` 的整倍数）。像素路径
    复用 ``ImageDataset`` 现有的 resize-cover + center-crop（零 padding），因此有效区
    永远填满整张 latent（navit 缓存路径不携带 mask 的前提成立）。
    """
    source_width: int
    source_height: int
    width: int
    height: int
    token_w: int
    token_h: int
    token_count: int
    was_downscaled: bool


def plan_native_fit_image(
    width: int,
    height: int,
    *,
    max_tokens: int = 0,
    max_side_tokens: int = 0,
    align: int = 16,
    over_budget: str = "downscale",
) -> NativeFitImagePlan:
    """规划单张图的原生定尺寸（floor 对齐到 ``align``，可选超预算 downscale）。

    对齐单元 ``align = patch_spatial(2) × vae_downsample(8) = 16px``。

    - 普通情形（原生 token ≤ ``max_tokens`` 且各边 ≤ ``max_side_tokens``）：每边 floor 到
      ``align`` 整倍数（丢 ≤align-1 px），不缩放、宽高比几乎不变。
    - 超预算：``downscale``（默认）等比缩到同时满足两上限，再各轴 floor 到 align；
      ``fail`` 直接报错。``max_tokens`` / ``max_side_tokens`` 为 0 表示该维不设限。
    """
    W, H = int(width), int(height)
    if W <= 0 or H <= 0:
        raise ValueError(f"image dimensions must be positive, got {W}x{H}")
    align = max(1, int(align))
    max_tokens = max(0, int(max_tokens or 0))
    max_side = max(0, int(max_side_tokens or 0))

    gw, gh = W // align, H // align
    if gw <= 0 or gh <= 0:
        raise ValueError(
            f"image {W}x{H} smaller than one align unit ({align}px); cannot form a token"
        )

    over_side = bool(max_side and (gw > max_side or gh > max_side))
    over_budget_tokens = bool(max_tokens and gw * gh > max_tokens)

    if not over_side and not over_budget_tokens:
        return NativeFitImagePlan(
            source_width=W, source_height=H,
            width=gw * align, height=gh * align,
            token_w=gw, token_h=gh, token_count=gw * gh,
            was_downscaled=False,
        )

    strategy = (over_budget or "downscale").lower()
    if strategy == "fail":
        reasons = []
        if over_budget_tokens:
            reasons.append(f"{gw * gh} tokens > navit_token_budget={max_tokens}")
        if over_side:
            reasons.append(
                f"side {max(gw, gh)} tokens > RoPE 单边上限 {max_side}"
                f"（≈{max_side * align}px）"
            )
        raise ValueError(
            f"[navit-native] 图 {W}x{H} 原生尺寸超限（{'；'.join(reasons)}）。"
            "调大 navit_token_budget / 提高 max_img_h·max_img_w，或把 "
            "navit_native_over_budget 设为 downscale（默认，自动等比降采样）。"
        )
    if strategy != "downscale":
        raise ValueError(
            f"unknown navit_native_over_budget={over_budget!r}; expected downscale or fail"
        )

    s = 1.0
    if over_side:
        s = min(s, max_side / float(max(gw, gh)))
    if max_tokens and gw * gh > max_tokens:
        s = min(s, math.sqrt(max_tokens / float(gw * gh)))
    ngw = max(1, int(gw * s))
    ngh = max(1, int(gh * s))
    if max_side:
        ngw, ngh = min(ngw, max_side), min(ngh, max_side)
    if max_tokens and ngw * ngh > max_tokens:
        if ngw >= ngh:
            ngw = max(1, max_tokens // ngh)
        else:
            ngh = max(1, max_tokens // ngw)
    return NativeFitImagePlan(
        source_width=W, source_height=H,
        width=ngw * align, height=ngh * align,
        token_w=ngw, token_h=ngh, token_count=ngw * ngh,
        was_downscaled=True,
    )


class BucketManager:
    """ARB 分桶管理.

    SYNC WITH ``studio/web/src/lib/trainBuckets.ts``. The crop page on the web
    UI predicts trainer buckets to pre-align cluster crops so the trainer
    doesn't re-resize them — that prediction depends on a TS port of this
    class. Any change to the algorithm or to the default parameters
    (``base_reso``, ``min_reso``, ``max_reso``, ``step``, the 0.1 area
    tolerance, the 2.0 AR cap) MUST land in both files in the same commit,
    or the frontend's predicted bucket ≠ trainer's actual bucket and crops
    will silently degrade.

    See ``docs/design/preprocess-crop-design.md`` §7 for the UX policy and
    rationale.
    """
    def __init__(self, base_reso=1024, min_reso=512, max_reso=2048, step=64,
                 max_ar=2.0, area_tolerance=0.1):
        self.base_reso = base_reso
        self.buckets = self._generate(min_reso, max_reso, step, base_reso,
                                      max_ar, area_tolerance)

    def _generate(self, min_r, max_r, step, base, max_ar=2.0, area_tolerance=0.1):
        # Keep algorithm identical to trainBuckets.generateBuckets() in TS:
        #   - double loop over (w, h) in [min_r, max_r] step `step`
        #   - area within ±area_tolerance of base² (0.1 default)
        #   - max AR ratio ≤ max_ar (2.0 default)
        # With DEFAULT params both sides must still yield exactly 37 buckets —
        # covered by `studio/web/src/lib/trainBuckets.test.ts` asserting == 37.
        # These params are now runtime-configurable (multi-res bucket ratios);
        # the TS defaults are unchanged so the crop-page prediction stays aligned
        # for the default case (see docstring §SYNC).
        buckets = []
        base_area = base * base
        for w in range(min_r, max_r + 1, step):
            for h in range(min_r, max_r + 1, step):
                if abs(w * h - base_area) / base_area > area_tolerance:
                    continue
                if max(w/h, h/w) > max_ar:
                    continue
                buckets.append((w, h))
        return buckets

    def get_bucket(self, w, h):
        # Snap by ABSOLUTE AR distance — not relative. The TS port
        # `trainBuckets.snapToBucket()` mirrors this exactly.
        aspect = w / h
        best = (self.base_reso, self.base_reso)
        best_diff = float("inf")
        for bw, bh in self.buckets:
            diff = abs(aspect - bw/bh)
            if diff < best_diff:
                best_diff = diff
                best = (bw, bh)
        return best


class ImageDataset(Dataset):
    """
    图像数据集
    
    支持两种 caption 格式：
    1. JSON 文件（优先）- 支持分类 shuffle
    2. TXT 文件（回退）- 传统 shuffle
    """
    # 保持与 studio/datasets.py:IMAGE_EXTS 同步（anima_train.py 是独立 CLI 脚本，
    # 不强制 import studio package；改一处时另一处也要跟着改）。
    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

    def __init__(self, data_dir, resolution=1024, bucket_mgr=None,
                 shuffle_caption=False, keep_tokens=0, flip_augment=False,
                 tag_dropout=0.0, prefer_json=True, caption_override=None,
                 native_resolution=False, native_max_tokens=0,
                 native_max_side_tokens=0, native_over_budget="downscale"):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.bucket_mgr = bucket_mgr
        self.shuffle_caption = shuffle_caption
        self.keep_tokens = keep_tokens
        self.flip_augment = flip_augment
        self.tag_dropout = tag_dropout
        self.prefer_json = prefer_json
        self.caption_override = caption_override  # 正则集：统一 caption，如 "1girl, solo"
        # NaViT 原生定尺寸（navit_native_resolution，opt-in）：开启时单图按原生尺寸
        # floor 对齐 16px 定尺寸（绕过 ARB 桶量化），复用同一 resize-cover + center-crop
        # → 永远零 padding。关闭时（默认）走原有 bucket_mgr 路径，字节等价。
        self.native_resolution = bool(native_resolution)
        self.native_max_tokens = int(native_max_tokens or 0)
        self.native_max_side_tokens = int(native_max_side_tokens or 0)
        self.native_over_budget = str(native_over_budget or "downscale")
        
        # 尝试导入 caption_utils（直接导入避开 __init__.py）
        self.caption_utils = None
        if prefer_json:
            try:
                import importlib.util
                import sys
                
                # 直接加载 caption_utils.py（ADR 0003 PR-A 后 utils/ 在仓库根，
                # 不在 runtime/utils/；__file__ 是 runtime/training/dataset.py，
                # 因此要回溯三层 parent 到仓库根。）
                utils_path = Path(__file__).parent.parent.parent / "utils" / "caption_utils.py"
                if utils_path.exists():
                    spec = importlib.util.spec_from_file_location("caption_utils", utils_path)
                    caption_module = importlib.util.module_from_spec(spec)
                    sys.modules["caption_utils"] = caption_module
                    spec.loader.exec_module(caption_module)
                    
                    self.caption_utils = {
                        "load_and_build": caption_module.load_and_build_caption,
                        "load_json": caption_module.load_caption_json,
                        "normalize": caption_module.normalize_caption_json,
                        "build": caption_module.build_caption_from_json,
                    }
                    logger.info("JSON caption 模式已启用（分类 shuffle）")
                else:
                    logger.warning(f"caption_utils.py 未找到: {utils_path}")
            except Exception as e:
                logger.warning(f"caption_utils 加载失败: {e}，回退到 TXT 模式")
        
        self.samples = self._scan()
        json_count = sum(1 for s in self.samples if s.get("json_path"))
        txt_count = len(self.samples) - json_count
        unique_count = len(set(id(s) for s in self.samples))
        logger.info(f"数据集: {unique_count} 张图 → {len(self.samples)} 样本（含 repeat）(JSON: {json_count}, TXT: {txt_count})")

    @staticmethod
    def _parse_repeats_from_dir(name: str) -> int:
        """从文件夹名解析 Kohya 风格重复次数，如 '5_concept' → 5"""
        prefix = name.split("_", 1)[0]
        if prefix.isdigit():
            return max(int(prefix), 1)
        return 1

    def _make_sample(self, img_path):
        """为单张图构建 sample dict，找不到 caption 返回 None"""
        sample = {"image": img_path}
        json_path = img_path.with_suffix(".json")
        if self.prefer_json and json_path.exists():
            sample["json_path"] = json_path
            sample["txt_path"] = None
        else:
            txt_path = img_path.with_suffix(".txt")
            if not txt_path.exists():
                txt_path = img_path.with_suffix(".caption")
            if not txt_path.exists():
                return None
            sample["json_path"] = None
            sample["txt_path"] = txt_path
        return sample

    def _scan(self):
        """扫描数据集目录，支持 Kohya 风格文件夹重复。

        目录结构示例::

            dataset/
            ├── 1_old/       ← repeat 1
            │   ├── img.jpg
            │   └── img.txt
            └── 5_new/       ← repeat 5
                ├── img.jpg
                └── img.txt

        没有数字前缀的文件夹或根目录下的图片按 repeat=1 处理。
        """
        unique_samples = []
        folder_info = []  # (folder_name, repeat, count) for logging

        # 收集根目录下的图片（repeat=1）
        root_count = 0
        for p in sorted(self.data_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in self.EXTS:
                s = self._make_sample(p)
                if s:
                    s["_repeat"] = 1
                    unique_samples.append(s)
                    root_count += 1
        if root_count:
            folder_info.append(("(root)", 1, root_count))

        # 收集子文件夹中的图片（解析 repeat）
        for subdir in sorted(self.data_dir.iterdir()):
            if not subdir.is_dir():
                continue
            repeats = self._parse_repeats_from_dir(subdir.name)
            count = 0
            for img_path in sorted(subdir.rglob("*")):
                if img_path.suffix.lower() not in self.EXTS:
                    continue
                s = self._make_sample(img_path)
                if s:
                    s["_repeat"] = repeats
                    unique_samples.append(s)
                    count += 1
            if count:
                folder_info.append((subdir.name, repeats, count))

        # 展开 repeat：将每个样本按其 repeat 次数复制
        samples = []
        for s in unique_samples:
            r = s.pop("_repeat", 1)
            for _ in range(r):
                samples.append(s)

        # 日志：每个文件夹的 repeat 信息
        if folder_info:
            for name, rep, cnt in folder_info:
                logger.info(f"  文件夹 {name}: {cnt} 张 × repeat {rep} = {cnt * rep} 样本")

        return samples

    def _process_caption_txt(self, caption):
        """处理 TXT caption: 传统 tag 打乱 + keep_tokens"""
        if not caption:
            return ""
        if "," in caption:
            tags = [t.strip() for t in caption.split(",")]
        else:
            tags = caption.split()

        if self.keep_tokens > 0:
            kept = tags[:self.keep_tokens]
            rest = tags[self.keep_tokens:]
            if self.shuffle_caption:
                random.shuffle(rest)
            tags = kept + rest
        elif self.shuffle_caption:
            random.shuffle(tags)

        return ", ".join(tags)

    def _process_caption_json(self, json_path):
        """处理 JSON caption: 分类 shuffle"""
        if self.caption_utils is None:
            return None
        
        try:
            raw_json = self.caption_utils["load_json"](json_path)
            if raw_json is None:
                return None
            
            # 检查是否已经是标准格式
            if "tags" in raw_json and "meta" in raw_json:
                normalized = raw_json
            else:
                normalized = self.caption_utils["normalize"](raw_json)
            
            # 构建 caption（分类 shuffle）
            return self.caption_utils["build"](
                normalized,
                shuffle_appearance=self.shuffle_caption,
                shuffle_tags=self.shuffle_caption,
                shuffle_environment=self.shuffle_caption,
                tag_dropout=self.tag_dropout,
            )
        except Exception as e:
            logger.warning(f"JSON 处理失败 {json_path}: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def target_size_for(self, img_w, img_h):
        """单图目标像素尺寸 ``(tw, th)``。

        - ``native_resolution=False``（默认）：ARB 桶 ``bucket_mgr.get_bucket`` 或
          方形 ``resolution``（与历史行为逐字节一致）。
        - ``native_resolution=True``：``plan_native_fit_image`` 原生 floor-16 + 超预算
          downscale。cache 校验与 encode 都走本方法，保证尺寸一致。
        """
        if self.native_resolution:
            plan = plan_native_fit_image(
                img_w, img_h,
                max_tokens=self.native_max_tokens,
                max_side_tokens=self.native_max_side_tokens,
                align=16,
                over_budget=self.native_over_budget,
            )
            return plan.width, plan.height
        if self.bucket_mgr:
            return self.bucket_mgr.get_bucket(img_w, img_h)
        return self.resolution, self.resolution

    def __getitem__(self, idx):
        # 默认 path：DataLoader 不能传额外参数，所以由 flip_augment 决定是否随机翻转。
        # CachedLatentDataset 想显式控制 flip 时直接调 get_with_flip(idx, flip=...)，
        # 在 cache 阶段对每张图各 encode 一次 flip=False / flip=True，避免随机性 baked
        # 进 npz（kohya 风格双份 latent）。
        flip = self.flip_augment and random.random() > 0.5
        return self.get_with_flip(idx, flip=flip)

    def get_with_flip(self, idx, *, flip: bool):
        """带显式 flip 控制的 __getitem__。

        flip=True/False：强制翻 / 不翻，调用方负责决策；用于 cache 双份编码。
        flip 与 self.flip_augment 解耦，不读 self.flip_augment 也不掷随机数。
        """
        import numpy as np
        from PIL import Image
        sample = self.samples[idx]
        img = Image.open(sample["image"]).convert("RGB")

        # 获取 caption（正则集可用 caption_override 统一覆盖）
        caption = None
        if self.caption_override is not None:
            caption = self.caption_override
        elif sample.get("json_path"):
            caption = self._process_caption_json(sample["json_path"])

        if caption is None and sample.get("txt_path"):
            caption = sample["txt_path"].read_text(encoding="utf-8").strip()
            caption = self._process_caption_txt(caption)

        if caption is None:
            caption = ""

        # 目标尺寸：ARB 桶（默认）或原生 floor-16 定尺寸（native_resolution 开）
        tw, th = self.target_size_for(img.width, img.height)

        # 缩放裁剪
        scale = max(tw / img.width, th / img.height)
        nw, nh = int(img.width * scale), int(img.height * scale)
        img = img.resize((nw, nh), Image.LANCZOS)

        left = (nw - tw) // 2
        top = (nh - th) // 2
        img = img.crop((left, top, left + tw, top + th))

        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 转 tensor [-1, 1]
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        return {"pixel_values": tensor, "caption": caption}


class RepeatDataset(Dataset):
    """Kohya 风格数据集重复"""
    def __init__(self, dataset, repeats=1):
        self.dataset = dataset
        self.repeats = max(1, int(repeats))

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


class MergedDataset(Dataset):
    """合并主数据集与正则数据集（Kohya 风格 reg）"""
    def __init__(self, main_dataset, reg_dataset, reg_weight: float = 1.0):
        self.main_dataset = main_dataset
        self.reg_dataset = reg_dataset
        self.reg_weight = float(reg_weight)
        self._main_len = len(main_dataset)
        self._reg_len = len(reg_dataset)

        # 为 BucketBatchSampler 构建 bucket_for_index
        self.bucket_for_index = self._build_bucket_for_index()

    def _get_cached_dataset(self, d):
        if hasattr(d, "bucket_for_index"):
            return d
        if hasattr(d, "dataset"):
            return self._get_cached_dataset(d.dataset)
        return None

    def _build_bucket_for_index(self):
        main_cached = self._get_cached_dataset(self.main_dataset)
        reg_cached = self._get_cached_dataset(self.reg_dataset)
        buckets = []
        if main_cached and main_cached.bucket_for_index:
            main_base_len = len(main_cached.bucket_for_index)
            for idx in range(self._main_len):
                b = main_cached.bucket_for_index[idx % main_base_len]
                buckets.append(b if b is not None else (0, 0))
        else:
            buckets.extend([(0, 0)] * self._main_len)
        if reg_cached and reg_cached.bucket_for_index:
            reg_base_len = len(reg_cached.bucket_for_index)
            for idx in range(self._reg_len):
                b = reg_cached.bucket_for_index[idx % reg_base_len]
                buckets.append(b if b is not None else (0, 0))
        else:
            buckets.extend([(0, 0)] * self._reg_len)
        return buckets

    def __len__(self):
        return self._main_len + self._reg_len

    def __getitem__(self, idx):
        if idx < self._main_len:
            item = self.main_dataset[idx]
            item["loss_weight"] = 1.0
            item["is_reg"] = False
            return item
        item = self.reg_dataset[idx - self._main_len]
        item["loss_weight"] = self.reg_weight
        item["is_reg"] = True
        return item


class BucketBatchSampler:
    """Batch sampler that groups samples by bucket so latents in each batch have the same size."""
    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self._cached_dataset = self._get_cached_dataset(dataset)
        self._base_len = len(self._cached_dataset) if self._cached_dataset else 0

    def _get_cached_dataset(self, d):
        if hasattr(d, "bucket_for_index"):
            return d
        if hasattr(d, "dataset"):
            return self._get_cached_dataset(d.dataset)
        return None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        # ARB 下实际 batch 数 = Σ_bucket f(n_b, bs)；用全局 n 会偏（每桶各自有零头）。
        # 没有桶信息时退回到全局公式（线性 DataLoader 行为）。
        if self._cached_dataset is None:
            n = len(self.dataset)
            if self.drop_last:
                return n // self.batch_size
            return (n + self.batch_size - 1) // self.batch_size
        counts = {}
        for idx in range(len(self.dataset)):
            base_idx = idx % self._base_len
            bucket = self._cached_dataset.bucket_for_index[base_idx]
            if bucket is None:
                bucket = (0, 0)
            counts[bucket] = counts.get(bucket, 0) + 1
        total = 0
        for n in counts.values():
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        if self._cached_dataset is None:
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch
            return

        bucket_to_indices = {}
        for idx in range(len(self.dataset)):
            base_idx = idx % self._base_len
            bucket = self._cached_dataset.bucket_for_index[base_idx]
            if bucket is None:
                bucket = (0, 0)
            bucket_to_indices.setdefault(bucket, []).append(idx)

        buckets = list(bucket_to_indices.keys())
        if self.shuffle:
            rng.shuffle(buckets)
        for bucket in buckets:
            indices = bucket_to_indices[bucket]
            if self.shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch


# cache_encode_tiled 的默认像素预算：超过该像素数的图走分块 encode（4MP ≈ 2048²）
_CACHE_ENCODE_MAX_PIXELS = 4 * 1024 * 1024


class CachedLatentDataset(Dataset):
    """Kohya 风格 npz 文件缓存的数据集。

    flip_augment + cache_latents 同开时按 kohya 双份 latent 模式：
      - cache 阶段对每张图 encode 两次（flip=False / flip=True），分别存到
        npz 的 `latent` / `latent_flipped` 键
      - 训练时 __getitem__ 50% 概率取 flipped 版本
    旧版本静默把"cache 阶段那次随机翻转"baked 进 npz，导致 flip 永久失效 +
    50% 数据被永久镜像污染；新版通过 _is_cache_valid 检测缺 latent_flipped
    键，自动重 encode 修复。
    """
    def __init__(self, base_dataset, vae, device, dtype, cache_dir=None, cache_batch_size=1,
                 encode_tiled=False, encode_tile_px=1024, encode_tile_overlap=128,
                 encode_max_pixels=0):
        import numpy as np
        self.base_dataset = base_dataset
        self.base_image_dataset = self._get_base_image_dataset(base_dataset)
        self.np = np
        # 获取原始数据集的 samples 列表
        self.samples = self._get_base_samples(base_dataset)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.bucket_for_index = []
        # NaViT 打包器读的逐索引 token 数（patchify 后 = (h//2)*(w//2)）。
        # 与 bucket_for_index 同步填充；navit_packing 关闭时无人读取。
        self.token_count_for_index = []
        self.cache_batch_size = max(1, int(cache_batch_size or 1))
        # cache 是否需要双份 latent —— 取决于底层 ImageDataset.flip_augment
        self.flip_augment = bool(
            getattr(self.base_image_dataset, "flip_augment", False)
        )
        # cache_encode_tiled（opt-in）：超大图改走分块 encode + latent 羽化拼接，
        # 峰值显存 ∝ 单块像素。阈值内的图路径不变（逐字节等价）。
        self.encode_tiled = bool(encode_tiled)
        self.encode_tile_px = int(encode_tile_px or 1024)
        self.encode_tile_overlap = int(encode_tile_overlap or 128)
        self.encode_max_pixels = (
            int(encode_max_pixels) if int(encode_max_pixels or 0) > 0
            else _CACHE_ENCODE_MAX_PIXELS
        )
        self._build_cache(vae, device, dtype)

    def _get_base_samples(self, dataset):
        """获取原始 ImageDataset 的 samples"""
        if hasattr(dataset, "samples"):
            return dataset.samples
        elif hasattr(dataset, "dataset"):
            return self._get_base_samples(dataset.dataset)
        return []

    def _get_base_image_dataset(self, dataset):
        if hasattr(dataset, "samples") and hasattr(dataset, "bucket_mgr"):
            return dataset
        if hasattr(dataset, "dataset"):
            return self._get_base_image_dataset(dataset.dataset)
        return None

    def _expected_bucket_size(self, img_path):
        base = self.base_image_dataset
        if base is None:
            return None
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                # target_size_for 同时覆盖 ARB 桶与 native_resolution 路径，保证
                # cache 校验尺寸 == encode 实际尺寸（native 下取原生 floor-16）。
                if hasattr(base, "target_size_for"):
                    return base.target_size_for(img.width, img.height)
                if getattr(base, "bucket_mgr", None):
                    return base.bucket_mgr.get_bucket(img.width, img.height)
                resolution = int(getattr(base, "resolution"))
                return (resolution, resolution)
        except Exception:
            return None

    def _get_npz_path(self, img_path):
        """获取图像对应的 npz 缓存路径"""
        img_path = Path(img_path)
        return img_path.with_suffix(".npz")

    def _is_cache_valid(self, img_path, npz_path):
        """检查缓存是否有效（图像未修改，且格式兼容当前 flip_augment 设置）。

        - 缺 `latent` 键 / 其他模型的不兼容缓存 → 删除重 encode
        - flip_augment=True 且 npz 缺 `latent_flipped` 键 → 失效重 encode（旧
          单份 cache 即"flip 永久 baked"的污染状态，必须重 encode 修复）
        - flip_augment=False 且 npz 有 `latent_flipped` → 仍视为有效（双份
          cache 是 flip 模式的超集，关 flip 后只读 latent 不浪费）
        - bucket 尺寸不匹配 → 失效
        """
        if not npz_path.exists():
            return False
        if npz_path.stat().st_mtime < img_path.stat().st_mtime:
            return False
        try:
            with self.np.load(npz_path) as data:
                if "latent" not in data.files:
                    npz_path.unlink()
                    logger.debug(f"已删除不兼容缓存: {npz_path.name}")
                    return False
                if getattr(self, "flip_augment", False) and "latent_flipped" not in data.files:
                    return False
                expected_bucket = self._expected_bucket_size(img_path)
                if expected_bucket is not None:
                    if "bucket_w" not in data.files or "bucket_h" not in data.files:
                        return False
                    if (int(data["bucket_w"]), int(data["bucket_h"])) != expected_bucket:
                        return False
        except Exception:
            try:
                npz_path.unlink()
            except Exception:
                pass
            return False
        return True

    def _build_cache(self, vae, device, dtype):
        """构建/加载 npz 缓存。

        per-folder repeat（5_concept 前缀）让 ImageDataset.samples 里同一张图重复 N 次，
        但 npz 落点是 img_path.with_suffix(".npz") — 每张唯一图只对应一个 npz。
        按 npz_path 去重，每张图最多 encode 一次；否则同 npz 会被反复覆盖写 N 次
        （flip_augment 模式下再乘 2），首次构建 cache 时 80% 的 VAE encode 都是浪费。
        """
        logger.info("检查 VAE latent 缓存...")
        to_encode = []
        seen_npz = set()
        unique_total = 0
        for i, sample in enumerate(self.samples):
            img_path = sample["image"]
            npz_path = self._get_npz_path(img_path)
            if npz_path in seen_npz:
                continue
            seen_npz.add(npz_path)
            unique_total += 1
            if not self._is_cache_valid(img_path, npz_path):
                to_encode.append(i)

        if to_encode:
            logger.info(f"需要编码 {len(to_encode)}/{unique_total} 张图像...")
            self._encode_and_save(to_encode, vae, device, dtype)
        else:
            logger.info(f"所有 {unique_total} 张图像已缓存")

        self._fill_bucket_for_index()

    def _fill_bucket_for_index(self):
        """Fill bucket_for_index for all samples (needed for BucketBatchSampler).
        Uses latent spatial shape (h, w) as grouping key so batches have consistent tensor sizes."""
        self.bucket_for_index = [None] * len(self.samples)
        self.token_count_for_index = [0] * len(self.samples)
        for i in range(len(self.samples)):
            npz_path = self._get_npz_path(self.samples[i]["image"])
            if not npz_path.exists():
                continue
            with self.np.load(npz_path) as data:
                latent = data["latent"]
                s = latent.shape
            if len(s) == 5:
                _, _, _, h, w = s
            else:
                _, _, h, w = s
            self.bucket_for_index[i] = (int(h), int(w))
            # patch_spatial=2：patchify 后每图 token 数 = (h//2)*(w//2)。
            self.token_count_for_index[i] = (int(h) // 2) * (int(w) // 2)

    def _encode_and_save(self, indices, vae, device, dtype):
        """编码图像并保存为 npz。

        flip_augment=True 时对每张图编码两次（flip=False / flip=True）分别存到
        `latent` / `latent_flipped` 键；训练时 __getitem__ 随机选其一。
        flip_augment=False 时只编码一次，存 `latent`。

        按实际 bucket 尺寸分组并批量送入 VAE；不同尺寸不能 stack，分别攒批。
        cache_encode_tiled=True 时，超像素预算的图改走分块 encode + latent 羽化拼接。
        """
        base_img = self.base_image_dataset
        want_flip = self.flip_augment and base_img is not None
        pending = {}
        encoded_count = 0

        def _encode_pixels(pixel_tensors):
            pixels = torch.stack(pixel_tensors, dim=0).to(device, dtype=dtype)
            with torch.inference_mode():
                # 走 VAEWrapper.encode（含 auto/on 分块），大图/大 batch 不会撞 VRAM 崖
                latents = vae.encode(pixels.unsqueeze(2))
            return latents.detach().cpu().float()

        def _encode_tiled_single(pixel_tensor):
            """分块 encode 单张图（cache_encode_tiled 超像素预算时）。"""
            pixels = pixel_tensor.unsqueeze(0).to(device, dtype=dtype).unsqueeze(2)  # [1,C,1,H,W]
            with torch.inference_mode():
                # 直接用 VAEWrapper 的分块 encode（可配 tile 尺寸）：单层分块 + 统一
                # cosine 羽化，避免外层再套一层 vae.encode 导致的双重分块。
                lat = vae._tiled_encode(
                    pixels, self.encode_tile_px, self.encode_tile_overlap
                )
            return lat.detach().cpu().float()[0]

        def _flush(bucket_key):
            nonlocal encoded_count
            batch = pending.pop(bucket_key, [])
            if not batch:
                return

            h, w = int(batch[0]["bucket_h"]), int(batch[0]["bucket_w"])
            use_tiled = (
                getattr(self, "encode_tiled", False)
                and h > 0 and w > 0
                and h * w > self.encode_max_pixels
            )

            if use_tiled:
                logger.info(
                    "[cache-tiled] %dx%d 超像素预算，分块 encode（tile=%d overlap=%d）",
                    w, h, self.encode_tile_px, self.encode_tile_overlap,
                )
                for entry in batch:
                    lat = _encode_tiled_single(entry["pixels"])
                    lat_f = _encode_tiled_single(entry["pixels_flipped"]) if want_flip else None
                    npz_kwargs = {"latent": lat.numpy()}
                    if lat_f is not None:
                        npz_kwargs["latent_flipped"] = lat_f.numpy()
                    npz_path = self._get_npz_path(self.samples[entry["index"]]["image"])
                    self.np.savez(
                        npz_path,
                        bucket_w=entry["bucket_w"],
                        bucket_h=entry["bucket_h"],
                        **npz_kwargs,
                    )
                    encoded_count += 1
                    if encoded_count % 10 == 0 or encoded_count == len(indices):
                        logger.info(f"  编码进度: {encoded_count}/{len(indices)}")
                return

            latents = _encode_pixels([entry["pixels"] for entry in batch])
            if want_flip:
                latents_flipped = _encode_pixels([entry["pixels_flipped"] for entry in batch])
            else:
                latents_flipped = [None] * len(batch)

            for n, entry in enumerate(batch):
                npz_kwargs = {"latent": latents[n].numpy()}
                if want_flip:
                    npz_kwargs["latent_flipped"] = latents_flipped[n].numpy()

                npz_path = self._get_npz_path(self.samples[entry["index"]]["image"])
                self.np.savez(
                    npz_path,
                    bucket_w=entry["bucket_w"],
                    bucket_h=entry["bucket_h"],
                    **npz_kwargs,
                )
                encoded_count += 1
                if encoded_count % 10 == 0 or encoded_count == len(indices):
                    logger.info(f"  编码进度: {encoded_count}/{len(indices)}")

        logger.info(f"VAE cache batch size: {self.cache_batch_size}")
        for i in indices:
            if base_img is not None:
                # 显式控制 flip，避免随机性 baked 进 npz
                item = base_img.get_with_flip(i, flip=False)
            else:
                item = self.base_dataset[i]
            pixels = item["pixel_values"]
            _, ph, pw = pixels.shape
            bucket_w, bucket_h = pw, ph

            pixels_flipped = None
            if want_flip:
                item_f = base_img.get_with_flip(i, flip=True)
                pixels_flipped = item_f["pixel_values"]

            bucket_key = (bucket_h, bucket_w)
            pending.setdefault(bucket_key, []).append({
                "index": i,
                "pixels": pixels,
                "pixels_flipped": pixels_flipped,
                "bucket_w": bucket_w,
                "bucket_h": bucket_h,
            })
            if len(pending[bucket_key]) >= self.cache_batch_size:
                _flush(bucket_key)

        for bucket_key in list(pending):
            _flush(bucket_key)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        npz_path = self._get_npz_path(sample["image"])
        data = self.np.load(npz_path)
        # flip_augment=True 且 npz 有 latent_flipped 时 50% 概率取镜像版本，
        # 跟非 cache 路径 ImageDataset.__getitem__ 的 flip 概率一致。
        # 没有 latent_flipped 键（flip_augment=False 时的单份 cache）就只读 latent。
        use_flip = (
            self.flip_augment
            and "latent_flipped" in data.files
            and random.random() > 0.5
        )
        latent_key = "latent_flipped" if use_flip else "latent"
        latent = torch.from_numpy(data[latent_key])

        # 获取 base_dataset 的引用（处理可能的嵌套）
        base = self.base_dataset
        while hasattr(base, "dataset"):
            base = base.dataset
        
        # 处理 caption（正则集 caption_override 优先）
        caption = None
        if getattr(base, "caption_override", None) is not None:
            caption = base.caption_override
        elif sample.get("json_path") and hasattr(base, "_process_caption_json"):
            caption = base._process_caption_json(sample["json_path"])
        
        if caption is None and sample.get("txt_path"):
            caption = sample["txt_path"].read_text(encoding="utf-8").strip()
            if hasattr(base, "_process_caption_txt"):
                caption = base._process_caption_txt(caption)
        
        if caption is None:
            caption = ""
        
        return {"latent": latent, "caption": caption}


def collate_fn(batch):
    """DataLoader collate"""
    pixels = torch.stack([b["pixel_values"] for b in batch])
    captions = [b["caption"] for b in batch]
    result = {"pixel_values": pixels, "captions": captions}
    if "loss_weight" in batch[0]:
        result["loss_weight"] = torch.tensor([b["loss_weight"] for b in batch], dtype=torch.float32)
        result["is_reg"] = torch.tensor([b["is_reg"] for b in batch], dtype=torch.bool)
    return result


def collate_fn_cached(batch):
    """DataLoader collate for cached latents"""
    latents = torch.stack([b["latent"] for b in batch])
    captions = [b["caption"] for b in batch]
    result = {"latents": latents, "captions": captions}
    if "loss_weight" in batch[0]:
        result["loss_weight"] = torch.tensor([b["loss_weight"] for b in batch], dtype=torch.float32)
        result["is_reg"] = torch.tensor([b["is_reg"] for b in batch], dtype=torch.bool)
    return result


# ===========================================================================
# NaViT / Patch-n-Pack: token-budget 打包（NavitPackBatchSampler / collate）
# ===========================================================================


def pack_indices_by_budget(token_counts, token_budget, order, max_images_per_pack=0):
    """贪心 next-fit 打包：把样本索引分进 token 总数 ≤ budget 的包。

    NaViT 块对角打包无 padding，一个包的代价 = 各图 token 数之和。``order`` 是已打乱
    的索引序列；自身 token 数超 budget 的图单独成包。覆盖 ``order`` 每索引恰好一次。
    """
    packs = []
    cur, cur_sum = [], 0
    cap = int(max_images_per_pack or 0)
    budget = int(token_budget)
    for idx in order:
        n = int(token_counts[idx])
        over_budget = bool(cur) and (cur_sum + n > budget)
        over_count = cap > 0 and len(cur) >= cap
        if over_budget or over_count:
            packs.append(cur)
            cur, cur_sum = [], 0
        cur.append(idx)
        cur_sum += n
    if cur:
        packs.append(cur)
    return packs


def pack_indices_ffd_windowed(token_counts, token_budget, order,
                              max_images_per_pack=0, window=0):
    """First-Fit-Decreasing 窗口化打包：在（已打乱的）``order`` 的窗口内做 FFD。

    经典 FFD（按尺寸降序、逐个放入第一个能放下的桶）比 next-fit 打包更紧。``window``
    把 ``order`` 切成连续窗口、FFD 在窗口内运行——``order`` 每 epoch 重洗 → 窗口成员
    跨 epoch 变化，恢复大部分填充收益又保留 batch 多样性。``window<=0`` = 单个全局窗口。
    """
    budget = int(token_budget)
    cap = int(max_images_per_pack or 0)
    win = int(window or 0)
    order = list(order)
    if win <= 0:
        windows = [order]
    else:
        windows = [order[i:i + win] for i in range(0, len(order), win)]

    packs = []
    for w in windows:
        items = sorted(w, key=lambda i: int(token_counts[i]), reverse=True)
        bins = []  # each: [list_of_indices, summed_tokens]
        for idx in items:
            n = int(token_counts[idx])
            placed = False
            for b in bins:
                over_count = cap > 0 and len(b[0]) >= cap
                if (not over_count) and (b[1] + n <= budget):
                    b[0].append(idx)
                    b[1] += n
                    placed = True
                    break
            if not placed:
                bins.append([[idx], n])
        packs.extend(b[0] for b in bins)
    return packs


def _lookup_token_count_walk(d, idx):
    """遍历数据集包装器解析样本 token 数（MergedDataset 两分支 + 单链兜底）。"""
    main = getattr(d, "main_dataset", None)
    reg = getattr(d, "reg_dataset", None)
    if main is not None and reg is not None:
        ml = getattr(d, "_main_len", len(main))
        if idx < ml:
            return _lookup_token_count_walk(main, idx)
        return _lookup_token_count_walk(reg, idx - ml)
    inner = getattr(d, "dataset", None)
    if inner is not None and inner is not d and not isinstance(inner, list):
        return _lookup_token_count_walk(inner, idx % len(inner))
    counts = getattr(d, "token_count_for_index", None)
    if counts is not None and len(counts) > 0:
        return int(counts[idx % len(counts)])
    inner = getattr(d, "base_dataset", None)
    if inner is not None and inner is not d:
        return _lookup_token_count_walk(inner, idx % len(inner))
    return 0


def _walk_attr_list(dataset, attr):
    """在单链包装器（RepeatDataset/CachedLatentDataset）里找叶数据集的 per-index
    列表属性 ``attr``，按 ``% len`` 映射到 ``len(dataset)``。MergedDataset（两分支）
    或属性不存在时返回 None。"""
    cur = dataset
    for _ in range(12):
        if getattr(cur, "main_dataset", None) is not None and getattr(cur, "reg_dataset", None) is not None:
            return None  # MergedDataset: not a single chain
        v = getattr(cur, attr, None)
        if v is not None and len(v) > 0:
            n = len(dataset)
            return [v[i % len(v)] for i in range(n)]
        nxt = getattr(cur, "dataset", None)
        if nxt is None or nxt is cur or isinstance(nxt, list):
            nxt = getattr(cur, "base_dataset", None)
        if nxt is None or nxt is cur:
            return None
        cur = nxt
    return None


def dataset_token_counts(dataset, patch_spatial=2):
    """NaViT 打包的逐索引 token 数。

    优先用 ``token_count_for_index``（CachedLatentDataset 填充）；退化时从
    ``bucket_for_index=(h,w)`` 推导 ``(h//ps)*(w//ps)``；都不可用则逐索引 walk 兜底
    （返回 0 → 打包器 fail-fast）。
    """
    counts = _walk_attr_list(dataset, "token_count_for_index")
    if counts is not None and any(int(c) > 0 for c in counts):
        return [int(c) for c in counts]

    shapes = _walk_attr_list(dataset, "bucket_for_index")
    if shapes is not None:
        ps = max(1, int(patch_spatial))
        derived = []
        for s in shapes:
            if not s:
                derived.append(0)
                continue
            h, w = int(s[0]), int(s[1])
            derived.append((h // ps) * (w // ps))
        if any(c > 0 for c in derived):
            return derived

    return [int(_lookup_token_count_walk(dataset, i) or 0) for i in range(len(dataset))]


class NavitPackBatchSampler:
    """为 NaViT/Patch-n-Pack 块对角训练产出数据集索引包。

    每个产出的列表是一个打包训练序列：其各图 token 数之和 ≤ ``token_budget``，整包
    作为一个零 padding 的块对角 forward。把"每步图片数"与单图形状解耦——不同 token
    数和长宽比的图可共享一个包，小数据集也能填满大 effective batch。
    """

    def __init__(self, dataset, token_budget, max_images_per_pack=0,
                 shuffle=True, seed=42, drop_last=False,
                 strategy="next_fit", ffd_window=256):
        self.dataset = dataset
        self.token_budget = int(token_budget)
        self.max_images_per_pack = int(max_images_per_pack or 0)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.strategy = str(strategy or "next_fit").lower()
        if self.strategy not in ("next_fit", "ffd"):
            raise ValueError(
                f"navit pack strategy 必须是 'next_fit' 或 'ffd'，收到 {strategy!r}"
            )
        self.ffd_window = int(ffd_window or 0)
        self.epoch = 0
        self.token_counts = dataset_token_counts(dataset)
        self._cached_packs = None
        # Fail-fast：全 0 token 数 → 无法解析每图尺寸 → 整集打成一个巨包 → OOM。
        if not self.token_counts or not any(int(c) > 0 for c in self.token_counts):
            raise RuntimeError(
                "[NavitPack] 无法解析任一样本的 token 数（token_count_for_index 与 "
                "bucket_for_index 都不可用/全 0）。NaViT 打包需要缓存数据集 "
                "（cache_latents=true）以拿到每图 latent 形状。"
            )
        mx = max(self.token_counts) if self.token_counts else 0
        if self.token_counts and self.token_budget < mx:
            logger.warning(
                "[NavitPack] token_budget=%d < 最大单图 token=%d：该图将单独成包，"
                "可能超出预算并 OOM。建议 token_budget >= 最大单图 token。",
                self.token_budget, mx,
            )
        logger.info(
            "[NavitPack] dataset_len=%d token_budget=%d max_images_per_pack=%s "
            "strategy=%s ffd_window=%s (token 数范围 %d..%d)",
            len(self.token_counts), self.token_budget,
            self.max_images_per_pack or "∞", self.strategy,
            (self.ffd_window or "全局") if self.strategy == "ffd" else "-",
            min(self.token_counts) if self.token_counts else 0, mx,
        )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._cached_packs = None

    def _build_packs(self):
        order = list(range(len(self.token_counts)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        if self.strategy == "ffd":
            packs = pack_indices_ffd_windowed(
                self.token_counts, self.token_budget, order,
                self.max_images_per_pack, self.ffd_window,
            )
        else:
            packs = pack_indices_by_budget(
                self.token_counts, self.token_budget, order, self.max_images_per_pack
            )
        if self.drop_last and len(packs) > 1:
            last_sum = sum(self.token_counts[i] for i in packs[-1])
            if last_sum < self.token_budget:
                packs = packs[:-1]
        return packs

    def __iter__(self):
        packs = self._build_packs()
        self._cached_packs = packs
        for pack in packs:
            yield pack

    def __len__(self):
        if self._cached_packs is None:
            self._cached_packs = self._build_packs()
        return len(self._cached_packs)


def collate_fn_navit_pack(batch):
    """NaViT 打包 collate。

    一个包内缓存 latent 有不同空间形状、无法 stack；保留为列表。训练循环把每图
    patchify 为 token、拼接 per-image RoPE grid、编码 caption 并拼接 text_seqlens，
    再调用 ``forward_packed_navit``。
    """
    latents = [b["latent"] for b in batch]        # each [C, T, h_i, w_i]
    captions = [b["caption"] for b in batch]
    images = [b.get("image", "") for b in batch]
    result = {
        "navit_latents": latents,
        "captions": captions,
        "images": images,
    }
    # 正则集降权：透传 loss_weight / is_reg 供训练循环在 per-image loss 上应用。
    if "loss_weight" in batch[0]:
        result["loss_weight"] = torch.tensor(
            [b["loss_weight"] for b in batch], dtype=torch.float32
        )
        result["is_reg"] = torch.tensor([b["is_reg"] for b in batch], dtype=torch.bool)
    return result
