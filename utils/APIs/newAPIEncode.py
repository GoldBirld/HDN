# utils/APIs/newAPIEncode.py
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import os
from typing import Optional

# 检测相关（按需懒加载）
try:
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    )
except Exception:
    fasterrcnn_resnet50_fpn = None
    FasterRCNN_ResNet50_FPN_Weights = None

# 文本编码（按需懒加载）
try:
    from transformers import AutoModel, AutoConfig, CLIPTextModel
except Exception:
    AutoModel = AutoConfig = CLIPTextModel = None


import re
# 更强壮的模式：中英文括号/冒号、正负情感的多种写法
_LEAK_PAT = re.compile(r"[（(]?(?:积极|正向|好评|消极|负向|差评|positive|negative)[^，。,.)）]*[)）]?", re.I)

def strip_leak_markers(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return _LEAK_PAT.sub("", text).strip()



# -----------------------------
# Label 映射（保证 negative=0, positive=1）
# -----------------------------
def _label_to_name(label) -> str:
    s = str(label).strip().lower()
    if s in {"1", "pos", "positive", "true", "yes"}:
        return "positive"
    if s in {"0", "neg", "negative", "false", "no"}:
        return "negative"
    # 兜底：能转成数字时 1 为正，其余为负
    try:
        if int(s) == 1:
            return "positive"
    except Exception:
        pass
    return "negative"


def get_resize(image_size: int) -> int:
    side = 1
    for _ in range(20):
        if side >= image_size:
            return side
        side *= 2
    return image_size


def _safe_device(requested: Optional[str] = None):
    """统一的设备选择：CUDA 不可用时自动回退到 CPU。"""
    try:
        if requested is None:
            return torch.device("cuda") if (hasattr(torch, "cuda") and torch.cuda.is_available()) else torch.device("cpu")
        req = str(requested).lower()
        if req.startswith("cuda"):
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                return torch.device(req)
            print("[api_encode][warn] CUDA 不可用或 PyTorch 为 CPU 版，已回退到 CPU。")
            return torch.device("cpu")
        return torch.device("cpu")
    except Exception as e:
        print(f"[api_encode][warn] 选择设备失败({requested})：{e}，已回退到 CPU。")
        return torch.device("cpu")


@torch.no_grad()
def _frcnn_roi_vec(frcnn, img_tensor_0_1, topk=16, device="cpu"):
    """从一张 0~1 的 (3,H,W) 图像中提取 Faster R-CNN ROI 特征并加权成 1024 维向量。"""
    img_tensor_0_1 = img_tensor_0_1.to(device)
    image_list, _ = frcnn.transform([img_tensor_0_1], None)
    feats = frcnn.backbone(image_list.tensors)
    if isinstance(feats, torch.Tensor):
        feats = {"0": feats}
    proposals, _ = frcnn.rpn(image_list, feats)
    props = proposals[0][: topk]
    if props.numel() == 0:
        return torch.zeros(1024, device="cpu")
    box_feats = frcnn.roi_heads.box_roi_pool(feats, [props], image_list.image_sizes)
    box_feats = frcnn.roi_heads.box_head(box_feats)      # (k,1024)
    cls_scores, _ = frcnn.roi_heads.box_predictor(box_feats)
    probs = torch.softmax(cls_scores, dim=-1)[:, :-1]    # 去背景
    conf = probs.max(dim=-1).values                      # (k,)
    w = conf / (conf.sum() + 1e-6)
    roi_vec = (w.unsqueeze(0) @ box_feats).squeeze(0).to("cpu")  # (1024)
    return roi_vec


def _is_clip_text(backbone: str):
    """
    判断 text_backbone 是否是 CLIP 文本编码器；返回 (is_clip, enc_name)
    - 支持 "clip:openai/clip-vit-base-patch32" 或 "openai:clip-vit-base-patch32"
    """
    if backbone is None:
        return False, None
    name = str(backbone)
    low = name.lower()
    if low.startswith("clip:") or low.startswith("openai:"):
        enc_name = name.split(":", 1)[1] if ":" in name else name
        return True, enc_name
    if AutoConfig is not None:
        try:
            cfg = AutoConfig.from_pretrained(name)
            if getattr(cfg, "model_type", "") == "clip":
                return True, name
        except Exception:
            pass
    return False, name


# -----------------------------
# 主干图像 transform（按阶段切换随机增强）
# -----------------------------
def _build_main_img_transform(config):
    """
    根据 config 构造主干图像 transform：
    - config.augment: 训练 True 使用强增广；验证/测试 False 使用确定性流程
    - config.img_norm: 'imagenet' 或 'clip'
    可选超参：
      aug_rrc_scale_min (默认 0.7), aug_color (默认 0.2), hflip_p (默认 0.5)
    """
    do_aug = bool(getattr(config, "augment", False))
    hflip_p = float(getattr(config, "hflip_p", 0.5))
    rrc_scale_min = float(getattr(config, "aug_rrc_scale_min", 0.7))
    color = float(getattr(config, "aug_color", 0.2))
    norm = str(getattr(config, "img_norm", "imagenet")).lower()

    # 归一化空间
    if norm == "clip":
        mean = [0.48145466, 0.4578275, 0.40821073]
        std  = [0.26862954, 0.26130258, 0.27577711]
    else:  # imagenet
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

    # 版本兼容
    interp_mod = getattr(transforms, "InterpolationMode", None)
    BICUBIC = interp_mod.BICUBIC if interp_mod is not None else Image.BICUBIC
    has_antialias = "antialias" in transforms.Resize.__init__.__code__.co_varnames

    size = getattr(config, "image_size", 224)
    size_arg = size if isinstance(size, (tuple, list)) else int(size)

    steps = []
    if do_aug:
        steps.append(
            transforms.RandomResizedCrop(
                size=size_arg,
                scale=(rrc_scale_min, 1.0),
                interpolation=BICUBIC
            )
        )
        if hflip_p > 0:
            steps.append(transforms.RandomHorizontalFlip(p=hflip_p))
        if color > 0:
            steps.append(
                transforms.ColorJitter(
                    brightness=color, contrast=color, saturation=color, hue=0.0
                )
            )
    else:
        # 验证/测试：确定性 Resize + CenterCrop（与 ROI/缓存对齐）
        resize_kwargs = {"interpolation": BICUBIC}
        if has_antialias:
            resize_kwargs["antialias"] = True
        steps += [
            transforms.Resize(get_resize(size_arg), **resize_kwargs),
            transforms.CenterCrop(size_arg),
        ]

    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(steps)


# -----------------------------
# 编码主函数
# -----------------------------
def api_encode(
    data, labelvocab, config,
    *,
    # ROI 相关
    use_frcnn: bool = False,
    frcnn_topk: int = 16,
    roi_cache_dir: str = None,
    frcnn_device: str = None,
    roi_cache_load_only: bool = False,
    # 文本预计算相关
    use_text_precompute: bool = False,
    text_cache_dir: str = None,
    text_device: str = None,
    text_cache_load_only: bool = False,
):
    """
    常规返回: guids, encoded_texts(input_ids list), encoded_imgs(tensor list), encoded_labels
    若 use_frcnn=True 额外返回 encoded_rois(list[Tensor(1024,)])
    若 use_text_precompute=True 额外返回 encoded_tok_embeds(list[Tensor(T, enc_dim)])

    load-only:
      - roi_cache_load_only=True 只从 roi_cache_dir 读取，缺失直接报错
      - text_cache_load_only=True 只从 text_cache_dir 读取，缺失直接报错
    """

    # ====== 标签注册（保证 id 映射：negative=0, positive=1）======
    labelvocab.add_label('negative')
    labelvocab.add_label('positive')

    # ====== tokenizer（与 TextModel 对齐）======
    tb = getattr(config, "text_backbone", None)
    if tb is None or str(tb).strip() == "":
        tok_name = getattr(config, "bert_name", "bert-base-chinese")
    else:
        tok_name = tb.split(":", 1)[1] if str(tb).lower().startswith("openai:") else tb
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)

    cfg_max = int(getattr(config, "max_seq_len", 128))
    tk_max = getattr(tokenizer, "model_max_length", cfg_max)
    if tk_max is None or tk_max > 10000:
        tk_max = cfg_max
    max_len = min(cfg_max, int(tk_max))

    # ====== 合并配置（保持向后兼容）======
    use_frcnn = bool(getattr(config, "use_frcnn_regions", use_frcnn))
    frcnn_topk = int(getattr(config, "frcnn_topk", frcnn_topk))
    roi_cache_dir = getattr(config, "roi_cache_dir", None) or getattr(config, "api_roi_cache_dir", roi_cache_dir)
    roi_cache_load_only = bool(getattr(config, "roi_cache_load_only", roi_cache_load_only))
    frcnn_device = getattr(config, "frcnn_device", frcnn_device)

    use_text_precompute = bool(getattr(config, "precompute_text_embeds", use_text_precompute))
    text_cache_dir = getattr(config, "text_cache_dir", text_cache_dir)
    text_cache_load_only = bool(getattr(config, "text_cache_load_only", text_cache_load_only))
    text_device = getattr(config, "text_precompute_device", text_device)

    # ====== 主干图像 transform（按 augment/img_norm）======
    img_transform = _build_main_img_transform(config)

    # ====== FRCNN：固定 0~1 的、无增广的图像流（和主干增广解耦）======
    raw_transform = None
    frcnn = None
    frcnn_dev = torch.device("cpu")
    if use_frcnn:
        # ROI 流始终确定性：Resize+CenterCrop+ToTensor(0~1)
        raw_transform = transforms.Compose([
            transforms.Resize(get_resize(getattr(config, "image_size", 224)), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(getattr(config, "image_size", 224)),
            transforms.ToTensor()
        ])
        if roi_cache_dir:
            os.makedirs(roi_cache_dir, exist_ok=True)
        if roi_cache_load_only:
            print(f"[api_encode] FRCNN: load-only from {roi_cache_dir}")
        else:
            print(f"[api_encode] FRCNN: will build proposals if cache missing; topk={frcnn_topk}, cache={roi_cache_dir}")

        def _lazy_build_frcnn():
            nonlocal frcnn, frcnn_dev
            if frcnn is None:
                if fasterrcnn_resnet50_fpn is None or FasterRCNN_ResNet50_FPN_Weights is None:
                    raise ImportError("需要 torchvision 检测模块：fasterrcnn_resnet50_fpn / FasterRCNN_ResNet50_FPN_Weights")
                frcnn_dev = _safe_device(frcnn_device)
                frcnn = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
                try:
                    frcnn.to(frcnn_dev)
                except Exception as e:
                    print(f"[api_encode][warn] FRCNN 放到 {frcnn_dev} 失败：{e}，改用 CPU。")
                    frcnn = frcnn.cpu()
                    frcnn_dev = torch.device("cpu")
                for p in frcnn.parameters():
                    p.requires_grad = False
            return frcnn
    else:
        print("[api_encode] FRCNN disabled.")

    # ====== 文本预计算（懒加载）======
    txt_encoder = None
    txt_dev = torch.device("cpu")
    is_clip_text, enc_name = _is_clip_text(getattr(config, "text_backbone", tok_name))
    if use_text_precompute:
        if text_cache_dir:
            os.makedirs(text_cache_dir, exist_ok=True)
        if text_cache_load_only:
            print(f"[api_encode] Text precompute: load-only from {text_cache_dir}")
        else:
            print(f"[api_encode] Text precompute: will build if cache missing -> {text_cache_dir}")

        def _lazy_build_text_encoder():
            nonlocal txt_encoder, txt_dev
            if txt_encoder is None:
                if is_clip_text:
                    if CLIPTextModel is None:
                        raise ImportError("需要 transformers.CLIPTextModel 才能预计算 CLIP 文本特征。")
                    txt_encoder = CLIPTextModel.from_pretrained(enc_name)
                else:
                    if AutoModel is None:
                        raise ImportError("需要 transformers.AutoModel 才能预计算 BERT 文本特征。")
                    print('use bert')
                    txt_encoder = AutoModel.from_pretrained(enc_name)
                txt_dev = _safe_device(text_device)
                try:
                    txt_encoder.eval().to(txt_dev)
                except Exception as e:
                    print(f"[api_encode][warn] 文本编码器放到 {txt_dev} 失败：{e}，改用 CPU。")
                    txt_dev = torch.device("cpu")
                    txt_encoder = txt_encoder.cpu().eval()
                for p in txt_encoder.parameters():
                    p.requires_grad = False
            return txt_encoder
    else:
        print("[api_encode] Text precompute disabled.")

    # ====== 编码循环 =======
    guids, encoded_texts, encoded_imgs, encoded_labels = [], [], [], []
    encoded_rois = [] if use_frcnn else None
    encoded_tok_embeds = [] if use_text_precompute else None

    for guid, text, img, label in tqdm(data, desc='----- [Encoding]'):
        guid = str(guid)
        guids.append(guid)

        # 文本 ids
        text = ("" if text is None else str(text)).replace('#', '')
        text = strip_leak_markers(text)  # ★★★ 关键：去掉“积极词/消极词”尾缀
        input_ids = tokenizer.encode(
            text, add_special_tokens=True, truncation=True, max_length=max_len
        )

        encoded_texts.append(input_ids)

        # 文本预计算：读缓存或懒计算
        if use_text_precompute:
            t_path = os.path.join(text_cache_dir, f"{guid}.pt") if text_cache_dir else None
            tok_embed = None
            if t_path and os.path.isfile(t_path):
                try:
                    v = torch.load(t_path, map_location="cpu")
                    if isinstance(v, torch.Tensor) and v.dim() == 2:
                        tok_embed = v.float()
                except Exception:
                    tok_embed = None
            else:
                if text_cache_load_only:
                    raise FileNotFoundError(f"[api_encode] 缺少文本缓存：{t_path}（处于 load-only 模式，不再计算）")
                _lazy_build_text_encoder()
                with torch.no_grad():
                    ids = torch.tensor(input_ids, dtype=torch.long, device=txt_dev).unsqueeze(0)  # (1,T)
                    att = torch.ones_like(ids, device=txt_dev)
                    out = txt_encoder(input_ids=ids, attention_mask=att, return_dict=True)
                    tok_embed = out.last_hidden_state.squeeze(0).to("cpu").contiguous()  # (T, enc_dim)
                if t_path:
                    try:
                        torch.save(tok_embed, t_path)
                    except Exception:
                        pass
            encoded_tok_embeds.append(tok_embed)

        # 图像主干输入（按 augment/img_norm）
        if not isinstance(img, Image.Image):
            if isinstance(img, np.ndarray):
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)
                img = Image.fromarray(img.astype("uint8"))
            else:
                img = Image.new("RGB", (224, 224), (0, 0, 0))
        encoded_imgs.append(img_transform(img))

        # ROI：读缓存或懒计算（始终 0~1、无增广）
        if use_frcnn:
            r_path = os.path.join(roi_cache_dir, f"{guid}.pt") if roi_cache_dir else None
            roi_vec = None
            if r_path and os.path.isfile(r_path):
                try:
                    v = torch.load(r_path, map_location="cpu")
                    if isinstance(v, torch.Tensor) and v.numel() == 1024:
                        roi_vec = v.float()
                except Exception:
                    roi_vec = None
            else:
                if roi_cache_load_only:
                    raise FileNotFoundError(f"[api_encode] 缺少 ROI 缓存：{r_path}（处于 load-only 模式，不再计算）")
                _lazy_build_frcnn()
                with torch.no_grad():
                    raw_tensor = raw_transform(img)  # (3,H,W), 0~1
                    roi_vec = _frcnn_roi_vec(frcnn, raw_tensor, topk=frcnn_topk, device=frcnn_dev)  # (1024,)
                if r_path:
                    try:
                        torch.save(roi_vec, r_path)
                    except Exception:
                        pass
            encoded_rois.append(roi_vec)

        # 标签
        name = _label_to_name(label)
        encoded_labels.append(labelvocab.label_to_id(name))

    # 返回
    if use_frcnn and use_text_precompute:
        return guids, encoded_texts, encoded_imgs, encoded_labels, encoded_rois, encoded_tok_embeds
    elif use_frcnn:
        return guids, encoded_texts, encoded_imgs, encoded_labels, encoded_rois
    elif use_text_precompute:
        return guids, encoded_texts, encoded_imgs, encoded_labels, encoded_tok_embeds
    else:
        return guids, encoded_texts, encoded_imgs, encoded_labels
