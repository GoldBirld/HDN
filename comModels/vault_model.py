# model.py
from typing import Optional, Tuple, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.FocalLoss import FocalLoss  # 与你工程保持一致

# ========= transformers =========
try:
    from transformers import ViltModel, ViltConfig
except Exception:
    ViltModel = ViltConfig = None


# -----------------------------
# VAuLT 组件：跨注意力适配器（Prompts + Cross-Attn）
# -----------------------------
class CrossAttnAdapter(nn.Module):
    """
    用 learnable prompts（可选注入外部情感特征）作为 Query，
    对 ViLT 的跨模态序列 encoder_hidden_states 做一次 cross-attention，
    再经轻量 FFN，最后对 prompts 维度做平均池化得到 (B,H)。
    """
    def __init__(self, hidden_size: int, num_prompts: int = 8,
                 aux_dim: int = 0, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden_size
        self.num_prompts = int(num_prompts)
        self.aux_dim = int(aux_dim)

        # 可学习提示/查询 (P, H)
        self.prompts = nn.Parameter(torch.randn(self.num_prompts, hidden_size) * 0.02)

        # 外部情感特征（可选），映射到 H 后与 prompts 相加
        self.aux_proj = nn.Linear(aux_dim, hidden_size) if aux_dim > 0 else None

        # Cross-Attention（Q=prompts，K/V=ViLT 序列）
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self,
                encoder_hidden_states: torch.Tensor,           # (B,L,H)
                encoder_padding_mask: Optional[torch.Tensor],  # (B,L) True=padding
                aux_feats: Optional[torch.Tensor] = None       # (B,aux_dim) or None
                ) -> torch.Tensor:
        B = encoder_hidden_states.size(0)

        q = self.prompts.unsqueeze(0).expand(B, -1, -1)  # (B,P,H)
        if self.aux_proj is not None and aux_feats is not None:
            q = q + self.aux_proj(aux_feats).unsqueeze(1)  # 注入外部情感特征

        x, _ = self.attn(query=q,
                         key=encoder_hidden_states,
                         value=encoder_hidden_states,
                         key_padding_mask=encoder_padding_mask)  # (B,P,H)

        x = self.ln1(q + self.drop(x))
        y = self.ffn(x)
        y = self.ln2(x + self.drop(y))                     # (B,P,H)

        pooled = y.mean(dim=1)                             # (B,H)
        return pooled


# -----------------------------
# VAuLT 主干：ViLT + 文本/视觉适配器 + 门控融合
# -----------------------------
class VAuLTBackbone(nn.Module):
    def __init__(self,
                 backbone_name: str = "dandelin/vilt-b32-mlm",
                 hidden_size: Optional[int] = None,
                 freeze_vilt: bool = False,
                 gradient_checkpointing: bool = True,
                 auto_preprocess_images: bool = True,
                 # 适配器与门控
                 num_text_prompts: int = 8,
                 num_visual_prompts: int = 8,
                 text_aux_dim: int = 0,
                 visual_aux_dim: int = 0,
                 adapter_heads: int = 8,
                 adapter_dropout: float = 0.1,
                 use_gate: bool = True):
        super().__init__()
        assert ViltModel is not None, "请安装/升级 transformers，并确保 ViltModel 可用。"
        self.vilt = ViltModel.from_pretrained(backbone_name)
        base_h = self.vilt.config.hidden_size
        self.hidden = base_h if (hidden_size is None) else int(hidden_size)

        # 若用户显式更改 hidden_size，做线性投影以对齐（通常不需要）
        self.need_proj = (self.hidden != base_h)
        self.proj = nn.Linear(base_h, self.hidden) if self.need_proj else nn.Identity()

        if gradient_checkpointing and hasattr(self.vilt, "gradient_checkpointing_enable"):
            self.vilt.gradient_checkpointing_enable()
        if freeze_vilt:
            for p in self.vilt.parameters():
                p.requires_grad = False

        # 两个跨注意力适配器
        self.text_adapter = CrossAttnAdapter(
            hidden_size=self.hidden,
            num_prompts=num_text_prompts,
            aux_dim=text_aux_dim,
            num_heads=adapter_heads,
            dropout=adapter_dropout
        )
        self.visual_adapter = CrossAttnAdapter(
            hidden_size=self.hidden,
            num_prompts=num_visual_prompts,
            aux_dim=visual_aux_dim,
            num_heads=adapter_heads,
            dropout=adapter_dropout
        )

        # 门控头：从 [CLS] 预测 gate_t, gate_v
        self.use_gate = bool(use_gate)
        if self.use_gate:
            self.gate_head = nn.Sequential(
                nn.Linear(self.hidden, self.hidden),
                nn.GELU(),
                nn.Linear(self.hidden, 2)
            )
        else:
            self.gate_head = None

        # 内置图像预处理（以匹配 ViLT 期望）
        self.auto_pre = bool(auto_preprocess_images)
        self.register_buffer("_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("_std",  torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self._target_size = int(getattr(getattr(self.vilt, "config", None), "image_size", 384))

    @staticmethod
    def _to_key_padding_mask(attn_mask: Optional[torch.Tensor], L: int) -> Optional[torch.Tensor]:
        if attn_mask is None:
            return None
        if attn_mask.size(1) != L:
            if attn_mask.size(1) > L:
                attn_mask = attn_mask[:, :L]
            else:
                pad = torch.zeros(attn_mask.size(0), L - attn_mask.size(1),
                                  dtype=attn_mask.dtype, device=attn_mask.device)
                attn_mask = torch.cat([attn_mask, pad], dim=1)
        return (attn_mask == 0)

    def _maybe_preprocess_images(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        接受原工程传入的 (B,3,H,W) 张量，自动：
          1) 若像素范围大于 1，先 /255；
          2) resize 到 ViLT 的 image_size（默认 384）；
          3) 按 ViLT 习惯用 (x-0.5)/0.5 归一化到 [-1,1]。
        """
        x = imgs
        if not self.auto_pre:
            return x
        if x.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64) or x.max() > 1.0:
            x = x.float() / 255.0
        if x.shape[-1] != self._target_size or x.shape[-2] != self._target_size:
            x = F.interpolate(x, size=(self._target_size, self._target_size),
                              mode="bicubic", align_corners=False)
        x = (x - self._mean) / self._std
        return x

    def forward(self,
                input_ids: torch.Tensor,                      # (B,T)
                attention_mask: torch.Tensor,                 # (B,T)
                pixel_values: torch.Tensor,                   # (B,3,H,W) —— 原工程 imgs 直传
                pixel_mask: Optional[torch.Tensor] = None,    # (B, Patches) 可选
                token_type_ids: Optional[torch.Tensor] = None,
                text_aux_feats: Optional[torch.Tensor] = None,   # (B,Ft) 可选
                visual_aux_feats: Optional[torch.Tensor] = None   # (B,Fv) 可选
                ) -> torch.Tensor:
        """
        返回 VAuLT 融合后的 (B,H) 向量
        """
        # 自动预处理到 ViLT 期望空间（与原工程 imgs 输入兼容）
        pixel_values = self._maybe_preprocess_images(pixel_values)

        out = self.vilt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            return_dict=True,
            output_hidden_states=False
        )
        hs = self.proj(out.last_hidden_state)     # (B,L,H)
        cls = hs[:, 0, :]                         # (B,H) ViLT 的 [CLS]

        # 给适配器一个（可选的）padding mask；多数情况下可为 None
        key_padding_mask = None  # self._to_key_padding_mask(attention_mask, hs.size(1))

        t_aug = self.text_adapter(hs, key_padding_mask, aux_feats=text_aux_feats)     # (B,H)
        v_aug = self.visual_adapter(hs, key_padding_mask, aux_feats=visual_aux_feats) # (B,H)

        if self.use_gate:
            gates = torch.sigmoid(self.gate_head(cls))  # (B,2)
            g_t = gates[:, 0:1]
            g_v = gates[:, 1:2]
            fused = cls + g_t * t_aug + g_v * v_aug
        else:
            fused = (cls + t_aug + v_aug) / 3.0

        return fused  # (B,H)


# -----------------------------
# 顶层分类模型（输出结构与原工程保持一致：return pred, loss）
# -----------------------------
class Model(nn.Module):
    """
    与原工程保持同名 `Model` 类，并保留 forward(texts, texts_mask, imgs, labels, **kwargs) 接口。
    其中：
      - texts        -> input_ids
      - texts_mask   -> attention_mask
      - imgs         -> 原始图像张量 (B,3,H,W)，内部自动预处理（可通过 config.auto_preprocess_images=False 关闭）
      - 可选 kwargs:  pixel_mask / token_type_ids / text_aux_feats / visual_aux_feats
    输出：
      - pred: (B,)  = argmax(logits)
      - loss: 标量；如未传 labels，则返回与原工程一致的 0.0 张量
    """
    def __init__(self, config):
        super().__init__()

        # === VAuLT 主干 ===
        self.backbone = VAuLTBackbone(
            backbone_name=getattr(config, "vilt_backbone", "dandelin/vilt-b32-mlm"),
            hidden_size=getattr(config, "vilt_hidden", None),
            freeze_vilt=getattr(config, "freeze_vilt", False),
            gradient_checkpointing=getattr(config, "vilt_grad_ckpt", True),
            auto_preprocess_images=getattr(config, "auto_preprocess_images", True),
            num_text_prompts=getattr(config, "num_text_prompts", 8),
            num_visual_prompts=getattr(config, "num_visual_prompts", 8),
            text_aux_dim=getattr(config, "text_aux_dim", 0),
            visual_aux_dim=getattr(config, "visual_aux_dim", 0),
            adapter_heads=getattr(config, "adapter_heads", 8),
            adapter_dropout=getattr(config, "adapter_dropout", 0.1),
            use_gate=getattr(config, "use_gate", True),
        )
        hidden = self.backbone.hidden

        # === 分类头 ===
        self.dropout = nn.Dropout(getattr(config, "dropout", 0.1))
        self.cls = nn.Linear(hidden, getattr(config, "num_classes", 3))

        # === 损失 ===
        loss_name = str(getattr(config, "loss", "ce")).lower()
        if loss_name in ("focal", "focalloss"):
            focal_alpha = getattr(config, "focal_alpha", None)       # 例如 [α_neg, α_neu, α_pos]
            focal_gamma = float(getattr(config, "focal_gamma", 2.0))
            class_weight = getattr(config, "class_weight", None)     # 与 CE 的 class_weight 一致的额外权重
            if float(getattr(config, "label_smoothing", 0.0)) != 0.0:
                print("[warn] FocalLoss 下建议关闭 label_smoothing，已忽略。")
            self.crit = FocalLoss(alpha=focal_alpha,
                                  gamma=focal_gamma,
                                  class_weight=class_weight,
                                  reduction="mean")
        else:
            weight_cfg = getattr(config, "class_weight", None)
            self.crit = nn.CrossEntropyLoss(
                weight=(torch.tensor(weight_cfg, dtype=torch.float32) if weight_cfg is not None else None),
                label_smoothing=float(getattr(config, "label_smoothing", 0.0))
            )

        # === 可选：torch.compile 子模块 ===
        if getattr(config, "compile_submodules", False):
            self._maybe_compile()

    def _maybe_compile(self):
        try:
            self.backbone = torch.compile(self.backbone, mode="max-autotune")
            self.cls = torch.compile(self.cls, mode="max-autotune")
        except Exception:
            pass

    def forward(self,
                texts: torch.Tensor,             # input_ids
                texts_mask: torch.Tensor,        # attention_mask
                imgs: torch.Tensor,              # 原始图像张量 (B,3,H,W)
                labels: Optional[torch.Tensor] = None,
                roi_vec: Optional[torch.Tensor] = None,  # 兼容老签名（忽略）
                token_embeds: Optional[torch.Tensor] = None,  # 兼容老签名（忽略）
                token_lengths: Optional[torch.Tensor] = None, # 兼容老签名（忽略）
                **kwargs):
        """
        兼容原工程 Trainer 的调用方式：
          - 额外可在 kwargs 里传入：
              pixel_mask: (B, Lp)
              token_type_ids: (B,T)
              text_aux_feats: (B,Ft)
              visual_aux_feats: (B,Fv)
        """
        pixel_mask = kwargs.get("pixel_mask", None)
        token_type_ids = kwargs.get("token_type_ids", None)
        text_aux_feats = kwargs.get("text_aux_feats", None)
        visual_aux_feats = kwargs.get("visual_aux_feats", None)

        fused = self.backbone(
            input_ids=texts,
            attention_mask=texts_mask,
            pixel_values=imgs,
            pixel_mask=pixel_mask,
            token_type_ids=token_type_ids,
            text_aux_feats=text_aux_feats,
            visual_aux_feats=visual_aux_feats
        )  # (B,H)

        logits = self.cls(self.dropout(fused))  # (B,C)
        pred = torch.argmax(logits, dim=-1)

        if labels is not None:
            loss = self.crit(logits, labels)
        else:
            loss = torch.tensor(0.0, device=logits.device)

        # —— 与你的输出结构完全一致 ——
        return pred, loss
