# model.py —— BridgeTower 版（工程输出结构一致）
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.FocalLoss import FocalLoss

# ===== transformers (BridgeTower) =====
try:
    from transformers import BridgeTowerModel, BridgeTowerConfig
except Exception:
    BridgeTowerModel = BridgeTowerConfig = None


# -----------------------------
# BridgeTower 主干封装
# -----------------------------
class BridgeTowerBackbone(nn.Module):
    """
    轻封装 BridgeTower：
      - 返回跨模态融合后的 (B, H) 向量（优先 pooler_output；否则回退 [CLS]）。
      - 保留梯度检查点与可选冻结。
    """
    def __init__(self,
                 backbone_name: str = "BridgeTower/bridgetower-base",
                 hidden_size: Optional[int] = None,
                 freeze_backbone: bool = False,
                 gradient_checkpointing: bool = True):
        super().__init__()
        assert BridgeTowerModel is not None, "请安装包含 BridgeTowerModel 的 transformers 版本。"
        self.model = BridgeTowerModel.from_pretrained(backbone_name)
        base_h = int(self.model.config.hidden_size)
        self.hidden = base_h if hidden_size is None else int(hidden_size)

        # 若外部指定 hidden 与主干不一致，则做一次线性投影（一般不会触发）
        self._need_proj = (self.hidden != base_h)
        self.out_proj = nn.Linear(base_h, self.hidden) if self._need_proj else nn.Identity()

        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    @staticmethod
    def _safe_pool(out):
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output   # (B,H)
        hs = out.last_hidden_state     # (B,L,H)
        return hs[:, 0, :]             # 退回 [CLS]

    def forward(self,
                input_ids: torch.Tensor,                # (B,T)
                attention_mask: torch.Tensor,           # (B,T)
                pixel_values: torch.Tensor,             # (B,3,H,W)，来自 BridgeTowerProcessor
                pixel_mask: Optional[torch.Tensor]=None,# (B,Patches)，可选
                token_type_ids: Optional[torch.Tensor]=None
                ) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            return_dict=True
        )
        fused = self._safe_pool(out)        # (B, base_H)
        fused = self.out_proj(fused)        # (B, H)
        return fused


# -----------------------------
# 顶层分类器（工程对齐版）
# -----------------------------
class Model(nn.Module):
    """
    与工程保持一致：
      forward(texts, texts_mask, imgs, labels=None, **kwargs) -> (pred, loss)

    - texts        -> input_ids
    - texts_mask   -> attention_mask
    - imgs         -> pixel_values（必须由 BridgeTowerProcessor 生成）
    - kwargs 可选：pixel_mask, token_type_ids
    """
    def __init__(self, config):
        super().__init__()
        self.backbone = BridgeTowerBackbone(
            backbone_name=getattr(config, "bridgetower_backbone", "BridgeTower/bridgetower-base"),
            hidden_size=getattr(config, "bridgetower_hidden", None),
            freeze_backbone=getattr(config, "freeze_bridgetower", False),
            gradient_checkpointing=getattr(config, "bridgetower_grad_ckpt", True),
        )
        hidden = self.backbone.hidden

        # 简单线性头（与原工程风格一致）
        self.dropout = nn.Dropout(getattr(config, "dropout", 0.1))
        self.cls = nn.Linear(hidden, getattr(config, "num_classes", 3))

        # 损失函数：CE / Focal
        loss_name = str(getattr(config, "loss", "ce")).lower()
        if loss_name in ("focal", "focalloss"):
            focal_alpha = getattr(config, "focal_alpha", None)       # 如 [α_neg, α_neu, α_pos]
            focal_gamma = float(getattr(config, "focal_gamma", 2.0))
            class_weight = getattr(config, "class_weight", None)
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

        # 可选：torch.compile
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
                imgs: torch.Tensor,              # pixel_values (BridgeTowerProcessor)
                labels: Optional[torch.Tensor] = None,
                roi_vec: Optional[torch.Tensor] = None,          # 兼容老签名（忽略）
                token_embeds: Optional[torch.Tensor] = None,     # 兼容老签名（忽略）
                token_lengths: Optional[torch.Tensor] = None,    # 兼容老签名（忽略）
                **kwargs):
        pixel_mask = kwargs.get("pixel_mask", None)
        token_type_ids = kwargs.get("token_type_ids", None)

        feats = self.backbone(
            input_ids=texts,
            attention_mask=texts_mask,
            token_type_ids=token_type_ids,
            pixel_values=imgs,
            pixel_mask=pixel_mask
        )  # (B,H)

        logits = self.cls(self.dropout(feats))              # (B,C)
        pred = torch.argmax(logits, dim=-1).to(torch.long)  # 明确 int64

        if labels is not None:
            loss = self.crit(logits, labels)
        else:
            # 返回与 logits 同设备/同精度的 0.0，避免混精度下的类型不一致
            loss = torch.zeros((), dtype=logits.dtype, device=logits.device)

        return pred, loss
