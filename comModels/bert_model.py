# model.py
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.FocalLoss import FocalLoss

# ========= transformers =========
try:
    from transformers import AutoModel, AutoConfig, CLIPTextModel, CLIPModel
except Exception:
    AutoModel = AutoConfig = CLIPTextModel = CLIPModel = None

# ========= torchvision（只保留 ResNet）=========
try:
    import torchvision
    from torchvision.models import resnet18, resnet34, resnet50, resnet152
    from torchvision.models import (
        ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet152_Weights,
    )
except Exception:
    torchvision = None
    resnet18 = resnet34 = resnet50 = resnet152 = None
    ResNet18_Weights = ResNet34_Weights = ResNet50_Weights = ResNet152_Weights = None


# -----------------------------
# 词级注意力：批量向量化（已修复 FP16 溢出）
# -----------------------------
class WordAttention(nn.Module):
    """词级注意力（批量向量化版），输入 H:(B,T,D), mask:(B,T)，输出 ui:(B,D)"""
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 原始打分
        score = torch.tanh(self.proj(H))      # (B,T,H)
        score = self.v(score).squeeze(-1)     # (B,T)

        # 确保 bool mask
        if mask.dtype != torch.bool:
            mask = mask != 0

        # 在 FP32 中做 masking + softmax，避免半精度溢出
        score_fp32 = score.float()
        score_fp32 = score_fp32.masked_fill(~mask, torch.finfo(score_fp32.dtype).min)
        a = torch.softmax(score_fp32, dim=-1).to(H.dtype)  # (B,T)

        # 加权求和
        ui = torch.einsum("bt,btd->bd", a, H)  # (B,D)
        return ui


# -----------------------------
# 文本编码（BERT → BiGRU + 注意力）
#   - 常规：input_ids + attention_mask → BERT → BiGRU → 注意力 → 句级GRU
#   - 预计算：token_embeds + attention_mask（可选）

#   H_sent:(B,1,2h)  S_doc:(B,2h)  mask_sent:(B,1)
# -----------------------------
class TextModel(nn.Module):
    def __init__(self,
                 backbone: str = "bert-base-chinese",
                 bert_dim: int = 768,          # 投影后的特征维度
                 gru_hidden: int = 384,        # BiGRU 隐层（每向）
                 attn_hidden: int = 512):      # 词级注意力隐层
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers.AutoModel 不可用，无法加载 BERT 文本骨干。")

        # 加载 BERT
        self.encoder = AutoModel.from_pretrained(backbone)
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if hasattr(self.encoder, "config"):
            try:
                self.encoder.config.use_cache = False
            except Exception:
                pass

        # 编码器输出维度
        self._enc_dim = getattr(getattr(self.encoder, "config", None), "hidden_size", bert_dim)

        # 适配到 bert_dim
        self.in_proj = nn.Linear(self._enc_dim, bert_dim) if self._enc_dim != bert_dim else nn.Identity()

        # BiGRU + 注意力 + 句级 GRU
        self.bigru_word = nn.GRU(bert_dim, gru_hidden, bidirectional=True, batch_first=True)
        self.word_att = WordAttention(2 * gru_hidden, attn_hidden)
        self.bigru_sent = nn.GRU(2 * gru_hidden, gru_hidden, bidirectional=True, batch_first=True)

    # ===== 预计算路径（直接传入 token 向量） =====
    def forward_from_embeds(
        self,
        token_embeds: torch.Tensor,       # (B,T,enc_dim 或 bert_dim)
        attention_mask: torch.Tensor,     # (B,T)
        lengths: Optional[torch.Tensor]=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, enc_dim = token_embeds.shape

        # 投影到 bert_dim
        if isinstance(self.in_proj, nn.Identity):
            X = token_embeds
        else:
            X = self.in_proj(token_embeds) if enc_dim == self._enc_dim else nn.Linear(enc_dim, self.in_proj.out_features, bias=False).to(token_embeds.device)(token_embeds)

        # 变长打包
        if attention_mask.dtype != torch.bool:
            mask_bool = attention_mask != 0
        else:
            mask_bool = attention_mask
        if lengths is None:
            lengths = mask_bool.long().sum(dim=1).detach().cpu()
        else:
            lengths = lengths.detach().cpu()

        packed = nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
        H_word_packed, _ = self.bigru_word(packed)
        H_word, _ = nn.utils.rnn.pad_packed_sequence(H_word_packed, batch_first=True, total_length=T)  # (B,T,2h)

        ui = self.word_att(H_word, attention_mask)     # (B,2h)
        Hi, _ = self.bigru_sent(ui.unsqueeze(1))       # (B,1,2h)

        mask_sent = attention_mask.new_ones((B, 1))
        return Hi, ui, mask_sent

    # ===== 常规路径（BERT 编码） =====
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_embeds: torch.Tensor = None,     # 兼容 Trainer 的可选参数
        token_lengths: torch.Tensor = None,    # 兼容 Trainer 的可选参数
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # 直通预计算向量
        if token_embeds is not None:
            return self.forward_from_embeds(token_embeds, attention_mask, token_lengths)

        B, T = input_ids.size(0), input_ids.size(1)

        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        X = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]  # (B,T,_enc_dim)
        X = self.in_proj(X)  # (B,T,bert_dim)

        lengths = attention_mask.sum(dim=1).detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
        H_word_packed, _ = self.bigru_word(packed)
        H_word, _ = nn.utils.rnn.pad_packed_sequence(H_word_packed, batch_first=True, total_length=T)  # (B,T,2h)

        ui = self.word_att(H_word, attention_mask)         # (B,2h)
        Hi, _ = self.bigru_sent(ui.unsqueeze(1))           # (B,1,2h)
        H_sent = Hi
        S_doc = ui
        mask_sent = attention_mask.new_ones((B, 1))
        return H_sent, S_doc, mask_sent


# -----------------------------
# 图像编码：ResNet 全局；ROI 由数据阶段提供
# -----------------------------
class ImageModel(nn.Module):
    """
    对外接口: forward(images, roi_vec=None) -> (P, r)
      - P: (B, proj_dim) 全局向量（CLIP/ResNet）
      - r: (B, proj_dim) 区域向量；若提供 roi_vec(=B×1024) 则映射后返回；否则 r=P
    """
    def __init__(self,
                 proj_dim: int = 512,
                 use_frcnn_regions: bool = False,
                 frcnn_topk: int = 16,
                 backbone: str = "resnet50",
                 global_only: bool = False,
                 region_heads: int = 8,
                 region_pool: Optional[int] = None,
                 activation: str = "relu",
                 input_space: str = "imagenet",
                 roi_in_dim: int = 1024):
        super().__init__()
        self.proj_dim = proj_dim
        self.global_only = bool(global_only)
        self.input_space = input_space.lower()

        act = activation.lower()
        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.GELU()

        # 归一化常量
        self.register_buffer("_imnet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("_imnet_std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.register_buffer("_clip_mean",  torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1))
        self.register_buffer("_clip_std",   torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1))

        # 全局主干：CLIP 或 ResNet
        self._use_clip_visual = False
        self.clip = None
        self.full_resnet = None
        self.cnn = None
        self.pool = None
        self.proj_global = None
        self._clip_image_size = None

        name_l = backbone.lower()
        if name_l.startswith("openai") or (("clip" in name_l) and CLIPModel is not None):
            if CLIPModel is None:
                raise ImportError("transformers.CLIPModel 不可用，但你配置了 CLIP 视觉骨干。")
            clip_name = backbone.split(":", 1)[1] if ":" in backbone else backbone
            self.clip = CLIPModel.from_pretrained(clip_name)
            clip_dim = self.clip.config.projection_dim  # 通常 512
            self.proj_global = nn.Identity() if clip_dim == proj_dim else nn.Linear(clip_dim, proj_dim)
            self._use_clip_visual = True
            vc = getattr(self.clip, "config", None)
            vc = getattr(vc, "vision_config", None)
            self._clip_image_size = getattr(vc, "image_size", 224)
            # 默认冻结
            for p in self.clip.vision_model.parameters():
                p.requires_grad = False
            if hasattr(self.clip, "visual_projection"):
                for p in self.clip.visual_projection.parameters():
                    p.requires_grad = False
        else:
            if torchvision is None:
                raise ImportError("torchvision 不可用，无法加载 ResNet 视觉骨干。")
            if name_l == "resnet18":
                net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if ResNet18_Weights else None)
                feat_dim = 512
            elif name_l == "resnet34":
                net = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if ResNet34_Weights else None)
                feat_dim = 512
            elif name_l == "resnet50":
                net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if ResNet50_Weights else None)
                feat_dim = 2048
            elif name_l == "resnet152":
                net = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1 if ResNet152_Weights else None)
                feat_dim = 2048
            else:
                raise ValueError(f"Unsupported image backbone: {backbone}")

            self.full_resnet = net
            self.cnn = nn.Sequential(*list(net.children())[:-2])  # (B,C,h,w)
            self.cnn.to(memory_format=torch.channels_last)
            self.pool = nn.AdaptiveAvgPool2d((1,1))
            self.proj_global = nn.Linear(feat_dim, proj_dim)

        # ROI 投影层（离线 ROI -> proj_dim）
        self.proj_roi = nn.Linear(roi_in_dim, proj_dim)

    # ---------- 归一化互转 ----------
    def _to_raw(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_space == "imagenet":
            return (x * self._imnet_std + self._imnet_mean).clamp(0, 1)
        elif self.input_space == "clip":
            return (x * self._clip_std + self._clip_mean).clamp(0, 1)
        elif self.input_space == "raw":
            return x.clamp(0, 1)
        else:
            raise ValueError(f"Unsupported input_space: {self.input_space}")

    def _to_clip(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_space == "clip":
            return x
        elif self.input_space == "imagenet":
            raw = x * self._imnet_std + self._imnet_mean
            return (raw - self._clip_mean) / self._clip_std
        elif self.input_space == "raw":
            return (x - self._clip_mean) / self._clip_std
        else:
            raise ValueError(f"Unsupported input_space: {self.input_space}")

    # ---------- 前向 ----------
    def forward(self, images: torch.Tensor, roi_vec: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = images.contiguous(memory_format=torch.channels_last)

        # 全局特征 P
        if self._use_clip_visual:
            x_clip = self._to_clip(x)
            if self._clip_image_size is not None and (
                x_clip.shape[-1] != self._clip_image_size or x_clip.shape[-2] != self._clip_image_size
            ):
                x_clip = F.interpolate(x_clip, size=(self._clip_image_size, self._clip_image_size),
                                       mode="bicubic", align_corners=False)
            g_vec = self.clip.get_image_features(pixel_values=x_clip)  # (B, clip_dim)
            g_vec = F.normalize(g_vec, dim=-1)
            P = self.act(self.proj_global(g_vec))
        else:
            feat = self.cnn(x)                # (B,C,h,w)
            g = self.pool(feat).flatten(1)    # (B,C)
            P = self.act(self.proj_global(g)) # (B,proj_dim)

        # 区域特征 r
        if roi_vec is not None:
            r = self.act(self.proj_roi(roi_vec.to(x.device)))
        else:
            r = P  # 无 ROI 时退化为全局向量

        return P, r


# -----------------------------
# 对齐图文融合（已修复 FP16 溢出）
# -----------------------------
class AlignedImageTextFusion(nn.Module):
    def __init__(self, dim_text: int, proj_dim: int,
                 attn_dropout: float = 0.1,
                 temperature: float = None):
        super().__init__()
        self.Wq = nn.Linear(dim_text, proj_dim)          # (Dt→Dp)
        self.gamma = nn.Parameter(torch.randn(proj_dim)) # (Dp,)
        # 温度：默认 sqrt(Dp)
        self.tau = nn.Parameter(torch.tensor(
            temperature if temperature is not None else math.sqrt(proj_dim),
            dtype=torch.float32
        ), requires_grad=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.ln = nn.LayerNorm(dim_text)                 # 残差 + LN

    def forward(self, H: torch.Tensor, mask_sent: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """
        H: (B,L,Dt)  mask_sent:(B,L)  P:(B,Dp)
        return c: (B,Dt)
        """
        B, L, Dt = H.shape
        mask = mask_sent.bool()

        Q  = F.gelu(self.Wq(H))                 # (B,L,Dp)
        Pe = P.unsqueeze(1).expand(-1, L, -1)   # (B,L,Dp)

        # 乘性融合 + 温度缩放
        fuse  = Pe * Q + Q                      # (B,L,Dp)
        score = torch.einsum("bld,d->bl", fuse, self.gamma) / (self.tau + 1e-6)  # (B,L)

        # 在 FP32 中做 masking + softmax
        score_fp32 = score.float()
        score_fp32 = score_fp32.masked_fill(~mask, torch.finfo(score_fp32.dtype).min)
        lam   = torch.softmax(score_fp32, dim=-1).to(H.dtype)    # (B,L)
        lam   = self.attn_drop(lam)             # 注意力 dropout（正则）

        c = torch.einsum("bl,bld->bd", lam, H)  # (B,Dt)

        # 残差：与 masked-mean 做融合，再 LayerNorm
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (H * mask.unsqueeze(-1)).sum(dim=1) / denom  # (B,Dt)
        c = self.ln(c + pooled)
        return c


# -----------------------------
# 低秩张量融合（向量化稳定版）
# -----------------------------
class LowRankFusion(nn.Module):
    def __init__(self, dim_c: int, dim_r: int, out_dim: int, rank: int = 4):
        super().__init__()
        self.rank = rank
        self.Wc = nn.Parameter(torch.empty(rank, dim_c + 1, out_dim))
        self.Wr = nn.Parameter(torch.empty(rank, dim_r + 1, out_dim))
        nn.init.xavier_uniform_(self.Wc)
        nn.init.xavier_uniform_(self.Wr)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        B = c.size(0)
        Zc = torch.cat([c, c.new_ones(B, 1)], dim=-1)          # (B, dc+1)
        Zr = torch.cat([r, r.new_ones(B, 1)], dim=-1)          # (B, dr+1)
        Vc = torch.einsum('bd,rdp->brp', Zc, self.Wc)          # (B,rank,out_dim)
        Vr = torch.einsum('bd,rdp->brp', Zr, self.Wr)          # (B,rank,out_dim)
        O = (Vc * Vr).sum(dim=1) + self.bias                   # (B,out_dim)
        return self.norm(O)


# -----------------------------
# 分层动态邻域融合（HDN）
# -----------------------------
class EnergyHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, T: float = 1.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self.T = T

    def forward(self, x: torch.Tensor):
        z = self.fc(x) / max(self.T, 1e-4)
        energy = self.T * torch.logsumexp(z, dim=-1)
        logp = -energy / max(self.T, 1e-4)
        return energy, logp


class HierarchicalDynamicNeighborhood(nn.Module):
    def __init__(self, dim_v: int, dim_o: int, dim_s: int, hidden: int,
                 num_classes: int, T: float = 1.0):
        super().__init__()
        self.Ev = EnergyHead(dim_v, num_classes, T)
        self.Eo = EnergyHead(dim_o, num_classes, T)
        self.Es = EnergyHead(dim_s, num_classes, T)

        self.Pv = nn.Linear(dim_v, hidden)
        self.Po = nn.Linear(dim_o, hidden)
        self.Ps = nn.Linear(dim_s, hidden)
        self.norm = nn.LayerNorm(hidden)

    @staticmethod
    def _object_layer(v, o, s):
        return v + v * o + v * s, o + o * v + o * s, s + s * v + s * o

    @staticmethod
    def _eps(v, o, s):
        d = torch.tensor([
            1 / max(v.size(-1), 1),
            1 / max(o.size(-1), 1),
            1 / max(s.size(-1), 1)
        ], device=v.device).view(1, 3)
        return torch.softmax(torch.log(d), dim=-1)

    def forward(self, V: torch.Tensor, O: torch.Tensor, S: torch.Tensor):
        eV, lpV = self.Ev(V)
        eO, lpO = self.Eo(O)
        eS, lpS = self.Es(S)
        alpha = torch.softmax(torch.stack([-eV, -eO, -eS], dim=-1), dim=-1)  # (B,3)

        V1 = self.norm(self.Pv(V)) * alpha[:, 0:1] * (1 + lpV.unsqueeze(-1))
        O1 = self.norm(self.Po(O)) * alpha[:, 1:2] * (1 + lpO.unsqueeze(-1))
        S1 = self.norm(self.Ps(S)) * alpha[:, 2:3] * (1 + lpS.unsqueeze(-1))

        v2, o2, s2 = self._object_layer(V1, O1, S1)
        v2, o2, s2 = self.norm(v2), self.norm(o2), self.norm(s2)

        eps3 = self._eps(v2, o2, s2)
        mix3 = eps3[:, 0:1] * v2 + eps3[:, 1:2] * o2 + eps3[:, 2:3] * s2
        v3 = o3 = s3 = self.norm(mix3)

        eps4 = self._eps(v3, o3, s3)
        mix4 = eps4[:, 0:1] * v3 + eps4[:, 1:2] * o3 + eps4[:, 2:3] * s3
        v4 = o4 = s4 = self.norm(mix4)

        rho4 = v4 * o4 * s4
        aux = {"alpha": alpha, "eps3": eps3, "eps4": eps4}
        return rho4, aux


# -----------------------------
# 整体模型：文本/图像 → 对齐 → 低秩 → HDN → 分类
# -----------------------------
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 文本侧（BERT）
        self.text_model = TextModel(
            backbone=getattr(config, "text_backbone", "bert-base-chinese"),
            bert_dim=getattr(config, "text_hidden", 768),
            gru_hidden=getattr(config, "text_gru_hidden", 384),
            attn_hidden=getattr(config, "hdn_hidden", 512),
        )
        Dt = 2 * getattr(config, "text_gru_hidden", 384)

        # 图像侧（与原版一致）
        self.img_model = ImageModel(
            proj_dim=getattr(config, "hdn_hidden", 512),
            backbone=getattr(config, "image_backbone", "resnet50"),
            global_only=getattr(config, "image_global_only", False),
            activation=getattr(config, "image_activation", "relu"),
            input_space=getattr(config, "image_input_space", "imagenet"),
            roi_in_dim=getattr(config, "roi_in_dim", 1024),
        )

        # 融合（保持不变）
        self.align = AlignedImageTextFusion(
            dim_text=Dt,
            proj_dim=getattr(config, "hdn_hidden", 512),
            attn_dropout=getattr(config, "attn_dropout", 0.1),
            temperature=getattr(config, "attn_temperature", None),  # None → √proj_dim
        )

        self.lowrank = LowRankFusion(
            dim_c=Dt,
            dim_r=getattr(config, "hdn_hidden", 512),
            out_dim=getattr(config, "lowrank_out", 512),
            rank=getattr(config, "lowrank_rank", 4),
        )

        self.hdn = HierarchicalDynamicNeighborhood(
            dim_v=getattr(config, "hdn_hidden", 512),
            dim_o=getattr(config, "lowrank_out", 512),
            dim_s=Dt,
            hidden=getattr(config, "hdn_hidden", 512),
            num_classes=getattr(config, "num_classes", 2),
            T=getattr(config, "temperature", 1.0),
        )

        self.dropout = nn.Dropout(getattr(config, "dropout", 0.1))
        self.cls = nn.Linear(getattr(config, "hdn_hidden", 512), getattr(config, "num_classes", 2))

        # ---- 选择损失：CE 或 Focal ----
        loss_name = str(getattr(config, "loss", "ce")).lower()
        if loss_name in ("focal", "focalloss"):
            focal_alpha = getattr(config, "focal_alpha", None)
            focal_gamma = float(getattr(config, "focal_gamma", 2.0))
            class_weight = getattr(config, "class_weight", None)
            if hasattr(config, "label_smoothing") and float(getattr(config, "label_smoothing", 0.0)) != 0.0:
                print("[warn] FocalLoss 下不建议使用 label_smoothing，已忽略。")
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

        # 可选：冻结主干层（BERT 可按最后 n 层微调）
        self._maybe_freeze_backbones(config)

        # 可选：torch.compile（保留原版钩子名 bigru_word/bigru_sent）
        if getattr(config, "compile_submodules", False):
            self._maybe_compile_submodules()

    def _maybe_freeze_backbones(self, config):
        # 文本：只训练最后 n 层
        n_train_layers = int(getattr(config, "train_bert_last_n_layers", 4))
        bert = getattr(self.text_model, "encoder", None)
        if bert is not None and hasattr(bert, "encoder") and hasattr(bert.encoder, "layer"):
            if n_train_layers >= 0:
                for p in bert.parameters():
                    p.requires_grad = False
                L = len(bert.encoder.layer)
                for i in range(max(0, L - n_train_layers), L):
                    for p in bert.encoder.layer[i].parameters():
                        p.requires_grad = True
            pooler = getattr(bert, "pooler", None)
            if pooler is not None:
                for p in pooler.parameters():
                    p.requires_grad = True

        # 图像：仅当是 ResNet 时可按层名选择
        train_layers = tuple(getattr(config, "train_resnet_layers", ("layer4",)))
        if hasattr(self.img_model, "full_resnet") and self.img_model.full_resnet is not None:
            for name, p in self.img_model.full_resnet.named_parameters():
                flag = any(name.startswith(tl) for tl in train_layers)
                p.requires_grad = flag

    def _maybe_compile_submodules(self):
        try:
            self.text_model.bigru_word = torch.compile(self.text_model.bigru_word, mode="max-autotune")
            self.text_model.bigru_sent = torch.compile(self.text_model.bigru_sent, mode="max-autotune")
            self.align = torch.compile(self.align, mode="max-autotune")
            self.lowrank = torch.compile(self.lowrank, mode="max-autotune")
            self.hdn = torch.compile(self.hdn, mode="max-autotune")
        except Exception:
            pass

    def forward(self,
                texts: torch.Tensor,
                texts_mask: torch.Tensor,
                imgs: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                roi_vec: Optional[torch.Tensor] = None,
                token_embeds: Optional[torch.Tensor] = None,
                token_lengths: Optional[torch.Tensor] = None,
                **kwargs):
        # 文本（兼容预计算/常规两条路径）
        try:
            H_sent, S_doc, mask_sent = self.text_model(
                texts, texts_mask,
                token_embeds=token_embeds, token_lengths=token_lengths
            )
        except TypeError:
            H_sent, S_doc, mask_sent = self.text_model(texts, texts_mask)

        # 图像
        P, r = self.img_model(imgs, roi_vec=roi_vec)
        # 对齐
        c = self.align(H_sent, mask_sent, P)
        # 低秩
        O = self.lowrank(c, r)
        # HDN
        rho4, _ = self.hdn(V=r, O=O, S=S_doc)
        # 分类
        logits = self.cls(self.dropout(rho4))
        pred = torch.argmax(logits, dim=-1)
        loss = self.crit(logits, labels) if labels is not None else torch.tensor(0.0, device=logits.device)
        return pred, loss
