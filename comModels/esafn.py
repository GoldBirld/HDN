# model.py
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 你原有的 FocalLoss 路径（如无可改为相对/本地）
from ..utils.FocalLoss import FocalLoss

# ============== 视觉主干：ResNet-152 用于提取网格特征（ESAFN原文也以ResNet为例） ==============
try:
    import torchvision
    from torchvision.models import resnet152, ResNet152_Weights
except Exception:
    torchvision = None
    resnet152 = None
    ResNet152_Weights = None


# ============== 工具层 ==============
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.1, act="gelu"):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU() if act.lower() == "gelu" else nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc2(self.drop(self.act(self.fc1(x))))
        return x


# ============== 实体敏感注意力（文本/视觉共用） ==============
class EntitySensitiveAttention(nn.Module):
    """
    对序列特征 H 进行实体敏感注意力：
        α_i ∝ v^T tanh(W_h h_i + W_e e + b)
    输入:
        H: (B, L, D) 序列特征
        mask: (B, L) 序列有效位
        e: (B, De) 实体向量（可由实体token平均或lookup）
    输出:
        c: (B, D) 加权和
        attn: (B, L)
    """
    def __init__(self, d_h: int, d_e: int, d_attn: int = 256):
        super().__init__()
        self.W_h = nn.Linear(d_h, d_attn, bias=True)
        self.W_e = nn.Linear(d_e, d_attn, bias=False)
        self.v = nn.Linear(d_attn, 1, bias=False)

    def forward(self, H: torch.Tensor, mask: torch.Tensor, e: torch.Tensor):
        if mask.dtype != torch.bool:
            mask = mask != 0
        # B,L,Dh -> B,L,Da
        s = torch.tanh(self.W_h(H) + self.W_e(e).unsqueeze(1))
        score = self.v(s).squeeze(-1)  # (B,L)
        score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
        attn = torch.softmax(score.float(), dim=-1).to(H.dtype)  # (B,L)
        c = torch.einsum("bl,bld->bd", attn, H)  # (B,Dh)
        return c, attn


# ============== 文本编码与融合（BiLSTM + 实体敏感注意力 + 文本融合层） ==============
class TextEncoderESAFN(nn.Module):
    """
    - 词嵌入 -> BiLSTM -> H
    - 实体向量 e: 对 entity_mask 位置的 token embedding 求平均（或对 H 求平均）
    - 三路文本表征（可选）：Left / Target / Right（为保证兼容，我们默认一体化注意力；如有 L/T/R 切分，可传对应 mask）
    - 文本融合层：将 [c_txt, t_avg, c_txt ⊙ t_avg, |c_txt - t_avg|] 经过 MLP 得到最终文本向量 S
    """
    def __init__(self,
                 vocab_size: int = 21128,
                 embed_dim: int = 300,
                 lstm_hidden: int = 384,
                 dropout: float = 0.1,
                 pad_id: int = 0,
                 pretrained_weight: Optional[torch.Tensor] = None,
                 freeze_embed: bool = False,
                 attn_hidden: int = 256,
                 out_dim: int = 512):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        if pretrained_weight is not None:
            assert isinstance(pretrained_weight, torch.Tensor)
            assert tuple(pretrained_weight.shape) == (vocab_size, embed_dim)
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained_weight)
        if freeze_embed:
            self.embedding.weight.requires_grad = False

        self.emb_drop = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.txt_attn = EntitySensitiveAttention(d_h=2 * lstm_hidden, d_e=embed_dim, d_attn=attn_hidden)
        # 文本融合 MLP：输入拼接维度 = 2h (c_txt) + E (t_avg_in_emb) + 2h (c⊙proj) + 2h (abs diff after proj)
        self.proj_c = nn.Linear(2 * lstm_hidden, out_dim)
        self.proj_t = nn.Linear(embed_dim, out_dim)
        self.fuse_mlp = MLP(in_dim=out_dim * 4, hidden=out_dim, out_dim=out_dim, dropout=dropout)

    def _avg_pool_masked(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # X:(B,L,D), mask:(B,L) -> (B,D)
        if mask.dtype != torch.bool:
            mask = mask != 0
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        return (X * mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self,
                input_ids: torch.Tensor,          # (B,T)
                attention_mask: torch.Tensor,     # (B,T)
                entity_mask: Optional[torch.Tensor] = None  # (B,T) 指示实体token
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        x = self.emb_drop(self.embedding(input_ids))          # (B,T,E)

        # 实体嵌入 e：默认对实体 token 的嵌入平均；若未提供 entity_mask，则退化为整句平均
        if entity_mask is None:
            entity_mask = attention_mask
        e_emb = self._avg_pool_masked(x, entity_mask)         # (B,E)

        # BiLSTM
        lengths = attention_mask.long().sum(dim=1).detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        H_packed, _ = self.bilstm(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(H_packed, batch_first=True, total_length=T)  # (B,T,2h)

        # 实体敏感文本注意力
        c_txt, attn_txt = self.txt_attn(H, attention_mask, e_emb)    # (B,2h), (B,T)

        # 文本融合：将 c_txt 与实体嵌入的平均 t_avg（即 e_emb）进行可交互融合
        c_proj = torch.tanh(self.proj_c(c_txt))            # (B,Do)
        t_proj = torch.tanh(self.proj_t(e_emb))            # (B,Do)
        z = torch.cat([c_proj,
                       t_proj,
                       c_proj * t_proj,
                       torch.abs(c_proj - t_proj)], dim=-1)  # (B, 4*Do)
        S_txt = self.fuse_mlp(z)                           # (B,Do)
        # 返回序列级（用于下游对齐的格式，与旧代码对齐：H_sent(B,1,Do), S_doc(B,Do), mask_sent(B,1)=1）
        return S_txt.unsqueeze(1), S_txt, attention_mask.new_ones((B, 1))


# ============== 视觉编码：ResNet 网格特征 / RoI 特征 + 实体导向注意力 + 视觉门控 ==============
class VisualEncoderESAFN(nn.Module):
    """
    - 若 roi_vec 为 (B,K,D)，直接当作区域特征；否则用 ResNet-152 提取 (h*w,C) 网格特征；
    - 实体导向视觉注意力：与文本 ESA 结构一致；
    - 视觉门控：g = σ( w^T [v_e ; S_txt ; v_e ⊙ S_txt ; |v_e - S_txt|] )，最终 v' = g ⊙ v_e
    """
    def __init__(self,
                 out_dim: int = 512,
                 attn_hidden: int = 256,
                 use_resnet: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.use_resnet = use_resnet

        self.backbone = None
        self.grid_pool = None
        feat_dim = 2048  # ResNet-152 最后一层通道
        if use_resnet:
            if torchvision is None or resnet152 is None:
                raise ImportError("torchvision/resnet152 不可用。")
            net = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1 if ResNet152_Weights else None)
            # 去掉 avgpool 和 fc，保留 C×H×W 特征
            self.backbone = nn.Sequential(*list(net.children())[:-2])
            self.grid_pool = nn.AdaptiveAvgPool2d((7, 7))  # 固定网格，便于 batch 化

        self.es_attn = EntitySensitiveAttention(d_h=feat_dim, d_e=out_dim, d_attn=attn_hidden)
        self.txt_proj_for_vis = nn.Linear(out_dim, out_dim)  # S_txt 投影到与视觉同尺度的查询维
        self.vis_proj = nn.Linear(feat_dim, out_dim)

        gate_in = out_dim * 4
        self.gate = nn.Sequential(
            nn.Linear(gate_in, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, 1),
            nn.Sigmoid()
        )

    def _extract_regions(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (B,3,H,W) -> feats: (B,K,C)  (K = 7*7)
        """
        x = imgs.contiguous(memory_format=torch.channels_last)
        feat = self.backbone(x)          # (B,C,h,w)
        feat = self.grid_pool(feat)      # (B,C,7,7)
        B, C, H, W = feat.shape
        return feat.flatten(2).transpose(1, 2)  # (B,49,C)

    def forward(self,
                imgs: Optional[torch.Tensor],              # (B,3,H,W) 或 None
                S_txt: torch.Tensor,                       # (B,Do) 文本融合后的向量
                roi_vec: Optional[torch.Tensor] = None     # (B,K,D) 或 (B,D) 或 None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 取区域特征
        if roi_vec is not None:
            if roi_vec.dim() == 2:
                R = roi_vec.unsqueeze(1)                   # (B,1,D)
            elif roi_vec.dim() == 3:
                R = roi_vec                                # (B,K,D)
            else:
                raise ValueError("roi_vec 维度应为 (B,D) 或 (B,K,D)")
            feat_dim = R.size(-1)
            if feat_dim != 2048:  # 若外部ROI维度不是2048，可线性映射
                R = F.linear(R, torch.empty(2048, feat_dim, device=R.device).normal_(std=0.02))
        else:
            if imgs is None:
                raise ValueError("未提供 roi_vec 与 imgs，无法得到视觉特征。")
            R = self._extract_regions(imgs)                # (B,K,2048)

        B, K, C = R.shape
        # 将文本向量作查询（投影后作为实体/文本条件）
        e_q = torch.tanh(self.txt_proj_for_vis(S_txt))     # (B,Do)
        # 视觉 ESA 注意力
        v_ctx, attn = self.es_attn(R, R.new_ones((B, K)).bool(), e_q)  # (B,2048), (B,K)
        v_ctx_proj = torch.tanh(self.vis_proj(v_ctx))       # (B,Do)

        # 视觉门控（噪声抑制）
        gate_vec = torch.cat([v_ctx_proj, S_txt, v_ctx_proj * S_txt, torch.abs(v_ctx_proj - S_txt)], dim=-1)
        g = self.gate(gate_vec)                             # (B,1)
        v_final = g * v_ctx_proj                            # (B,Do)
        return v_final, attn


# ============== 文本-视觉双线性交互层（Hadamard 近似双线性） ==============
class BilinearInteraction(nn.Module):
    """
    简洁双线性交互：z = tanh(Wt*S) ⊙ tanh(Wv*V) -> MLP
    """
    def __init__(self, dim_in: int, dim_hidden: int = 512, dim_out: int = 512, dropout: float = 0.1):
        super().__init__()
        self.t_proj = nn.Linear(dim_in, dim_hidden)
        self.v_proj = nn.Linear(dim_in, dim_hidden)
        self.fuse = MLP(in_dim=dim_hidden, hidden=dim_out, out_dim=dim_out, dropout=dropout)

    def forward(self, S_txt: torch.Tensor, V_vis: torch.Tensor) -> torch.Tensor:
        t = torch.tanh(self.t_proj(S_txt))
        v = torch.tanh(self.v_proj(V_vis))
        z = t * v
        return self.fuse(z)  # (B, dim_out)


# ============== 顶层模型（ESAFN） ==============
class Model(nn.Module):
    """
    ESAFN: 文本实体敏感注意力 + 文本融合 → 视觉实体导向注意力 + 门控 → 文本-视觉双线性交互 → 分类
    forward(
        texts: LongTensor (B,T),
        texts_mask: 0/1 (B,T),
        imgs: FloatTensor (B,3,H,W),
        labels: LongTensor (B,) 可选,
        roi_vec: Optional[(B,K,D) or (B,D)] 区域特征可选,
        entity_mask: 0/1 (B,T) 可选
    )
    """
    def __init__(self, config):
        super().__init__()
        # ------- 超参 -------
        vocab_size = getattr(config, "vocab_size", 21128)
        embed_dim = getattr(config, "text_embed_dim", 300)
        lstm_hidden = getattr(config, "text_gru_hidden", 384)
        txt_out = getattr(config, "hdn_hidden", 512)  # 作为各分支对齐维度
        vis_out = txt_out
        bilinear_out = getattr(config, "lowrank_out", 512)
        num_classes = getattr(config, "num_classes", 2)
        dropout = getattr(config, "dropout", 0.1)

        self.text_encoder = TextEncoderESAFN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            lstm_hidden=lstm_hidden,
            dropout=getattr(config, "text_dropout", 0.1),
            pad_id=getattr(config, "pad_id", 0),
            pretrained_weight=getattr(config, "pretrained_wordvec", None),
            freeze_embed=getattr(config, "freeze_word_embed", False),
            attn_hidden=getattr(config, "attn_hidden", 256),
            out_dim=txt_out
        )

        self.visual_encoder = VisualEncoderESAFN(
            out_dim=vis_out,
            attn_hidden=getattr(config, "vis_attn_hidden", 256),
            use_resnet=getattr(config, "use_resnet_backbone", True)
        )

        self.inter_bilinear = BilinearInteraction(
            dim_in=txt_out,
            dim_hidden=getattr(config, "bilinear_hidden", bilinear_out),
            dim_out=bilinear_out,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bilinear_out, num_classes)
        )

        # 损失
        loss_name = str(getattr(config, "loss", "ce")).lower()
        if loss_name in ("focal", "focalloss"):
            self.crit = FocalLoss(alpha=getattr(config, "focal_alpha", None),
                                  gamma=float(getattr(config, "focal_gamma", 2.0)),
                                  class_weight=getattr(config, "class_weight", None),
                                  reduction="mean")
        else:
            weight_cfg = getattr(config, "class_weight", None)
            self.crit = nn.CrossEntropyLoss(
                weight=(torch.tensor(weight_cfg, dtype=torch.float32) if weight_cfg is not None else None),
                label_smoothing=float(getattr(config, "label_smoothing", 0.0))
            )

        # 可选：编译加速（与原工程保持风格）
        if getattr(config, "compile_submodules", False):
            try:
                self.text_encoder.bilstm = torch.compile(self.text_encoder.bilstm, mode="max-autotune")
                self.visual_encoder.es_attn = torch.compile(self.visual_encoder.es_attn, mode="max-autotune")
                self.inter_bilinear = torch.compile(self.inter_bilinear, mode="max-autotune")
            except Exception:
                pass

    def forward(self,
                texts: torch.Tensor,              # (B,T)
                texts_mask: torch.Tensor,         # (B,T)
                imgs: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                roi_vec: Optional[torch.Tensor] = None,
                entity_mask: Optional[torch.Tensor] = None,
                **kwargs):
        # --- 文本：实体敏感注意力 + 文本融合 ---
        H_sent, S_txt, mask_sent = self.text_encoder(texts, texts_mask, entity_mask=entity_mask)  # S_txt:(B,Do)

        # --- 视觉：实体导向注意力 + 门控 ---
        V_vis, _ = self.visual_encoder(imgs, S_txt, roi_vec=roi_vec)  # (B,Do)

        # --- 模态间双线性交互 ---
        Z = self.inter_bilinear(S_txt, V_vis)  # (B,Bo)

        # --- 分类 ---
        logits = self.classifier(Z)            # (B,C)
        pred = torch.argmax(logits, dim=-1)
        loss = self.crit(logits, labels) if labels is not None else torch.tensor(0.0, device=logits.device)
        return pred, loss
