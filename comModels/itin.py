# model.py
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# your FocalLoss path (保持与现工程一致)
from ..utils.FocalLoss import FocalLoss

# ===== torchvision (仅在无 roi_vec 时用作回退) =====
try:
    import torchvision
    from torchvision.models import resnet50, resnet152
    from torchvision.models import ResNet50_Weights, ResNet152_Weights
except Exception:
    torchvision = None
    resnet50 = resnet152 = None
    ResNet50_Weights = ResNet152_Weights = None


# -----------------------------
# 小工具
# -----------------------------
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x:(B,L,D), mask:(B,L)->(B,D)"""
    if mask.dtype != torch.bool:
        mask = mask != 0
    den = mask.sum(dim=1, keepdim=True).clamp_min(1)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / den


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.1, act="gelu"):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU() if act.lower() == "gelu" else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


# -----------------------------
# 文本编码：Embedding -> BiLSTM （词级上下文）
# -----------------------------
class TextEncoder(nn.Module):
    def __init__(self,
                 vocab_size: int = 21128,
                 embed_dim: int = 300,
                 lstm_hidden: int = 384,
                 pad_id: int = 0,
                 dropout: float = 0.1,
                 pretrained_weight: Optional[torch.Tensor] = None,
                 freeze_embed: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        if pretrained_weight is not None:
            assert isinstance(pretrained_weight, torch.Tensor)
            assert tuple(pretrained_weight.shape) == (vocab_size, embed_dim)
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained_weight)
        if freeze_embed:
            self.embedding.weight.requires_grad = False
        self.emb_drop = nn.Dropout(dropout)
        self.encoder = nn.LSTM(embed_dim, lstm_hidden, bidirectional=True, batch_first=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        B, T = input_ids.shape
        x = self.emb_drop(self.embedding(input_ids))  # (B,T,E)
        lengths = attention_mask.long().sum(dim=1).detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        H_pack, _ = self.encoder(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(H_pack, batch_first=True, total_length=T)  # (B,T,2h)
        return H  # 词级隐表示 (B,T,Dt=2h)


# -----------------------------
# 视觉编码：优先使用 ROI 特征；否则 CNN 网格回退
# 返回：
#   R: (B,K,Dv)   区域特征（Object View）
#   Vg: (B,Dv)    全局特征（Scene View / Visual Context）
#   mask_R: (B,K) 区域有效位
# -----------------------------
class VisualEncoder(nn.Module):
    def __init__(self, out_dim: int = 512, backbone: str = "resnet50", grid: int = 7):
        super().__init__()
        self.out_dim = out_dim
        self.grid = grid
        self.backbone = backbone.lower()
        self.has_cnn = torchvision is not None and self.backbone in ("resnet50", "resnet152")
        if self.has_cnn:
            if self.backbone == "resnet50":
                net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if ResNet50_Weights else None)
                feat_dim = 2048
            else:
                net = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1 if ResNet152_Weights else None)
                feat_dim = 2048
            self.cnn = nn.Sequential(*list(net.children())[:-2])
            self.obj_proj = nn.Linear(feat_dim, out_dim)
            self.scn_proj = nn.Linear(feat_dim, out_dim)
            self.grid_pool = nn.AdaptiveAvgPool2d((grid, grid))
        else:
            # 当只用 ROI 时，投影层在 ITIN 模块里处理
            self.cnn = None
            self.obj_proj = None
            self.scn_proj = None

    def forward(self, imgs: Optional[torch.Tensor], roi_vec: Optional[torch.Tensor]):
        if roi_vec is not None:
            if roi_vec.dim() == 2:
                roi_vec = roi_vec.unsqueeze(1)  # (B,1,D)
            B, K, Din = roi_vec.shape
            # 将 ROI 投影到统一维度
            roi_proj = nn.Linear(Din, self.out_dim).to(roi_vec.device)
            R = roi_proj(roi_vec)                      # (B,K,Dv)
            mask_R = R.new_ones((B, K), dtype=torch.bool)
            # 全局：若有图片则从 CNN 得；否则用 ROI 均值近似
            if imgs is not None and self.has_cnn:
                feat = self.cnn(imgs.contiguous(memory_format=torch.channels_last))
                Vg = self.scn_proj(F.adaptive_avg_pool2d(feat, 1).flatten(1))  # (B,Dv)
            else:
                Vg = R.mean(dim=1)
            return R, Vg, mask_R

        # 无 ROI，使用 CNN 网格特征作为“伪 ROI”
        assert imgs is not None and self.has_cnn, "需要 roi_vec 或可用的 CNN backbone。"
        feat = self.cnn(imgs.contiguous(memory_format=torch.channels_last))   # (B,C,h,w)
        grid = self.grid_pool(feat)                                           # (B,C,G,G)
        B, C, G, _ = grid.shape
        R = grid.flatten(2).transpose(1, 2)                                   # (B,K=G*G,C)
        R = self.obj_proj(R)                                                  # (B,K,Dv)
        Vg = self.scn_proj(F.adaptive_avg_pool2d(feat, 1).flatten(1))         # (B,Dv)
        mask_R = R.new_ones((B, G * G), dtype=torch.bool)
        return R, Vg, mask_R


# -----------------------------
# Cross-Modal Alignment (CMA)
#   以“区域 r_i 为查询”在词序列 H 上做加性注意，得到对齐 t_i
#   矢量化实现：对所有 K 个区域并行计算
# -----------------------------
class CrossModalAlignment(nn.Module):
    def __init__(self, d_text: int, d_vis: int, d_attn: int = 256):
        super().__init__()
        self.Wt = nn.Linear(d_text, d_attn, bias=True)   # for words
        self.Wr = nn.Linear(d_vis,  d_attn, bias=False)  # for regions
        self.v  = nn.Linear(d_attn, 1, bias=False)

    def forward(self, H: torch.Tensor, mask_t: torch.Tensor, R: torch.Tensor, mask_r: torch.Tensor):
        """
        H:(B,T,Dt), mask_t:(B,T), R:(B,K,Dv), mask_r:(B,K)
        return:
          T_aligned:(B,K,Dt)  每个区域对应的词向量
          A:(B,K,T)           区域->词 注意力
        """
        if mask_t.dtype != torch.bool:
            mask_t = mask_t != 0
        B, T, Dt = H.shape
        B, K, Dv = R.shape

        H_ = self.Wt(H)                         # (B,T,Da)
        R_ = self.Wr(R)                         # (B,K,Da)
        # score[b,k,t] = v^T tanh(H_[b,t] + R_[b,k])
        s = torch.tanh(H_.unsqueeze(1) + R_.unsqueeze(2))   # (B,K,T,Da)
        score = self.v(s).squeeze(-1)                       # (B,K,T)
        score = score.masked_fill(~mask_t.unsqueeze(1), torch.finfo(score.dtype).min)
        A = torch.softmax(score.float(), dim=-1).to(H.dtype)  # (B,K,T)
        T_aligned = torch.einsum("bkt,btd->bkd", A, H)        # (B,K,Dt)

        # 遮罩无效区域
        if mask_r is not None:
            A = A * mask_r.unsqueeze(-1).to(A.dtype)
        return T_aligned, A


# -----------------------------
# Cross-Modal Gating (CMG)
#   自适应门控融合：g = sigmoid(MLP([r, t, r*t, |r-t|]))
#   u = g ⊙ r' + (1-g) ⊙ t'  （r', t' 为到同维度的投影）
# -----------------------------
class CrossModalGating(nn.Module):
    def __init__(self, d_text: int, d_vis: int, d_out: int = 512, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.r_proj = nn.Linear(d_vis,  d_out)
        self.t_proj = nn.Linear(d_text, d_out)
        self.gate = MLP(in_dim=d_out * 4, hidden=hidden, out_dim=d_out, dropout=dropout)  # 逐维门
        self.sigmoid = nn.Sigmoid()

    def forward(self, R: torch.Tensor, T_aligned: torch.Tensor):
        """
        R:(B,K,Dv), T_aligned:(B,K,Dt)
        return U:(B,K,Do)
        """
        r = self.r_proj(R)               # (B,K,Do)
        t = self.t_proj(T_aligned)       # (B,K,Do)
        z = torch.cat([r, t, r * t, torch.abs(r - t)], dim=-1)  # (B,K,4*Do)
        g = self.sigmoid(self.gate(z))   # (B,K,Do) 逐维门
        U = g * r + (1.0 - g) * t        # (B,K,Do)
        return U


# -----------------------------
# 交互特征聚合（区域注意汇聚）
#   β = softmax( v^T tanh(Wu * u_k + Wc * c_text) )
#   f_inter = Σ β_k u_k
# -----------------------------
class InteractionAggregator(nn.Module):
    def __init__(self, d_u: int, d_ctx: int, d_attn: int = 256):
        super().__init__()
        self.Wu = nn.Linear(d_u,   d_attn, bias=True)
        self.Wc = nn.Linear(d_ctx, d_attn, bias=False)
        self.v  = nn.Linear(d_attn, 1,    bias=False)

    def forward(self, U: torch.Tensor, ctx_text: torch.Tensor, mask_r: Optional[torch.Tensor] = None):
        """
        U:(B,K,Du)  ctx_text:(B,Dt)  mask_r:(B,K)
        return f_inter:(B,Du)
        """
        s = torch.tanh(self.Wu(U) + self.Wc(ctx_text).unsqueeze(1))  # (B,K,Da)
        score = self.v(s).squeeze(-1)                                # (B,K)
        if mask_r is not None:
            if mask_r.dtype != torch.bool:
                mask_r = mask_r != 0
            score = score.masked_fill(~mask_r, torch.finfo(score.dtype).min)
        beta = torch.softmax(score.float(), dim=-1).to(U.dtype)      # (B,K)
        f_inter = torch.einsum("bk,bkd->bd", beta, U)                # (B,Du)
        return f_inter


# -----------------------------
# ITIN 顶层
# -----------------------------
class Model(nn.Module):
    """
    ITIN (Zhu et al., IEEE TMM 2022)
    forward(
        texts: LongTensor (B,T),
        texts_mask: 0/1 (B,T),
        imgs: FloatTensor (B,3,H,W) or None,
        labels: LongTensor (B,) optional,
        roi_vec: Optional[(B,K,D)] image region features
    )
    """
    def __init__(self, config):
        super().__init__()
        # ---- 维度配置 ----
        vocab_size   = getattr(config, "vocab_size", 21128)
        embed_dim    = getattr(config, "text_embed_dim", 300)
        lstm_hidden  = getattr(config, "text_gru_hidden", 384)
        Dt           = 2 * lstm_hidden

        view_dim     = getattr(config, "view_dim", 512)   # 统一视觉/交互维度
        attn_hidden  = getattr(config, "attn_hidden", 256)
        grid_size    = getattr(config, "grid_size", 7)
        dropout      = getattr(config, "dropout", 0.1)
        num_classes  = getattr(config, "num_classes", 2)

        # ---- Stage 1: 文本/视觉编码 + 单模态上下文 ----
        self.txt_enc = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            lstm_hidden=lstm_hidden,
            pad_id=getattr(config, "pad_id", 0),
            dropout=getattr(config, "text_dropout", 0.1),
            pretrained_weight=getattr(config, "pretrained_wordvec", None),
            freeze_embed=getattr(config, "freeze_word_embed", False),
        )
        self.vis_enc = VisualEncoder(
            out_dim=view_dim,
            backbone=getattr(config, "image_backbone", "resnet50"),
            grid=grid_size
        )

        # ---- Stage 2: 跨模态对齐 + 门控 ----
        self.cma = CrossModalAlignment(d_text=Dt, d_vis=view_dim, d_attn=attn_hidden)
        self.cmg = CrossModalGating(d_text=Dt, d_vis=view_dim, d_out=view_dim,
                                    hidden=getattr(config, "cmg_hidden", 512), dropout=dropout)

        # 交互聚合（以文本上下文为查询）
        self.inter_aggr = InteractionAggregator(d_u=view_dim, d_ctx=Dt, d_attn=attn_hidden)

        # 单模态上下文投影到统一维度
        self.txt_ctx_proj = nn.Linear(Dt, view_dim)
        # 视觉上下文已在 VisualEncoder 输出为 view_dim

        # ---- Stage 3: 融合与分类 ----
        fuse_in = view_dim * 3  # [f_inter, v_ctx, t_ctx]
        self.fuse = MLP(in_dim=fuse_in,
                        hidden=getattr(config, "fusion_hidden", 512),
                        out_dim=getattr(config, "fusion_out", 512),
                        dropout=dropout)
        self.cls = nn.Sequential(nn.Dropout(dropout),
                                 nn.Linear(getattr(config, "fusion_out", 512), num_classes))

        # ---- 损失 ----
        loss_name = str(getattr(config, "loss", "ce")).lower()
        if loss_name in ("focal", "focalloss"):
            self.crit = FocalLoss(
                alpha=getattr(config, "focal_alpha", None),
                gamma=float(getattr(config, "focal_gamma", 2.0)),
                class_weight=getattr(config, "class_weight", None),
                reduction="mean",
            )
        else:
            weight_cfg = getattr(config, "class_weight", None)
            self.crit = nn.CrossEntropyLoss(
                weight=(torch.tensor(weight_cfg, dtype=torch.float32) if weight_cfg is not None else None),
                label_smoothing=float(getattr(config, "label_smoothing", 0.0)),
            )

        # 可选：torch.compile
        if getattr(config, "compile_submodules", False):
            try:
                self.txt_enc.encoder = torch.compile(self.txt_enc.encoder, mode="max-autotune")
                self.cma = torch.compile(self.cma, mode="max-autotune")
                self.cmg = torch.compile(self.cmg, mode="max-autotune")
                self.inter_aggr = torch.compile(self.inter_aggr, mode="max-autotune")
                self.fuse = torch.compile(self.fuse, mode="max-autotune")
            except Exception:
                pass

    def forward(self,
                texts: torch.Tensor,
                texts_mask: torch.Tensor,
                imgs: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                roi_vec: Optional[torch.Tensor] = None,
                **kwargs):

        # ----- 文本侧：词级隐状态 + 文本上下文 -----
        H_txt = self.txt_enc(texts, texts_mask)          # (B,T,Dt)
        ctx_txt = masked_mean(H_txt, texts_mask)         # (B,Dt)

        # ----- 视觉侧：区域集合 + 全局上下文 -----
        R, Vg, mask_R = self.vis_enc(imgs, roi_vec=roi_vec)   # R:(B,K,Dv=view_dim), Vg:(B,view_dim)

        # ----- 跨模态对齐（区域->词） -----
        T_aligned, _ = self.cma(H_txt, texts_mask, R, mask_R) # (B,K,Dt)

        # ----- 跨模态门控融合 -----
        U = self.cmg(R, T_aligned)                            # (B,K,view_dim)

        # ----- 区域注意聚合（以文本上下文为查询） -----
        f_inter = self.inter_aggr(U, ctx_txt, mask_R)         # (B,view_dim)

        # ----- 单模态上下文投影 -----
        t_ctx = torch.tanh(self.txt_ctx_proj(ctx_txt))        # (B,view_dim)
        v_ctx = Vg                                            # (B,view_dim)

        # ----- 融合 + 分类 -----
        Z = torch.cat([f_inter, v_ctx, t_ctx], dim=-1)        # (B,3*view_dim)
        Z = self.fuse(Z)                                      # (B,fusion_out)
        logits = self.cls(Z)                                  # (B,num_classes)
        pred = torch.argmax(logits, dim=-1)
        loss = self.crit(logits, labels) if labels is not None else torch.tensor(0.0, device=logits.device)
        return pred, loss
