# model.py
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 可切换到你的路径
from ..utils.FocalLoss import FocalLoss

# ===== torchvision backbone for visual features =====
try:
    import torchvision
    from torchvision.models import resnet50, resnet152
    from torchvision.models import ResNet50_Weights, ResNet152_Weights
except Exception:
    torchvision = None
    resnet50 = resnet152 = None
    ResNet50_Weights = ResNet152_Weights = None


# -----------------------------
# Utils
# -----------------------------
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x:(B,L,D), mask:(B,L)->(B,D)"""
    if mask.dtype != torch.bool:
        mask = mask != 0
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


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
# 基础：实体/查询条件的“加性注意” (Bahdanau style)
# score_i = v^T tanh(Wk * k_i + Wq * q)
# -----------------------------
class AdditiveAttention(nn.Module):
    def __init__(self, d_k: int, d_q: int, d_attn: int = 256):
        super().__init__()
        self.Wk = nn.Linear(d_k, d_attn, bias=True)
        self.Wq = nn.Linear(d_q, d_attn, bias=False)
        self.v = nn.Linear(d_attn, 1, bias=False)

    def forward(self, K: torch.Tensor, q: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        K: (B,L,Dk), q: (B,Dq), mask:(B,L) optional
        return: ctx:(B,Dk), attn:(B,L)
        """
        B, L, Dk = K.shape
        s = torch.tanh(self.Wk(K) + self.Wq(q).unsqueeze(1))  # (B,L,Da)
        score = self.v(s).squeeze(-1)                         # (B,L)
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask != 0
            score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
        attn = torch.softmax(score.float(), dim=-1).to(K.dtype)  # (B,L)
        ctx = torch.einsum("bl,bld->bd", attn, K)                # (B,Dk)
        return ctx, attn


# -----------------------------
# 文本特征映射（Embedding -> BiLSTM）
# -----------------------------
class TextMapper(nn.Module):
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
        self.encoder = nn.LSTM(embed_dim, lstm_hidden, batch_first=True, bidirectional=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        B, T = input_ids.shape
        x = self.emb_drop(self.embedding(input_ids))  # (B,T,E)
        lengths = attention_mask.long().sum(dim=1).detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        H_pack, _ = self.encoder(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(H_pack, batch_first=True, total_length=T)  # (B,T,2h)
        return H  # word-level hidden states


# -----------------------------
# 视觉特征映射（Object-View / Scene-View）
# -----------------------------
class VisualMapper(nn.Module):
    """
    - Object-View: 若 roi_vec 提供 (B,K,D)，直接映射；否则从 ResNet 的 7x7 网格特征作为 K=49 区域
    - Scene-View : ResNet 全局池化得到 (B,C) 单向量
    """
    def __init__(self, backbone: str = "resnet50", out_dim: int = 512, grid: int = 7):
        super().__init__()
        assert backbone in ("resnet50", "resnet152")
        if torchvision is None:
            raise ImportError("torchvision is required for VisualMapper.")
        if backbone == "resnet50":
            net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if ResNet50_Weights else None)
            feat_dim = 2048
        else:
            net = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1 if ResNet152_Weights else None)
            feat_dim = 2048

        # 去掉 avgpool/fc
        self.cnn = nn.Sequential(*list(net.children())[:-2])
        self.grid_pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.feat_dim = feat_dim
        self.out_dim = out_dim

        # 线性映射到统一维度
        self.obj_proj = nn.Linear(feat_dim, out_dim)
        self.scn_proj = nn.Linear(feat_dim, out_dim)

    def forward(self, imgs: Optional[torch.Tensor], roi_vec: Optional[torch.Tensor] = None):
        """
        return:
          O: (B,K,Do)   object view (K>=1)
          S: (B,Do)     scene view
          mask_O: (B,K) object mask (all True here)
        """
        if roi_vec is not None:
            if roi_vec.dim() == 2:
                roi_vec = roi_vec.unsqueeze(1)  # (B,1,D)
            # 将 roi 映射到 out_dim
            if roi_vec.size(-1) != self.out_dim:
                O = self.obj_proj(roi_vec)  # (B,K,Do) 允许输入 D!=feat_dim
            else:
                O = roi_vec
            # 需要 scene 向量；若无图像，退化为 obj 平均
            if imgs is not None:
                with torch.no_grad():
                    x = imgs.contiguous(memory_format=torch.channels_last)
                    feat = self.cnn(x)             # (B,C,H,W)
                    g = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B,C)
            else:
                g = O.mean(dim=1)  # (B,Do)（已是 out_dim）
                return O, g, O.new_ones(O.size()[:2], dtype=torch.bool)
            S = self.scn_proj(g)  # (B,Do)
            mask_O = O.new_ones(O.size()[:2], dtype=torch.bool)
            return O, S, mask_O

        # 无 roi 时，从 CNN 提取 7x7 网格
        x = imgs.contiguous(memory_format=torch.channels_last)
        feat = self.cnn(x)                            # (B,C,h,w)
        grid = self.grid_pool(feat)                   # (B,C,G,G)
        B, C, G, _ = grid.shape
        O = grid.flatten(2).transpose(1, 2)           # (B,K=CELL=G*G,C)
        O = self.obj_proj(O)                          # (B,K,Do)
        S = self.scn_proj(F.adaptive_avg_pool2d(feat, 1).flatten(1))  # (B,Do)
        mask_O = O.new_ones((B, G * G), dtype=torch.bool)
        return O, S, mask_O


# -----------------------------
# 交互学习模块（多跳记忆：跨视图注意 + GRUCell 更新）
#   Memory state: Mt (text), Mo (object), Ms (scene)
#   每跳：
#     q_t  = mean(H_t)                # 文本查询
#     o_ctx = Attn(O, q_t), s_ctx = Attn(S_as_seq, q_t)
#     t_from_o = Attn(H, o_ctx), t_from_s = Attn(H, s_ctx)
#     Mt = GRU([t_from_o || t_from_s], Mt)
#     Mo = GRU(o_ctx, Mo)
#     Ms = GRU(s_ctx, Ms)

class MultiViewMemory(nn.Module):
    def __init__(self, d_text: int, d_view: int, d_attn: int = 256, hops: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hops = hops
        self.attn_o = AdditiveAttention(d_k=d_view, d_q=d_text, d_attn=d_attn)
        self.attn_s = AdditiveAttention(d_k=d_view, d_q=d_text, d_attn=d_attn)
        self.attn_t = AdditiveAttention(d_k=d_text, d_q=d_view, d_attn=d_attn)
        self.gru_t = nn.GRUCell(2 * d_text, d_text)
        self.gru_o = nn.GRUCell(d_view, d_view)
        self.gru_s = nn.GRUCell(d_view, d_view)
        self.drop = nn.Dropout(dropout)
        # 将 scene 向量扩展成长度为 1 的“序列”
        self.register_buffer("_ones", torch.ones(1, 1, dtype=torch.bool))

    def forward(self,
                H_txt: torch.Tensor,           # (B,T,Dt)
                mask_txt: torch.Tensor,        # (B,T)
                O_obj: torch.Tensor,           # (B,K,Dv)
                mask_obj: torch.Tensor,        # (B,K)
                S_scn: torch.Tensor            # (B,Dv)
                ):
        B, T, Dt = H_txt.shape
        _, K, Dv = O_obj.shape

        Mt = masked_mean(H_txt, mask_txt)     # (B,Dt) init
        Mo = O_obj.mean(dim=1)                # (B,Dv)  init
        Ms = S_scn                             # (B,Dv)

        for _ in range(self.hops):
            q_t = Mt                          # (B,Dt)

            # text -> object/scene
            o_ctx, _ = self.attn_o(O_obj, q_t, mask_obj)              # (B,Dv)
            s_ctx, _ = self.attn_s(S_scn.unsqueeze(1), q_t, self._ones.expand(B, 1))  # (B,Dv)

            # object/scene -> text
            t_from_o, _ = self.attn_t(H_txt, o_ctx, mask_txt)         # (B,Dt)
            t_from_s, _ = self.attn_t(H_txt, s_ctx, mask_txt)         # (B,Dt)

            # 更新记忆
            Mt = self.gru_t(self.drop(torch.cat([t_from_o, t_from_s], dim=-1)), Mt)
            Mo = self.gru_o(self.drop(o_ctx), Mo)
            Ms = self.gru_s(self.drop(s_ctx), Ms)

        return Mt, Mo, Ms  # 更新后的三视图记忆


# -----------------------------
# 多视图融合（拼接 + 互作项 -> MLP）
# -----------------------------
class MultiViewFusion(nn.Module):
    def __init__(self, dim_t: int, dim_v: int, out_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        # 对齐到相同维度
        self.t_proj = nn.Linear(dim_t, out_dim)
        self.o_proj = nn.Linear(dim_v, out_dim)
        self.s_proj = nn.Linear(dim_v, out_dim)
        fuse_in = out_dim * 9  # [T,O,S, T*O,T*S,O*S, |T-O|,|T-S|,|O-S|]
        self.fuse = MLP(fuse_in, hidden=out_dim, out_dim=out_dim, dropout=dropout)

    def forward(self, Mt: torch.Tensor, Mo: torch.Tensor, Ms: torch.Tensor):
        t = torch.tanh(self.t_proj(Mt))
        o = torch.tanh(self.o_proj(Mo))
        s = torch.tanh(self.s_proj(Ms))
        feats = torch.cat([
            t, o, s,
            t * o, t * s, o * s,
            torch.abs(t - o), torch.abs(t - s), torch.abs(o - s)
        ], dim=-1)
        return self.fuse(feats)  # (B,out_dim)


# -----------------------------
# 顶层：MVAN
# -----------------------------
class Model(nn.Module):
    """
    Multi-View Attentional Network (Yang et al., TMM'20)
    forward(
        texts: LongTensor (B,T),
        texts_mask: 0/1 (B,T),
        imgs: FloatTensor (B,3,H,W) or None,
        labels: LongTensor (B,) optional,
        roi_vec: Optional[(B,K,D)] object-view features
    )
    """
    def __init__(self, config):
        super().__init__()
        # ---- config & dims ----
        vocab_size = getattr(config, "vocab_size", 21128)
        embed_dim = getattr(config, "text_embed_dim", 300)
        lstm_hidden = getattr(config, "text_gru_hidden", 384)
        Dt = 2 * lstm_hidden

        view_dim = getattr(config, "view_dim", 512)       # 视觉映射输出维
        hops = getattr(config, "memory_hops", 3)
        attn_hidden = getattr(config, "attn_hidden", 256)
        fuse_out = getattr(config, "lowrank_out", 512)     # 作为融合输出维（复用你原命名）
        dropout = getattr(config, "dropout", 0.1)
        num_classes = getattr(config, "num_classes", 2)

        # ---- Stage 1: Feature Mapping ----
        self.text_mapper = TextMapper(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            lstm_hidden=lstm_hidden,
            pad_id=getattr(config, "pad_id", 0),
            dropout=getattr(config, "text_dropout", 0.1),
            pretrained_weight=getattr(config, "pretrained_wordvec", None),
            freeze_embed=getattr(config, "freeze_word_embed", False),
        )
        self.vis_mapper = VisualMapper(
            backbone=getattr(config, "image_backbone", "resnet50"),
            out_dim=view_dim,
            grid=getattr(config, "grid_size", 7),
        )

        # ---- Stage 2: Interactive Learning (Memory hops) ----
        self.mv_memory = MultiViewMemory(
            d_text=Dt, d_view=view_dim, d_attn=attn_hidden, hops=hops, dropout=dropout
        )

        # ---- Stage 3: Multi-View Fusion ----
        self.fusion = MultiViewFusion(dim_t=Dt, dim_v=view_dim, out_dim=fuse_out, dropout=dropout)

        # ---- Classifier & Loss ----
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(fuse_out, num_classes))

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
                self.text_mapper.encoder = torch.compile(self.text_mapper.encoder, mode="max-autotune")
                self.mv_memory = torch.compile(self.mv_memory, mode="max-autotune")
                self.fusion = torch.compile(self.fusion, mode="max-autotune")
            except Exception:
                pass

    def forward(self,
                texts: torch.Tensor,
                texts_mask: torch.Tensor,
                imgs: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                roi_vec: Optional[torch.Tensor] = None,
                **kwargs):
        # ---- Stage 1: Feature Mapping ----
        H_txt = self.text_mapper(texts, texts_mask)                 # (B,T,Dt)
        O_obj, S_scn, mask_O = self.vis_mapper(imgs, roi_vec=roi_vec)  # (B,K,Dv),(B,Dv),(B,K)

        # ---- Stage 2: Interactive Learning (Memory hops) ----
        Mt, Mo, Ms = self.mv_memory(H_txt, texts_mask, O_obj, mask_O, S_scn)  # (B,Dt),(B,Dv),(B,Dv)

        # ---- Stage 3: Multi-View Fusion ----
        Z = self.fusion(Mt, Mo, Ms)                                 # (B,F)

        # ---- Classification ----
        logits = self.cls(Z)                                        # (B,C)
        pred = torch.argmax(logits, dim=-1)
        loss = self.crit(logits, labels) if labels is not None else torch.tensor(0.0, device=logits.device)
        return pred, loss
