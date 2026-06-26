# model.py
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.FocalLoss import FocalLoss

# ========= transformers (仅为兼容 CLIP 视觉分支) =========
try:
    from transformers import AutoModel, AutoConfig, CLIPTextModel, CLIPModel
except Exception:
    AutoModel = AutoConfig = CLIPTextModel = CLIPModel = None

# ========= timm（RepVGG 来自 timm）=========
try:
    import timm  # pip install timm
except Exception:
    timm = None


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
        score = torch.tanh(self.proj(H))      # (B,T,H)
        score = self.v(score).squeeze(-1)     # (B,T)
        if mask.dtype != torch.bool:
            mask = mask != 0
        score_fp32 = score.float()
        score_fp32 = score_fp32.masked_fill(~mask, torch.finfo(score_fp32.dtype).min)
        a = torch.softmax(score_fp32, dim=-1).to(H.dtype)  # (B,T)
        ui = torch.einsum("bt,btd->bd", a, H)              # (B,D)
        return ui


class TextModel(nn.Module):
    """
    返回:
        H_sent   : (B,1,2*gru_hidden)
        S_doc    : (B,2*gru_hidden)
        mask_sent: (B,1)  # 句级 mask，这里恒为 1
    """
    def __init__(self,
                 backbone: str = "bert-base-chinese",
                 bert_dim: int = 768,
                 gru_hidden: int = 384,
                 attn_hidden: int = 512):
        super().__init__()
        self.bert_dim = bert_dim

        # ---- 判断是否使用 CLIP 文本编码器 ----
        use_clip = False
        name_l = backbone.lower()
        if name_l.startswith("openai:"):
            use_clip = True
            clip_name = backbone.split(":", 1)[1]
        else:
            if AutoConfig is None:
                clip_name = backbone
            else:
                cfg = AutoConfig.from_pretrained(backbone)
                use_clip = getattr(cfg, "model_type", "") == "clip"
                clip_name = backbone

        # ---- 加载编码器 ----
        if use_clip:
            if CLIPTextModel is None:
                raise ImportError("transformers.CLIPTextModel 不可用，但你配置了 CLIP 文本骨干。")
            self.encoder = CLIPTextModel.from_pretrained(clip_name)
        else:
            if AutoModel is None:
                raise ImportError("transformers.AutoModel 不可用，无法加载 BERT 文本骨干。")
            print('use bert')
            self.encoder = AutoModel.from_pretrained(backbone)

        # 优化设置
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if hasattr(self.encoder, "config"):
            try:
                self.encoder.config.use_cache = False
            except Exception:
                pass

        # 记录 hidden_size
        self._enc_dim = getattr(getattr(self.encoder, "config", None), "hidden_size", bert_dim)

        # 适配到 bert_dim
        self.in_proj = nn.Linear(self._enc_dim, bert_dim) if self._enc_dim != bert_dim else nn.Identity()

        # GRU + 注意力
        self.bigru_word = nn.GRU(bert_dim, gru_hidden, bidirectional=True, batch_first=True)
        self.word_att = WordAttention(2 * gru_hidden, attn_hidden)
        self.bigru_sent = nn.GRU(2 * gru_hidden, gru_hidden, bidirectional=True, batch_first=True)

        self._use_clip_text = use_clip

    # ===== 预计算路径 =====
    def forward_from_embeds(self,
                            token_embeds: torch.Tensor,       # (B,T,enc_dim)
                            attention_mask: torch.Tensor,     # (B,T) → bool / 0/1
                            lengths: Optional[torch.Tensor]=None  # (B,) 可选
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, enc_dim = token_embeds.shape

        # 若需要，动态建立 in_proj（注意：需在构建优化器前确定）
        if isinstance(self.in_proj, nn.Identity):
            if enc_dim != self.bert_dim:
                self.in_proj = nn.Linear(enc_dim, self.bert_dim).to(token_embeds.device)
        else:
            if self.in_proj.in_features != enc_dim:
                raise ValueError(f"[TextModel] 预计算 embeddings 的维度({enc_dim}) 与 in_proj.in_features({self.in_proj.in_features}) 不一致。")

        X = self.in_proj(token_embeds)  # (B,T,bert_dim)

        if lengths is None:
            lengths = attention_mask.sum(dim=1).detach().cpu()
        else:
            lengths = lengths.detach().cpu()

        packed = nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
        H_word_packed, _ = self.bigru_word(packed)
        H_word, _ = nn.utils.rnn.pad_packed_sequence(H_word_packed, batch_first=True, total_length=T)  # (B,T,2h)

        ui = self.word_att(H_word, attention_mask)     # (B,2h)
        Hi, _ = self.bigru_sent(ui.unsqueeze(1))       # (B,1,2h)

        mask_sent = attention_mask.new_ones((B, 1))
        return Hi, ui, mask_sent

    # ===== 常规路径 =====
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_embeds: torch.Tensor = None,     # 兼容 Trainer 的可选参数
        token_lengths: torch.Tensor = None,    # 兼容 Trainer 的可选参数
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # 直通预计算向量
        if token_embeds is not None:
            X = self.in_proj(token_embeds)  # (B,T,bert_dim)
            B, T, _ = X.shape
            if token_lengths is None:
                with torch.no_grad():
                    token_lengths = (token_embeds.abs().sum(dim=-1) > 0).long().sum(dim=1)  # (B,)

            device = X.device
            arange = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # (B,T)
            attn_mask = (arange < token_lengths.unsqueeze(1)).to(X.dtype)
            attention_mask = attn_mask

            lengths = token_lengths.detach().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
            H_word_packed, _ = self.bigru_word(packed)
            H_word, _ = nn.utils.rnn.pad_packed_sequence(H_word_packed, batch_first=True, total_length=T)  # (B,T,2h)

            ui = self.word_att(H_word, attention_mask)         # (B,2h)
            Hi, _ = self.bigru_sent(ui.unsqueeze(1))           # (B,1,2h)

            H_sent = Hi
            S_doc = ui
            mask_sent = attention_mask.new_ones((B, 1))
            return H_sent, S_doc, mask_sent

        # 正常 Transformer 路径
        B, T = input_ids.size(0), input_ids.size(1)
        if self._use_clip_text:
            max_len = getattr(getattr(self.encoder, "config", None), "max_position_embeddings", 77)
            if T > max_len:
                input_ids = input_ids[:, :max_len]
                attention_mask = attention_mask[:, :max_len]
                T = max_len

            out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True
            )
            X = out.last_hidden_state
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            X = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

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
# 图像编码：RepVGG（timm）/ CLIP；ROI 由数据阶段提供
# -----------------------------
class ImageModel(nn.Module):
    """
    forward(images, roi_vec=None) -> (P, r)
      - P: (B, proj_dim) 全局向量（RepVGG 或 CLIP）
      - r: (B, proj_dim) 区域向量；若提供 roi_vec(=B×roi_in_dim) 则映射后返回；否则 r=P

    支持 backbone:
      - "repvgg_b0" / "repvgg_b1"(默认) / "repvgg_b1g4" / "repvgg_b2" / "repvgg_b3" ...
      - "openai:clip-..." / "clip-..."（兼容 CLIP 视觉分支）
    """
    def __init__(self,
                 proj_dim: int = 512,
                 use_frcnn_regions: bool = False,
                 frcnn_topk: int = 16,
                 backbone: str = "repvgg_b1",
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

        # 主干：CLIP 或 RepVGG
        self._use_clip_visual = False
        self.clip = None
        self.cnn = None
        self.proj_global = None
        self._clip_image_size = None
        self._repvgg_image_size = 224  # 典型输入尺寸

        name_l = backbone.lower()
        if name_l.startswith("openai") or (("clip" in name_l) and CLIPModel is not None):
            # ---- CLIP 分支 ----
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
            for p in self.clip.vision_model.parameters():
                p.requires_grad = False
            if hasattr(self.clip, "visual_projection"):
                for p in self.clip.visual_projection.parameters():
                    p.requires_grad = False
        else:
            # ---- RepVGG 分支（需要 timm）----
            if timm is None:
                raise ImportError(
                    "未检测到 timm 库，RepVGG 依赖 timm。请先安装：pip install timm"
                )
            # num_classes=0 + global_pool='avg' → 直接输出 (B, feat_dim)
            self.cnn = timm.create_model(
                name_l, pretrained=True, num_classes=0, global_pool='avg'
            )
            feat_dim = getattr(self.cnn, "num_features", None)
            if feat_dim is None:
                # 兜底：部分 timm 版本可通过 feature_info
                fi = getattr(self.cnn, "feature_info", None)
                if fi and len(fi) > 0 and "num_chs" in fi[-1]:
                    feat_dim = fi[-1]["num_chs"]
                else:
                    raise RuntimeError("无法确定 RepVGG 输出维度（num_features 不存在）。请升级 timm。")
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
            # RepVGG 常见输入为 224×224，这里如需可统一 resize（可按需注释）
            if x.shape[-1] != self._repvgg_image_size or x.shape[-2] != self._repvgg_image_size:
                x = F.interpolate(x, size=(self._repvgg_image_size, self._repvgg_image_size),
                                  mode="bilinear", align_corners=False)
            g = self.cnn(x)                    # timm: (B, feat_dim)，已全局池化
            P = self.act(self.proj_global(g))  # (B, proj_dim)

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
        self.Wq = nn.Linear(dim_text, proj_dim)
        self.gamma = nn.Parameter(torch.randn(proj_dim))
        self.tau = nn.Parameter(torch.tensor(
            temperature if temperature is not None else math.sqrt(proj_dim),
            dtype=torch.float32
        ), requires_grad=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.ln = nn.LayerNorm(dim_text)

    def forward(self, H: torch.Tensor, mask_sent: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        B, L, Dt = H.shape
        mask = mask_sent.bool()
        Q  = F.gelu(self.Wq(H))                 # (B,L,Dp)
        Pe = P.unsqueeze(1).expand(-1, L, -1)   # (B,L,Dp)
        fuse  = Pe * Q + Q
        score = torch.einsum("bld,d->bl", fuse, self.gamma) / (self.tau + 1e-6)
        score_fp32 = score.float()
        score_fp32 = score_fp32.masked_fill(~mask, torch.finfo(score_fp32.dtype).min)
        lam   = torch.softmax(score_fp32, dim=-1).to(H.dtype)
        lam   = self.attn_drop(lam)
        c = torch.einsum("bl,bld->bd", lam, H)  # (B,Dt)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (H * mask.unsqueeze(-1)).sum(dim=1) / denom
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
# 整体模型
# -----------------------------
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 文本：Bi-SRU
        self.text_model = TextModel(
            backbone="bisru",
            bert_dim=getattr(config, "text_embed_dim", getattr(config, "text_hidden", 300)),
            gru_hidden=getattr(config, "text_gru_hidden", 384),
            attn_hidden=getattr(config, "hdn_hidden", 512),
            vocab_size=getattr(config, "vocab_size", 21128),
            pad_id=getattr(config, "pad_id", 0),
            dropout=getattr(config, "text_dropout", 0.1),
            pretrained_weight=getattr(config, "pretrained_wordvec", None),
            freeze_embed=getattr(config, "freeze_word_embed", False),
        )
        Dt = 2 * getattr(config, "text_gru_hidden", 384)

        # 图像：RepVGG（默认 b1）
        self.img_model = ImageModel(
            proj_dim=getattr(config, "hdn_hidden", 512),
            backbone=getattr(config, "image_backbone", "repvgg_b1"),
            global_only=getattr(config, "image_global_only", False),
            activation=getattr(config, "image_activation", "relu"),
            input_space=getattr(config, "image_input_space", "imagenet"),
            roi_in_dim=getattr(config, "roi_in_dim", 1024),
        )

        # 融合/低秩/HDN/分类
        self.align = AlignedImageTextFusion(
            dim_text=Dt,
            proj_dim=getattr(config, "hdn_hidden", 512),
            attn_dropout=getattr(config, "attn_dropout", 0.1),
            temperature=getattr(config, "attn_temperature", None),
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

        # 损失
        loss_name = str(getattr(config, "loss", "ce")).lower()
        if loss_name in ("focal", "focalloss"):
            focal_alpha = getattr(config, "focal_alpha", None)
            focal_gamma = float(getattr(config, "focal_gamma", 2.0))
            class_weight = getattr(config, "class_weight", None)
            if float(getattr(config, "label_smoothing", 0.0)) != 0.0:
                print("[warn] FocalLoss 下不建议使用 label_smoothing，已忽略。")
            self.crit = FocalLoss(alpha=focal_alpha, gamma=focal_gamma,
                                  class_weight=class_weight, reduction="mean")
        else:
            weight_cfg = getattr(config, "class_weight", None)
            self.crit = nn.CrossEntropyLoss(
                weight=(torch.tensor(weight_cfg, dtype=torch.float32) if weight_cfg is not None else None),
                label_smoothing=float(getattr(config, "label_smoothing", 0.0))
            )

        # 冻结主干（仅对 ResNet 层名逻辑有效；RepVGG 不走该分支，安全跳过）
        self._maybe_freeze_backbones(config)

        if getattr(config, "compile_submodules", False):
            self._maybe_compile_submodules()

    def _maybe_freeze_backbones(self, config):
        # 文本侧：仅当存在 BERT/CLIP encoder 时生效；Bi-SRU 无 encoder
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

        # 图像侧：原逻辑仅针对 ResNet 的层名冻结；RepVGG 不适用，跳过
        train_layers = tuple(getattr(config, "train_resnet_layers", ("layer4",)))
        if hasattr(self, "img_model") and hasattr(self.img_model, "full_resnet") and self.img_model.full_resnet is not None:
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
        # 文本
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
