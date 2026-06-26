# model.py
'''
1.文本：BERT(本文使用这个）/CLIPText → BiGRU(词) → 词注意力 → 句向量 → BiGRU(句) 得到 S_doc 和一个一步“句序列”H_sent

2.图像：CLIP/ResNet(本文使用这个） → 全局向量 P；（可选）用 roi_vec(1024) 映射得 r

3.对齐：P 引导 H_sent 做注意力，得到文本摘要 c

4.低秩融合：c 和 r → O

5.HDN：融合 V=r、O、S=S_doc → rho4

6.分类：rho4 → Dropout → Linear → logits → pred/loss
'''

# model.py
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.FocalLoss import FocalLoss   # 可选：FocalLoss；没有就用 CE

# ========= transformers =========
# 用于加载文本编码器(BERT/CLIP) 和图像编码器(CLIP Visual)
try:
    from transformers import AutoModel, AutoConfig, CLIPTextModel, CLIPModel
except Exception:
    AutoModel = AutoConfig = CLIPTextModel = CLIPModel = None

# ========= torchvision（只保留 ResNet）=========
# 用于加载 ResNet 图像主干
try:
    import torchvision
    from torchvision.models import resnet18, resnet34, resnet50, resnet152
    from torchvision.models import (
        ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet152_Weights,
    )
except Exception:
    torchvision = None
    resnet18 = resnet34 = resnet50 = None
    ResNet18_Weights = ResNet34_Weights = ResNet50_Weights = None


# -----------------------------
# 词级注意力：把一个句子里每个词的隐藏向量做“加权平均”
# -----------------------------
class WordAttention(nn.Module):
    """词级注意力（批量向量化版），输入 H:(B,T,D), mask:(B,T)，输出 ui:(B,D)
       - H: 词隐藏向量（B 批大小，T 序列长度，D 维度）
       - mask: 哪些位置是有效词（1/True=有效，0/False=padding）
       - 返回 ui: 按注意力权重对 T 个词加权得到的句子向量（每条样本一个）
    """
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)  # 先投影再打分
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 1) 计算每个词的重要性分数
        score = torch.tanh(self.proj(H))      # (B,T,H)
        score = self.v(score).squeeze(-1)     # (B,T)

        # 2) mask 转成 bool，True 表示保留，False 表示忽略
        if mask.dtype != torch.bool:
            mask = mask != 0

        # 3) 防止半精度(FP16)在 softmax 前数值爆炸：强制用 FP32 做 softmax
        score_fp32 = score.float()
        score_fp32 = score_fp32.masked_fill(~mask, torch.finfo(score_fp32.dtype).min)
        a = torch.softmax(score_fp32, dim=-1).to(H.dtype)  # (B,T)

        # 4) 按权重对词向量求和 -> 得到句子向量
        ui = torch.einsum("bt,btd->bd", a, H)  # (B,D)
        return ui


# -----------------------------
# 文本编码模块
# 支持两种“入口”：
#   A) 常规：input_ids + attention_mask -> 调用 Transformers 编码器
#   B) 预计算：token_embeds + attention_mask -> 直接用外部预计算的 Token 向量
# -----------------------------
class TextModel(nn.Module):
    """
    输出三样：
        H_sent   : (B,1,2*gru_hidden)   # 句级序列（这里只有1步，给“句子级GRU”输出）
        S_doc    : (B,2*gru_hidden)     # 文档语义向量（由词注意力得到）
        mask_sent: (B,1)                 # 句级 mask，这里恒为 1（表示存在句子）
    """
    def __init__(self,
                 backbone: str = "bert-base-chinese",  # 文本骨干，支持 BERT 或 "openai:clip-xxx"
                 bert_dim: int = 768,                  # 进入 GRU 前的维度
                 gru_hidden: int = 384,                # GRU 的一侧隐藏维度（双向 -> 输出 *2）
                 attn_hidden: int = 512):              # 注意力内部隐藏维度
        super().__init__()
        self.bert_dim = bert_dim

        # ---- 自动判断是否是 CLIP 的文本编码器 ----
        use_clip = False
        name_l = backbone.lower()
        if name_l.startswith("openai:"):
            # 形如 "openai:clip-vit-base-patch32"
            use_clip = True
            clip_name = backbone.split(":", 1)[1]
        else:
            if AutoConfig is None:
                clip_name = backbone
            else:
                cfg = AutoConfig.from_pretrained(backbone)
                use_clip = getattr(cfg, "model_type", "") == "clip"
                clip_name = backbone

        # ---- 加载编码器（BERT 或 CLIPText）----
        if use_clip:
            if CLIPTextModel is None:
                raise ImportError("transformers.CLIPTextModel 不可用，但配置了 CLIP 文本骨干。")
            self.encoder = CLIPTextModel.from_pretrained(clip_name)
        else:
            if AutoModel is None:
                raise ImportError("transformers.AutoModel 不可用，无法加载 BERT 文本骨干。")
            print('use bert')
            self.encoder = AutoModel.from_pretrained(backbone)

        # 性能/显存小优化
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if hasattr(self.encoder, "config"):
            try:
                # 关闭 cache（训练时节省显存）
                self.encoder.config.use_cache = False
            except Exception:
                pass

        # 记录编码器的输出维度
        self._enc_dim = getattr(getattr(self.encoder, "config", None), "hidden_size", bert_dim)

        # 编码器输出维度 -> 统一映射到 bert_dim（若相同就 Identity）
        self.in_proj = nn.Linear(self._enc_dim, bert_dim) if self._enc_dim != bert_dim else nn.Identity()

        # 词级双向 GRU + 注意力，再过一次“句级 GRU”（这里句长为1，更像是 MLP 的作用）
        self.bigru_word = nn.GRU(bert_dim, gru_hidden, bidirectional=True, batch_first=True)
        self.word_att = WordAttention(2 * gru_hidden, attn_hidden)
        self.bigru_sent = nn.GRU(2 * gru_hidden, gru_hidden, bidirectional=True, batch_first=True)

        self._use_clip_text = use_clip

    # ===== 路径B：直接吃预计算的 token 向量 =====
    def forward_from_embeds(self,
                            token_embeds: torch.Tensor,       # (B,T,enc_dim)
                            attention_mask: torch.Tensor,     # (B,T)
                            lengths: Optional[torch.Tensor]=None  # (B,)
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, enc_dim = token_embeds.shape

        # 若 in_proj 还是 Identity，但外部维度 enc_dim != bert_dim，则临时建一层线性适配
        if isinstance(self.in_proj, nn.Identity):
            if enc_dim != self.bert_dim:
                self.in_proj = nn.Linear(enc_dim, self.bert_dim).to(token_embeds.device)
        else:
            # 如果 in_proj 的输入维度不匹配，直接报错（更早发现问题）
            if self.in_proj.in_features != enc_dim:
                raise ValueError(f"[TextModel] 预计算向量维度({enc_dim}) 与 in_proj.in_features({self.in_proj.in_features}) 不一致。")

        # 统一到 bert_dim
        X = self.in_proj(token_embeds)  # (B,T,bert_dim)

        # 计算每个样本真实长度（可传入，也可由 mask 推得）
        if lengths is None:
            lengths = attention_mask.sum(dim=1).detach().cpu()
        else:
            lengths = lengths.detach().cpu()

        # 词级双向 GRU（pack 能跳过 padding）
        packed = nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
        H_word_packed, _ = self.bigru_word(packed)
        H_word, _ = nn.utils.rnn.pad_packed_sequence(H_word_packed, batch_first=True, total_length=T)  # (B,T,2h)

        # 词注意力 -> 句子向量；再过“句级 GRU”（这里时间步是1）
        ui = self.word_att(H_word, attention_mask)     # (B,2h)
        Hi, _ = self.bigru_sent(ui.unsqueeze(1))       # (B,1,2h)

        mask_sent = attention_mask.new_ones((B, 1))
        return Hi, ui, mask_sent

    # ===== 路径A：常规 BERT/CLIPText 编码 =====
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_embeds: torch.Tensor = None,     # 若传了，就走上面的“预计算路径”
        token_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # 若外部已提供 token_embeds，则直接复用，不走编码器
        if token_embeds is not None:
            X = self.in_proj(token_embeds)  # (B,T,bert_dim)
            B, T, _ = X.shape

            # 用非零判断得到真实长度（防止 mask 没传）
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

        # 走编码器（BERT/CLIPText）
        B, T = input_ids.size(0), input_ids.size(1)
        if self._use_clip_text:
            # CLIP 文本有最大长度限制（常见 77）
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

        # 下游统一到 bert_dim，再走词级 GRU + 注意力 + 句级 GRU
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
# 图像编码：支持两条路
#   1) CLIP 图像编码器（若 backbone 指定了 clip）
#   2) ResNet 主干（18/34/50/152），自适应平均池 + 线性到 proj_dim
# ROI（区域特征）在数据阶段就算好（1024维），这里只做线性映射成 proj_dim
# -----------------------------
class ImageModel(nn.Module):
    """
    forward(images, roi_vec=None) -> (P, r)
      - images: (B,3,H,W) 输入图像（已按 config 归一化/尺寸处理）
      - roi_vec: 可选，(B, 1024) Faster R-CNN 汇总的 ROI 向量
      - P: (B, proj_dim) 图像的全局向量（CLIP/ResNet）
      - r: (B, proj_dim) 区域向量（若没传 roi，就退化为 P）
    """
    def __init__(self,
                 proj_dim: int = 512,
                 use_frcnn_regions: bool = False,  # 这里留接口位，不在本类内部跑检测
                 frcnn_topk: int = 16,
                 backbone: str = "resnet50",
                 global_only: bool = False,
                 region_heads: int = 8,
                 region_pool: Optional[int] = None,
                 activation: str = "relu",
                 input_space: str = "imagenet",  # 图片归一化空间："imagenet"/"clip"/"raw"
                 roi_in_dim: int = 1024):
        super().__init__()
        self.proj_dim = proj_dim
        self.global_only = bool(global_only)
        self.input_space = input_space.lower()

        act = activation.lower()
        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.GELU()

        # 注册均值方差（做空间互转用）
        self.register_buffer("_imnet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("_imnet_std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.register_buffer("_clip_mean",  torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1))
        self.register_buffer("_clip_std",   torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1))

        # 选择图像主干：CLIP or ResNet
        self._use_clip_visual = False
        self.clip = None
        self.full_resnet = None
        self.cnn = None
        self.pool = None
        self.proj_global = None
        self._clip_image_size = None

        name_l = backbone.lower()
        if name_l.startswith("openai") or (("clip" in name_l) and CLIPModel is not None):
            # 使用 CLIP 视觉编码器（需 transformers 支持）
            if CLIPModel is None:
                raise ImportError("transformers.CLIPModel 不可用，但配置了 CLIP 视觉骨干。")
            clip_name = backbone.split(":", 1)[1] if ":" in backbone else backbone
            self.clip = CLIPModel.from_pretrained(clip_name)
            clip_dim = self.clip.config.projection_dim  # 一般是 512
            self.proj_global = nn.Identity() if clip_dim == proj_dim else nn.Linear(clip_dim, proj_dim)
            self._use_clip_visual = True
            vc = getattr(self.clip, "config", None)
            vc = getattr(vc, "vision_config", None)
            self._clip_image_size = getattr(vc, "image_size", 224)
            # 默认冻结 CLIP 视觉编码器参数（可按需要解冻）
            for p in self.clip.vision_model.parameters():
                p.requires_grad = False
            if hasattr(self.clip, "visual_projection"):
                for p in self.clip.visual_projection.parameters():
                    p.requires_grad = False
        else:
            # 使用 ResNet（torchvision）
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
            self.cnn = nn.Sequential(*list(net.children())[:-2])  # 去掉最后两层（池化+FC），保留 conv 特征
            self.cnn.to(memory_format=torch.channels_last)        # 节省显存/加速
            self.pool = nn.AdaptiveAvgPool2d((1,1))               # 全局平均池化
            self.proj_global = nn.Linear(feat_dim, proj_dim)      # 映射到下游统一维度

        # ROI 线性投影层：把 1024 维区域特征映射到 proj_dim
        self.proj_roi = nn.Linear(roi_in_dim, proj_dim)

    # ---------- 归一化互转（当主干与输入空间不一致时，做一次转换） ----------
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

    # ---------- 前向：输出全局向量 P 和 区域向量 r ----------
    def forward(self, images: torch.Tensor, roi_vec: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = images.contiguous(memory_format=torch.channels_last)

        # 1) 全局图像向量 P
        if self._use_clip_visual:
            # 若主干是 CLIP，则把输入空间转成 CLIP 的均值方差，并按 CLIP 要求的分辨率插值
            x_clip = self._to_clip(x)
            if self._clip_image_size is not None and (
                x_clip.shape[-1] != self._clip_image_size or x_clip.shape[-2] != self._clip_image_size
            ):
                x_clip = F.interpolate(x_clip, size=(self._clip_image_size, self._clip_image_size),
                                       mode="bicubic", align_corners=False)
            g_vec = self.clip.get_image_features(pixel_values=x_clip)  # (B, clip_dim)
            g_vec = F.normalize(g_vec, dim=-1)                         # 归一化（稳定）
            P = self.act(self.proj_global(g_vec))                      # -> proj_dim
        else:
            # ResNet 路线：CNN -> GAP -> Linear
            feat = self.cnn(x)                # (B,C,h,w)
            g = self.pool(feat).flatten(1)    # (B,C)
            P = self.act(self.proj_global(g)) # (B,proj_dim)

        # 2) 区域向量 r（若没有 roi_vec，则退化为 P）
        if roi_vec is not None:
            r = self.act(self.proj_roi(roi_vec.to(x.device)))
        else:
            r = P

        return P, r


# -----------------------------
# 图文对齐注意力（把文本序列 H 与图像全局 P 对齐，得到文本侧摘要 c）
# 关键点：softmax 在 FP32 中完成，避免 FP16 溢出
# -----------------------------
class AlignedImageTextFusion(nn.Module):
    def __init__(self, dim_text: int, proj_dim: int,
                 attn_dropout: float = 0.1,
                 temperature: float = None):
        super().__init__()
        self.Wq = nn.Linear(dim_text, proj_dim)          # 把文本序列 H 映射到与图像同维度
        self.gamma = nn.Parameter(torch.randn(proj_dim)) # 注意力向量（可学习）
        # 温度（缩放因子）：默认 sqrt(Dp)，可学习
        self.tau = nn.Parameter(torch.tensor(
            temperature if temperature is not None else math.sqrt(proj_dim),
            dtype=torch.float32
        ), requires_grad=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.ln = nn.LayerNorm(dim_text)                 # 残差 + LN 稳定训练

    def forward(self, H: torch.Tensor, mask_sent: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """
        H: (B,L,Dt) 文本序列向量
        mask_sent:(B,L) 文本有效位置
        P:(B,Dp)      图像全局向量
        return c: (B,Dt) 文本摘要向量（与图像对齐后的加权和）
        """
        B, L, Dt = H.shape
        mask = mask_sent.bool()

        Q  = F.gelu(self.Wq(H))                 # (B,L,Dp)
        Pe = P.unsqueeze(1).expand(-1, L, -1)   # (B,L,Dp)

        # 简单的乘性融合：Pe * Q + Q，再与 gamma 点积得到每个时间步的权重分数
        fuse  = Pe * Q + Q                      # (B,L,Dp)
        score = torch.einsum("bld,d->bl", fuse, self.gamma) / (self.tau + 1e-6)  # (B,L)

        # FP32 + mask 的 softmax，数值更稳
        score_fp32 = score.float()
        score_fp32 = score_fp32.masked_fill(~mask, torch.finfo(score_fp32.dtype).min)
        lam   = torch.softmax(score_fp32, dim=-1).to(H.dtype)    # (B,L)
        lam   = self.attn_drop(lam)                              # dropout 正则

        # 按权重对 H 求和 -> 得到对齐后的文本摘要 c
        c = torch.einsum("bl,bld->bd", lam, H)  # (B,Dt)

        # 残差：再和 masked-mean 做一次融合，最后 LayerNorm
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (H * mask.unsqueeze(-1)).sum(dim=1) / denom  # (B,Dt)
        c = self.ln(c + pooled)
        return c


# -----------------------------
# 低秩张量融合（常用的高效双模态融合技巧）
# 把 c(文本) 与 r(区域/图像) 融合得到 O
# -----------------------------
class LowRankFusion(nn.Module):
    def __init__(self, dim_c: int, dim_r: int, out_dim: int, rank: int = 4):
        super().__init__()
        self.rank = rank
        # 两套低秩参数（包含偏置项 +1），最后按秩求 Hadamard 积再求和
        self.Wc = nn.Parameter(torch.empty(rank, dim_c + 1, out_dim))
        self.Wr = nn.Parameter(torch.empty(rank, dim_r + 1, out_dim))
        nn.init.xavier_uniform_(self.Wc)
        nn.init.xavier_uniform_(self.Wr)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        B = c.size(0)
        # 拼上 1 实现“偏置项”的效果
        Zc = torch.cat([c, c.new_ones(B, 1)], dim=-1)          # (B, dc+1)
        Zr = torch.cat([r, r.new_ones(B, 1)], dim=-1)          # (B, dr+1)
        Vc = torch.einsum('bd,rdp->brp', Zc, self.Wc)          # (B,rank,out_dim)
        Vr = torch.einsum('bd,rdp->brp', Zr, self.Wr)          # (B,rank,out_dim)
        O = (Vc * Vr).sum(dim=1) + self.bias                   # (B,out_dim)
        return self.norm(O)


# -----------------------------
# 分层动态邻域（HDN）：核心融合头
# 目标：根据“能量/置信度”自适应地混合 V(视觉)、O(低秩融合)、S(文本)
# -----------------------------
class EnergyHead(nn.Module):
    """把向量映射到类别得分，并计算一个“能量”标量（越小越自信）"""
    def __init__(self, in_dim: int, num_classes: int, T: float = 1.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self.T = T

    def forward(self, x: torch.Tensor):
        z = self.fc(x) / max(self.T, 1e-4)          # 先按温度缩放
        energy = self.T * torch.logsumexp(z, dim=-1)  # E(x) ~ LogSumExp
        logp = -energy / max(self.T, 1e-4)            # 简单得到一个“置信度”趋势
        return energy, logp


class HierarchicalDynamicNeighborhood(nn.Module):
    """
    HDN 流程（直观版）：
      1) 用 EnergyHead 估计三路(V,O,S)的“能量/置信度”，得到自适应权 alpha
      2) 分别做一次线性变换/归一化 -> 第一层对象
      3) 三路两两交互：v+v*o+v*s, o+o*v+o*s, s+s*v+s*o  （一种简单但有效的互相影响）
      4) 两次“等权/维度自适应”的混合（eps3/eps4） -> v4,o4,s4
      5) 按元素相乘得到最终融合 rho4（带有三路耦合）
    """
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
        # 三路互相作用：保留自身 + 与其他路的交互项
        return v + v * o + v * s, o + o * v + o * s, s + s * v + s * o

    @staticmethod
    def _eps(v, o, s):
        # 根据维度做一个“等权倾向”（也可以换成可学习/注意力）
        d = torch.tensor([
            1 / max(v.size(-1), 1),
            1 / max(o.size(-1), 1),
            1 / max(s.size(-1), 1)
        ], device=v.device).view(1, 3)
        return torch.softmax(torch.log(d), dim=-1)

    def forward(self, V: torch.Tensor, O: torch.Tensor, S: torch.Tensor):
        # 1) 估计能量 -> 自适应权 alpha（能量越小，权重越大）
        eV, lpV = self.Ev(V)
        eO, lpO = self.Eo(O)
        eS, lpS = self.Es(S)
        alpha = torch.softmax(torch.stack([-eV, -eO, -eS], dim=-1), dim=-1)  # (B,3)

        # 2) 线性 + 归一化，再乘上权重；(1 + logp) 给置信度一点正向放大
        V1 = self.norm(self.Pv(V)) * alpha[:, 0:1] * (1 + lpV.unsqueeze(-1))
        O1 = self.norm(self.Po(O)) * alpha[:, 1:2] * (1 + lpO.unsqueeze(-1))
        S1 = self.norm(self.Ps(S)) * alpha[:, 2:3] * (1 + lpS.unsqueeze(-1))

        # 3) 三路互相影响
        v2, o2, s2 = self._object_layer(V1, O1, S1)
        v2, o2, s2 = self.norm(v2), self.norm(o2), self.norm(s2)

        # 4) 两次“等权/维度自适应”的混合（可以理解为层级聚合）
        eps3 = self._eps(v2, o2, s2)
        mix3 = eps3[:, 0:1] * v2 + eps3[:, 1:2] * o2 + eps3[:, 2:3] * s2
        v3 = o3 = s3 = self.norm(mix3)

        eps4 = self._eps(v3, o3, s3)
        mix4 = eps4[:, 0:1] * v3 + eps4[:, 1:2] * o3 + eps4[:, 2:3] * s3
        v4 = o4 = s4 = self.norm(mix4)

        # 5) 最终融合：三路按元素相乘（让三者都一致时信号最强）
        rho4 = v4 * o4 * s4
        aux = {"alpha": alpha, "eps3": eps3, "eps4": eps4}  # 便于可视化/调试
        return rho4, aux


# -----------------------------
# 总模型：文本/图像 -> 对齐 -> 低秩 -> HDN -> 分类
# 训练时可选：冻结主干层、torch.compile 加速、Focal/CE 损失
# -----------------------------
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1) 文本侧：BERT/CLIPText + BiGRU + 词注意力 + 句级GRU
        self.text_model = TextModel(
            backbone=getattr(config, "text_backbone", "bert-base-chinese"),
            bert_dim=getattr(config, "text_hidden", 768),
            gru_hidden=getattr(config, "text_gru_hidden", 384),
            attn_hidden=getattr(config, "hdn_hidden", 512),
        )
        Dt = 2 * getattr(config, "text_gru_hidden", 384)  # 双向 GRU 输出维度

        # 2) 图像侧：CLIP 或 ResNet，输出统一到 hdn_hidden 维
        self.img_model = ImageModel(
            proj_dim=getattr(config, "hdn_hidden", 512),
            backbone=getattr(config, "image_backbone", "resnet50"),
            global_only=getattr(config, "image_global_only", False),
            activation=getattr(config, "image_activation", "relu"),
            input_space=getattr(config, "image_input_space", "imagenet"),
            roi_in_dim=getattr(config, "roi_in_dim", 1024),
        )

        # 3) 文本-图像对齐注意力（用图像引导文本）
        self.align = AlignedImageTextFusion(
            dim_text=Dt,
            proj_dim=getattr(config, "hdn_hidden", 512),
            attn_dropout=getattr(config, "attn_dropout", 0.1),
            temperature=getattr(config, "attn_temperature", None),  # None -> sqrt(proj_dim)
        )

        # 4) 低秩张量融合：对齐后的文本 c 与 区域 r -> O
        self.lowrank = LowRankFusion(
            dim_c=Dt,
            dim_r=getattr(config, "hdn_hidden", 512),
            out_dim=getattr(config, "lowrank_out", 512),
            rank=getattr(config, "lowrank_rank", 4),
        )

        # 5) HDN：V(=r), O, S(=文本文档向量) 三路自适应融合 -> rho4
        self.hdn = HierarchicalDynamicNeighborhood(
            dim_v=getattr(config, "hdn_hidden", 512),      # 来自图像侧 r
            dim_o=getattr(config, "lowrank_out", 512),     # 来自低秩融合 O
            dim_s=Dt,                                      # 来自文本侧 S_doc
            hidden=getattr(config, "hdn_hidden", 512),
            num_classes=getattr(config, "num_classes", 2),
            T=getattr(config, "temperature", 1.0),
        )

        # 6) 分类器：对 rho4 做 dropout + Linear 到类别数
        self.dropout = nn.Dropout(getattr(config, "dropout", 0.1))
        self.cls = nn.Linear(getattr(config, "hdn_hidden", 512), getattr(config, "num_classes", 2))

        # 7) 损失：CE 或 Focal（支持 class_weight）
        loss_name = str(getattr(config, "loss", "ce")).lower()
        if loss_name in ("focal", "focalloss"):
            focal_alpha = getattr(config, "focal_alpha", None)   # e.g. [α_neg, α_pos]
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

        # 8) 可选：冻结主干（BERT 仅训练最后 n 层；ResNet 仅 layer4 等）
        self._maybe_freeze_backbones(config)

        # 9) 可选：子模块 torch.compile（PyTorch 2.x）
        if getattr(config, "compile_submodules", False):
            self._maybe_compile_submodules()

    # 冻结策略：减少训练参数/显存，适合小数据集或迁移学习
    def _maybe_freeze_backbones(self, config):
        # 文本骨干：只训练最后 n 层（n>=0）
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
            # BERT pooler（若有）默认解冻
            pooler = getattr(bert, "pooler", None)
            if pooler is not None:
                for p in pooler.parameters():
                    p.requires_grad = True

        # 图像骨干：只训练指定层（默认只训练 layer4）
        train_layers = tuple(getattr(config, "train_resnet_layers", ("layer4",)))
        if hasattr(self.img_model, "full_resnet") and self.img_model.full_resnet is not None:
            for name, p in self.img_model.full_resnet.named_parameters():
                flag = any(name.startswith(tl) for tl in train_layers)
                p.requires_grad = flag

    # compile 子模块：可能获得推理/训练加速（视环境而定）
    def _maybe_compile_submodules(self):
        try:
            self.text_model.bigru_word = torch.compile(self.text_model.bigru_word, mode="max-autotune")
            self.text_model.bigru_sent = torch.compile(self.text_model.bigru_sent, mode="max-autotune")
            self.align = torch.compile(self.align, mode="max-autotune")
            self.lowrank = torch.compile(self.lowrank, mode="max-autotune")
            self.hdn = torch.compile(self.hdn, mode="max-autotune")
        except Exception:
            pass

    # 统一的前向
    # 输入：
    #   texts, texts_mask       -> 文本 ids 和 mask
    #   imgs                    -> 图像张量（变换后）
    #   labels (可选)           -> 标签（用于计算 loss）
    #   roi_vec (可选)          -> (B,1024) 的 ROI 特征（数据阶段预先算好）
    #   token_embeds/lengths    -> 若启用“文本预计算”，可直接喂进来，跳过编码器
    # 输出：
    #   pred: (B,)  argmax(logits)
    #   loss: 标量；若 labels=None 则返回 0.0（与工程用法兼容）
    def forward(self,
                texts: torch.Tensor,
                texts_mask: torch.Tensor,
                imgs: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                roi_vec: Optional[torch.Tensor] = None,
                token_embeds: Optional[torch.Tensor] = None,
                token_lengths: Optional[torch.Tensor] = None,
                **kwargs):
        # 文本侧（自动兼容：直接 ids -> 编码；或传入 token_embeds -> 走预计算路径）
        try:
            H_sent, S_doc, mask_sent = self.text_model(
                texts, texts_mask,
                token_embeds=token_embeds, token_lengths=token_lengths
            )
        except TypeError:
            # 某些老版本 transformers 的 forward 签名不支持关键字参数
            H_sent, S_doc, mask_sent = self.text_model(texts, texts_mask)

        # 图像侧：输出全局向量 P 与 区域向量 r
        P, r = self.img_model(imgs, roi_vec=roi_vec)

        # 文图对齐：用图像全局 P 引导文本序列 H，得到文本摘要 c
        c = self.align(H_sent, mask_sent, P)

        # 低秩融合：把 c 与 r 合成 O
        O = self.lowrank(c, r)

        # HDN 融合：三路（V=r, O, S=S_doc）多层自适应融合，得到最终特征 rho4
        rho4, _ = self.hdn(V=r, O=O, S=S_doc)

        # 分类头
        logits = self.cls(self.dropout(rho4))
        pred = torch.argmax(logits, dim=-1)

        # 损失（训练时才用；推理时返回 0.0 张量，方便统一接口）
        loss = self.crit(logits, labels) if labels is not None else torch.tensor(0.0, device=logits.device)
        return pred, loss
