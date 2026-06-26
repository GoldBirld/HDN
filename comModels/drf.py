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

# ========= torchvision（保留 ResNet）=========
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
# 词级注意力（与原工程保持一致）
# -----------------------------
class WordAttention(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = torch.tanh(self.proj(H))          # (B,T,H)
        score = self.v(score).squeeze(-1)         # (B,T)
        if mask.dtype != torch.bool:
            mask = mask != 0
        score_fp32 = score.float()
        score_fp32 = score_fp32.masked_fill(~mask, torch.finfo(score_fp32.dtype).min)
        a = torch.softmax(score_fp32, dim=-1).to(H.dtype)      # (B,T)
        ui = torch.einsum("bt,btd->bd", a, H)                  # (B,D)
        return ui


# -----------------------------
# 文本编码：BERT -> BiGRU + 注意力（保持原接口）
# 输出:
#   H_sent:(B,1,2h)  S_doc:(B,2h)  mask_sent:(B,1)
# -----------------------------
class TextModel(nn.Module):
    def __init__(self,
                 backbone: str = "bert-base-chinese",
                 bert_dim: int = 768,
                 gru_hidden: int = 384,
                 attn_hidden: int = 512):
        super().__init__()
        self.bert_dim = bert_dim

        # 可能使用 CLIP 文本
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

        if use_clip:
            if CLIPTextModel is None:
                raise ImportError("transformers.CLIPTextModel 不可用，但你配置了 CLIP 文本骨干。")
            self.encoder = CLIPTextModel.from_pretrained(clip_name)
        else:
            if AutoModel is None:
                raise ImportError("transformers.AutoModel 不可用，无法加载 BERT 文本骨干。")
            self.encoder = AutoModel.from_pretrained(backbone)

        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if hasattr(self.encoder, "config"):
            try:
                self.encoder.config.use_cache = False
            except Exception:
                pass

        self._enc_dim = getattr(getattr(self.encoder, "config", None), "hidden_size", bert_dim)
        self.in_proj = nn.Linear(self._enc_dim, bert_dim) if self._enc_dim != bert_dim else nn.Identity()

        self.bigru_word = nn.GRU(bert_dim, gru_hidden, bidirectional=True, batch_first=True)
        self.word_att = WordAttention(2 * gru_hidden, attn_hidden)
        self.bigru_sent = nn.GRU(2 * gru_hidden, gru_hidden, bidirectional=True, batch_first=True)

        self._use_clip_text = use_clip

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_embeds: torch.Tensor = None,
        token_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 预计算向量直通
        if token_embeds is not None:
            X = self.in_proj(token_embeds)
            B, T, _ = X.shape
            if token_lengths is None:
                with torch.no_grad():
                    token_lengths = (token_embeds.abs().sum(dim=-1) > 0).long().sum(dim=1)
            lengths = token_lengths.detach().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
            H_word_packed, _ = self.bigru_word(packed)
            H_word, _ = nn.utils.rnn.pad_packed_sequence(H_word_packed, batch_first=True, total_length=T)
            ui = self.word_att(H_word, (torch.arange(T, device=X.device)[None, :] < token_lengths[:, None]))
            Hi, _ = self.bigru_sent(ui.unsqueeze(1))
            return Hi, ui, attention_mask.new_ones((B, 1))

        # 常规 BERT/CLIP
        B, T = input_ids.shape
        if self._use_clip_text:
            max_len = getattr(getattr(self.encoder, "config", None), "max_position_embeddings", 77)
            if T > max_len:
                input_ids = input_ids[:, :max_len]
                attention_mask = attention_mask[:, :max_len]
                T = max_len
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            X = out.last_hidden_state
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            X = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

        X = self.in_proj(X)
        lengths = attention_mask.sum(dim=1).detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
        H_word_packed, _ = self.bigru_word(packed)
        H_word, _ = nn.utils.rnn.pad_packed_sequence(H_word_packed, batch_first=True, total_length=T)
        ui = self.word_att(H_word, attention_mask)
        Hi, _ = self.bigru_sent(ui.unsqueeze(1))
        return Hi, ui, attention_mask.new_ones((B, 1))


# -----------------------------
# 图像编码：CLIP 视觉 或 ResNet -> 全局向量
# -----------------------------
class ImageModel(nn.Module):
    def __init__(self,
                 proj_dim: int = 512,
                 backbone: str = "resnet50",
                 global_only: bool = True,
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

        # ROI 投影（如需用离线 ROI 特征）
        self.proj_roi = nn.Linear(roi_in_dim, proj_dim)

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

    def forward(self, images: torch.Tensor, roi_vec: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = images.contiguous(memory_format=torch.channels_last)
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

        if roi_vec is not None:
            r = self.act(self.proj_roi(roi_vec.to(x.device)))
        else:
            r = P
        return P, r


# =========================================================
# 分布库（Distribution Bank）：每类 K 个原型（均值+方差）
# =========================================================
class DistributionBank(nn.Module):
    """
    按类维护 K 个原型（均值与各向同性方差），支持在推理时跨全类软匹配。
    mu:   (C, K, D)
    logv: (C, K, 1)  各向同性方差 log(sigma^2)
    """
    def __init__(self, num_classes: int, dim: int, num_protos: int = 4, init_std: float = 0.02):
        super().__init__()
        self.C = num_classes
        self.K = num_protos
        self.D = dim

        self.mu = nn.Parameter(torch.randn(self.C, self.K, self.D) * init_std)
        self.logv = nn.Parameter(torch.zeros(self.C, self.K, 1))  # 初始方差为 1

        # 温度/缩放
        self.tau_dist = nn.Parameter(torch.tensor(1.0))

    def _mahalanobis(self, x: torch.Tensor, mu: torch.Tensor, logv: torch.Tensor) -> torch.Tensor:
        # x:  (B, D)
        # mu: (C, K, D)
        # logv: (C, K, 1)
        # return dists: (B, C, K)
        B, D = x.shape
        C, K, _ = mu.shape
        x_exp = x[:, None, None, :].expand(B, C, K, D)
        var = torch.exp(logv).clamp_min(1e-6)  # (C,K,1)
        inv_var = 1.0 / var
        diff = x_exp - mu[None, :, :, :]
        # 各向同性：sum( (x-mu)^2 / var )
        d2 = (diff.pow(2) * inv_var[None, :, :, :]).sum(dim=-1)  # (B,C,K)
        # + log|Sigma| 项（各向同性时为 D*log sigma^2）
        logdet = (self.D * logv.squeeze(-1))[None, :, :]         # (1,C,K)
        dist = 0.5 * (d2 + logdet)
        return dist  # (B,C,K)

    def recover(self,
                x: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        输入特征 x(B,D)，可选标签 labels(B,)
        返回：
          x_rec(B,D): 恢复后的特征（靠近分布均值）
          conf(B,1):  置信度（根据最近原型的距离转化）
          cls_idx(B,): 若 labels=None，则为软最近类（依据最近原型所在类）
        """
        B, D = x.shape
        dist = self._mahalanobis(x, self.mu, self.logv)  # (B,C,K)

        if labels is not None:
            # 仅在该类原型上做软匹配
            labels = labels.clamp_min(0).clamp_max(self.C - 1)
            mask = F.one_hot(labels, num_classes=self.C).bool()          # (B,C)
            mask = mask.unsqueeze(-1).expand(-1, -1, self.K)             # (B,C,K)
            dist_sel = dist.masked_fill(~mask, float('inf'))             # (B,C,K) 其他类屏蔽
            cls_idx = labels
        else:
            dist_sel = dist
            # 最近原型所在类
            c_star = torch.argmin(dist.view(B, -1), dim=-1)              # (B,)
            cls_idx = (c_star // self.K)

        # 软权重（在被选集合上做 softmin）
        score = -dist_sel / (self.tau_dist.abs() + 1e-6)
        # 为了数值稳定，将 inf 变成 -inf，避免 softmax 影响
        score[torch.isinf(score)] = -float('inf')
        w = torch.softmax(score.float(), dim=-1).to(x.dtype)             # (B,C,K)

        # 计算加权均值 μ_hat
        mu_exp = self.mu[None, :, :, :].expand(B, -1, -1, -1)            # (B,C,K,D)
        mu_hat = (w.unsqueeze(-1) * mu_exp).sum(dim=(1, 2))              # (B,D)

        # 恢复：μ_hat + alpha*(x - μ_hat)
        # alpha 可学习（限制在 0..1）
        if not hasattr(self, "alpha"):
            self.alpha = nn.Parameter(torch.tensor(0.5))
        alpha = torch.sigmoid(self.alpha)
        x_rec = mu_hat + alpha * (x - mu_hat)

        # 置信度：由最近原型的距离 -> exp(-d/temperature)
        min_dist, _ = torch.min(dist.view(B, -1), dim=-1)                # (B,)
        conf = torch.exp(-min_dist / (self.tau_dist.abs() + 1e-6)).unsqueeze(-1)  # (B,1)
        conf = conf.clamp(max=1.0)

        return x_rec, conf, cls_idx

    def proto_pull_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """原型对齐正则：同类最近原型的马氏距离（越小越好）"""
        dist = self._mahalanobis(x, self.mu, self.logv)  # (B,C,K)
        B = x.size(0)
        y = y.clamp_min(0).clamp_max(self.C - 1)
        gather = dist[torch.arange(B, device=x.device), y, :]            # (B,K)
        min_d, _ = torch.min(gather, dim=-1)                             # (B,)
        return min_d.mean()


# =========================================================
# 分布式恢复 + 置信门控融合
# =========================================================
class DBFRFusion(nn.Module):
    """
    输入两模态恢复特征及置信度，做门控融合 + 互作 MLP。
    输出融合向量 fused(B,D)
    """
    def __init__(self, dim: int, hidden: int = 2_048, dropout: float = 0.1):
        super().__init__()
        self.gate_t = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )
        self.gate_v = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )

        # 互作向量：t, v, |t-v|, t*v
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, t_rec: torch.Tensor, v_rec: torch.Tensor,
                conf_t: torch.Tensor, conf_v: torch.Tensor) -> torch.Tensor:
        # 置信门控（sigmoid + conf），再归一化
        gt = torch.sigmoid(self.gate_t(t_rec)) * conf_t           # (B,1)
        gv = torch.sigmoid(self.gate_v(v_rec)) * conf_v           # (B,1)
        denom = (gt + gv).clamp_min(1e-6)
        wt = gt / denom
        wv = gv / denom

        # 互作通道
        z = torch.cat([t_rec, v_rec, torch.abs(t_rec - v_rec), t_rec * v_rec], dim=-1)
        inter = self.mlp(z)                                       # (B,D)

        fused = wt * t_rec + wv * v_rec
        fused = self.ln(fused + inter)                            # 残差 + LN
        return fused


# =========================================================
# 整体模型：Text/Image -> 线性投影 -> 分布恢复 -> 融合 -> 分类
# =========================================================
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_classes = getattr(config, "num_classes", 2)
        joint_dim = getattr(config, "joint_dim", 512)  # 统一对齐后的特征维度
        text_gru_hidden = getattr(config, "text_gru_hidden", 384)
        Dt = 2 * text_gru_hidden

        # 1) 文本侧（保持你工程中的 TextModel）
        self.text_model = TextModel(
            backbone=getattr(config, "text_backbone", "bert-base-chinese"),
            bert_dim=getattr(config, "text_hidden", 768),
            gru_hidden=text_gru_hidden,
            attn_hidden=getattr(config, "hdn_hidden", joint_dim),
        )
        self.txt_proj = nn.Linear(Dt, joint_dim)

        # 2) 图像侧（保持你工程中的 ImageModel）
        self.img_model = ImageModel(
            proj_dim=joint_dim,
            backbone=getattr(config, "image_backbone", "resnet50"),
            global_only=getattr(config, "image_global_only", True),
            activation=getattr(config, "image_activation", "relu"),
            input_space=getattr(config, "image_input_space", "imagenet"),
            roi_in_dim=getattr(config, "roi_in_dim", 1024),
        )

        # 3) 分布库（每类 K 个原型），分别为文本与图像
        num_protos = getattr(config, "num_prototypes", 4)
        self.bank_text = DistributionBank(num_classes=num_classes, dim=joint_dim, num_protos=num_protos)
        self.bank_vision = DistributionBank(num_classes=num_classes, dim=joint_dim, num_protos=num_protos)

        # 4) 融合（分布恢复后）
        self.fusion = DBFRFusion(dim=joint_dim,
                                 hidden=getattr(config, "fusion_hidden", 2048),
                                 dropout=getattr(config, "dropout", 0.1))

        # 5) 分类
        self.dropout = nn.Dropout(getattr(config, "dropout", 0.1))
        self.cls = nn.Linear(joint_dim, num_classes)

        # 6) 额外的单模态辅助头（可选，用于无标签推理时的类引导；这里保留，训练时不必强用）
        self.head_t = nn.Linear(joint_dim, num_classes)
        self.head_v = nn.Linear(joint_dim, num_classes)

        # 7) 损失
        loss_name = str(getattr(config, "loss", "ce")).lower()
        if loss_name in ("focal", "focalloss"):
            focal_alpha = getattr(config, "focal_alpha", None)
            focal_gamma = float(getattr(config, "focal_gamma", 2.0))
            class_weight = getattr(config, "class_weight", None)
            if float(getattr(config, "label_smoothing", 0.0)) != 0.0:
                print("[warn] FocalLoss 下已忽略 label_smoothing。")
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

        # 原型拉近正则的权重
        self.lambda_proto = float(getattr(config, "lambda_proto", 0.0))

        # 可选：冻结主干层（与原工程一致）
        self._maybe_freeze_backbones(config)

        # 可选：编译加速
        if getattr(config, "compile_submodules", False):
            self._maybe_compile_submodules()

    # 与原工程一致的冻结策略
    def _maybe_freeze_backbones(self, config):
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

        train_layers = tuple(getattr(config, "train_resnet_layers", ("layer4",)))
        if hasattr(self.img_model, "full_resnet") and self.img_model.full_resnet is not None:
            for name, p in self.img_model.full_resnet.named_parameters():
                flag = any(name.startswith(tl) for tl in train_layers)
                p.requires_grad = flag

    def _maybe_compile_submodules(self):
        try:
            self.text_model.bigru_word = torch.compile(self.text_model.bigru_word, mode="max-autotune")
            self.text_model.bigru_sent = torch.compile(self.text_model.bigru_sent, mode="max-autotune")
            self.fusion = torch.compile(self.fusion, mode="max-autotune")
            self.bank_text = torch.compile(self.bank_text, mode="max-autotune")
            self.bank_vision = torch.compile(self.bank_vision, mode="max-autotune")
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
        # 1) 文本编码
        try:
            H_sent, S_doc, _ = self.text_model(
                texts, texts_mask,
                token_embeds=token_embeds, token_lengths=token_lengths
            )
        except TypeError:
            H_sent, S_doc, _ = self.text_model(texts, texts_mask)
        t = self.txt_proj(S_doc)                           # (B, D)

        # 2) 图像编码
        P, r = self.img_model(imgs, roi_vec=roi_vec)       # (B, D), (B, D)
        v = r                                              # 使用区域/全局向量作为图像表示

        # 3) 分布恢复（若 labels=None，则跨全类软匹配）
        t_rec, conf_t, cls_t = self.bank_text.recover(t, labels=labels)
        v_rec, conf_v, cls_v = self.bank_vision.recover(v, labels=labels)

        # 4) 融合
        fused = self.fusion(t_rec, v_rec, conf_t, conf_v)  # (B, D)

        # 5) 分类
        logits = self.cls(self.dropout(fused))             # (B, C)
        pred = torch.argmax(logits, dim=-1)

        # 6) 损失
        if labels is not None:
            loss_main = self.crit(logits, labels)
            loss_reg = 0.0
            if self.lambda_proto > 0:
                loss_reg_t = self.bank_text.proto_pull_loss(t, labels)
                loss_reg_v = self.bank_vision.proto_pull_loss(v, labels)
                loss_reg = (loss_reg_t + loss_reg_v) * 0.5 * self.lambda_proto
            loss = loss_main + loss_reg
        else:
            loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        return pred, loss
