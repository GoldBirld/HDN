import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Multi-class focal loss (softmax 版本).
    支持:
      - alpha: 张量/列表，按类别 id 排列的权重（例如 [alpha_neg, alpha_pos]）
      - gamma: 聚焦参数，默认 2.0
      - class_weight: 额外的类别权重(与CE的weight一致语义)，会与 alpha 相乘
    建议 focal 下将 label_smoothing 设为 0.0
    """
    def __init__(self, alpha=None, gamma: float = 2.0,
                 reduction: str = "mean",
                 class_weight=None):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction

        if alpha is not None:
            a = torch.as_tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", a)
        else:
            self.alpha = None

        if class_weight is not None:
            cw = torch.as_tensor(class_weight, dtype=torch.float32)
            self.register_buffer("class_weight", cw)
        else:
            self.class_weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 在 FP32 中做 log_softmax，数值更稳定
        log_probs = F.log_softmax(logits.float(), dim=-1)        # (B,C)
        probs = log_probs.exp()                                  # (B,C)

        # 取出每个样本的 p_t 和 log(p_t)
        t = target.view(-1, 1)                                   # (B,1)
        log_pt = log_probs.gather(1, t).squeeze(1)               # (B,)
        pt = probs.gather(1, t).squeeze(1).clamp_(1e-8, 1.0)     # (B,)

        # (1 - p_t)^gamma
        focal = (1.0 - pt).pow(self.gamma)

        # 组合 alpha 与 class_weight（如果提供）
        if self.alpha is not None:
            a = self.alpha.to(logits.device).gather(0, target)   # (B,)
            focal = focal * a
        if self.class_weight is not None:
            cw = self.class_weight.to(logits.device).gather(0, target)  # (B,)
            focal = focal * cw

        loss = - focal * log_pt                                   # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
