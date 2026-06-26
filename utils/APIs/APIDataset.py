# APIs/APIDataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Any, List, Tuple, Optional


class APIDataset(Dataset):
    """
    接受以下多种输入（来自 api_encode）：
      1) guids, texts, imgs, labels
      2) guids, texts, imgs, labels, rois
      3) guids, texts, imgs, labels, tok_embeds
      4) guids, texts, imgs, labels, rois, tok_embeds

    约定：
      - rois: list[Tensor(roi_dim,)]，通常 roi_dim=1024
      - tok_embeds: list[Tensor(T_i, enc_dim)]
    """

    def __init__(self, *args) -> None:
        n = len(args)
        if n == 4:
            self.guids, self.texts, self.imgs, self.labels = args
            self.rois = None
            self.tok_embeds = None

        elif n == 5:
            self.guids, self.texts, self.imgs, self.labels, extra = args
            self.rois = None
            self.tok_embeds = None
            # 允许列表中存在 None；找到第一个 Tensor 决定类型（1D=rois，2D=tok_embeds）
            if isinstance(extra, list) and len(extra) > 0:
                first_tensor = next((x for x in extra if isinstance(x, torch.Tensor)), None)
                if isinstance(first_tensor, torch.Tensor):
                    if first_tensor.dim() == 1:
                        self.rois = extra
                    elif first_tensor.dim() == 2:
                        self.tok_embeds = extra

        elif n == 6:
            self.guids, self.texts, self.imgs, self.labels, self.rois, self.tok_embeds = args

        else:
            raise ValueError(f"Unexpected number of arguments: {n}")

        self.has_rois = self.rois is not None
        self.has_tok = self.tok_embeds is not None

    def __len__(self) -> int:
        return len(self.guids)

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        item: List[Any] = [self.guids[index], self.texts[index], self.imgs[index], self.labels[index]]
        if self.has_rois:
            item.append(self.rois[index])
        if self.has_tok:
            item.append(self.tok_embeds[index])
        return tuple(item)

    def _to_chw_tensor(self, img_any: Any) -> torch.Tensor:
        """
        兜底把图像转为 (C,H,W) float32。
        encode 正常会返回 ToTensor() 后的张量，这里是容错分支。
        """
        if isinstance(img_any, torch.Tensor):
            t = img_any
            if t.dim() == 3:
                # 若是 HWC 则转为 CHW
                if t.shape[0] != 3 and t.shape[-1] == 3:
                    t = t.permute(2, 0, 1)
            elif t.dim() == 2:  # 灰度 -> 3 通道
                t = t.unsqueeze(0).repeat(3, 1, 1)
            return t.float()

        # 非张量：尽量按 numpy 处理
        arr = np.array(img_any)
        if arr.ndim == 2:  # 灰度 -> 3 通道
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 3:  # HWC -> CHW
            arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).float()

    def collate_fn(self, batch: List[Tuple[Any, ...]]):
        """
        返回统一的 8 元组：
        (guids, paded_texts, paded_texts_mask, imgs, labels, rois, tok_padded, tok_lengths)
        缺失字段用 None。
        """
        # ---------- 基本项 ----------
        guids = [b[0] for b in batch]
        texts = [torch.as_tensor(b[1], dtype=torch.long) for b in batch]

        # imgs：通常已是 Tensor(3,H,W)，此处做兜底 HWC->CHW / 灰度->3 通道
        img_list = [self._to_chw_tensor(b[2]) for b in batch]
        imgs = torch.stack(img_list, dim=0).float()

        labels = torch.as_tensor([b[3] for b in batch], dtype=torch.long)

        # ---------- 文本 padding & mask ----------
        texts_mask = [torch.ones_like(t, dtype=torch.long) for t in texts]
        paded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        paded_texts_mask = pad_sequence(texts_mask, batch_first=True, padding_value=0).gt(0)  # bool

        # ---------- ROI（可选）----------
        # 在样本的第 4/5/6... 位中，寻找第一个 1D Tensor 作为 ROI
        def pick_roi(sample: Tuple[Any, ...]) -> Optional[torch.Tensor]:
            for x in sample[4:]:
                if isinstance(x, torch.Tensor) and x.dim() == 1:
                    return x
            return None

        roi_list = [pick_roi(b) for b in batch]
        has_rois = any(r is not None for r in roi_list)
        rois = None
        if has_rois:
            # 动态确定 roi 维度（默认 1024）
            first = next((r for r in roi_list if r is not None), None)
            roi_dim = int(first.numel()) if first is not None else 1024
            rois = torch.stack(
                [(r if r is not None else torch.zeros(roi_dim)) for r in roi_list],
                dim=0
            ).float()

        # ---------- 预计算文本特征（可选）----------
        # 在样本的第 4/5/6... 位中，寻找第一个 2D Tensor 作为 tok_embeds
        def pick_tok(sample: Tuple[Any, ...]) -> Optional[torch.Tensor]:
            for x in sample[4:]:
                if isinstance(x, torch.Tensor) and x.dim() == 2:
                    return x
            return None

        tok_list = [pick_tok(b) for b in batch]
        has_tok = any(t is not None for t in tok_list)
        tok_padded = None
        tok_lengths = None
        if has_tok:
            # 动态确定 enc_dim & T_max
            first_tok = next((t for t in tok_list if t is not None), None)
            enc_dim = int(first_tok.size(-1)) if first_tok is not None else 1
            T_max = max((t.size(0) if t is not None else 0) for t in tok_list)
            if T_max <= 0:
                T_max = 1

            tok_padded = torch.zeros(len(tok_list), T_max, enc_dim, dtype=torch.float32)
            lengths: List[int] = []
            for i, t in enumerate(tok_list):
                if t is None:
                    lengths.append(0)
                    continue
                L = t.size(0)
                tok_padded[i, :L] = t
                lengths.append(L)
            tok_lengths = torch.as_tensor(lengths, dtype=torch.long)

        # 统一返回 8 个元素
        return (
            guids,             # List[str]
            paded_texts,       # LongTensor (B, T_max)
            paded_texts_mask,  # BoolTensor (B, T_max)
            imgs,              # FloatTensor (B, 3, H, W)
            labels,            # LongTensor (B,)
            rois,              # FloatTensor (B, roi_dim) or None
            tok_padded,        # FloatTensor (B, T_max_tok, enc_dim) or None
            tok_lengths        # LongTensor (B,) or None
        )
