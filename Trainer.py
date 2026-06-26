# Trainer.py
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import json, os
import numpy as np  # predict 里会用到
import math
from collections import Counter
class Trainer():
    def __init__(self, config, processor, model,
                 device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        self.config = config
        self.processor = processor
        self.model = model.to(device)
        self.device = device

        # ====== 参数分组（与原逻辑一致）======
        def is_no_decay(name: str) -> bool:
            return ('bias' in name) or ('norm' in name.lower())

        param_groups = []
        used_ids = set()

        # 文本编码器
        text_enc = getattr(self.model.text_model, "encoder", None)
        if text_enc is not None:
            txt_named = [(n, p) for n, p in text_enc.named_parameters() if p.requires_grad]
            if txt_named:
                param_groups += [
                    {'params': [p for n, p in txt_named if not is_no_decay(n)],
                     'lr': self.config.bert_learning_rate, 'weight_decay': self.config.weight_decay},
                    {'params': [p for n, p in txt_named if is_no_decay(n)],
                     'lr': self.config.bert_learning_rate, 'weight_decay': 0.0},
                ]
                used_ids |= {id(p) for _, p in txt_named}

        # 图像主干（ResNet）
        full_resnet = getattr(self.model.img_model, "full_resnet", None)
        if full_resnet is not None:
            img_named = [(n, p) for n, p in full_resnet.named_parameters() if p.requires_grad]
            if img_named:
                param_groups += [
                    {'params': [p for n, p in img_named if not is_no_decay(n)],
                     'lr': self.config.resnet_learning_rate, 'weight_decay': self.config.weight_decay},
                    {'params': [p for n, p in img_named if is_no_decay(n)],
                     'lr': self.config.resnet_learning_rate, 'weight_decay': 0.0},
                ]
                used_ids |= {id(p) for _, p in img_named}

        other_params = [p for n, p in self.model.named_parameters()
                        if p.requires_grad and id(p) not in used_ids]
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay
            })

        self.optimizer = AdamW(param_groups, lr=config.learning_rate)

        # ====== AMP：只有 CUDA 时启用 ======
        self.use_amp = bool(getattr(self.config, 'use_amp', True) and torch.cuda.is_available())
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # ====== 早停相关 ======
        # 监控指标：默认 f1_macro，更适合不均衡数据；可在 config 里覆盖
        self.es_monitor = str(getattr(self.config, 'early_stop_monitor', 'f1_macro'))
        # 模式：loss 走 min，其余走 max；也可在 config 里强制指定
        self.es_mode = str(getattr(self.config, 'early_stop_mode',
                        ('min' if 'loss' in self.es_monitor.lower() else 'max'))).lower()
        self.es_patience = int(getattr(self.config, 'early_stop_patience', 8))
        self.es_min_delta = float(getattr(self.config, 'early_stop_min_delta', 1e-4))
        self.es_best = (math.inf if self.es_mode == 'min' else -math.inf)
        self.es_bad_epochs = 0

        self._epoch = None
        self._max_epoch = None

    def set_epoch(self, cur, total):
        self._epoch, self._max_epoch = cur, total

    def _fmt_epoch(self):
        return f"[E{self._epoch}/{self._max_epoch}] " if (self._epoch is not None and self._max_epoch is not None) else ""

    # ================= 新增：统一解包八元组 =================
    def _unpack_batch(self, batch):
        """
        collate_fn 现在总是返回 8 项：
        (guids, texts, texts_mask, imgs, labels, rois, tok_embeds, tok_lengths)
        老数据也向后兼容。
        """
        if len(batch) == 8:
            return batch
        elif len(batch) == 6:
            guids, texts, texts_mask, imgs, labels, rois = batch
            return guids, texts, texts_mask, imgs, labels, rois, None, None
        elif len(batch) == 5:
            guids, texts, texts_mask, imgs, labels = batch
            return guids, texts, texts_mask, imgs, labels, None, None, None
        else:
            guids, texts, texts_mask, imgs, labels = batch
            return guids, texts, texts_mask, imgs, labels, None, None, None

    def _step_batch(self, batch, train=True):
        if batch is None:
            return None
        guids, texts, texts_mask, imgs, labels, rois, tok_embeds, tok_lengths = self._unpack_batch(batch)

        texts = texts.to(self.device, non_blocking=True)
        texts_mask = texts_mask.to(self.device, non_blocking=True)
        imgs = imgs.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        labels = labels.to(self.device, non_blocking=True)
        rois = rois.to(self.device, non_blocking=True) if rois is not None else None
        tok_embeds = tok_embeds.to(self.device, non_blocking=True) if tok_embeds is not None else None
        tok_lengths = tok_lengths.to(self.device, non_blocking=True) if tok_lengths is not None else None

        if train and self.use_amp:
            with torch.cuda.amp.autocast():
                try:
                    pred, loss = self.model(
                        texts, texts_mask, imgs, labels=labels,
                        roi_vec=rois, token_embeds=tok_embeds, token_lengths=tok_lengths
                    )
                except TypeError:
                    pred, loss = self.model(texts, texts_mask, imgs, labels=labels)
        else:
            try:
                pred, loss = self.model(
                    texts, texts_mask, imgs, labels=labels,
                    roi_vec=rois, token_embeds=tok_embeds, token_lengths=tok_lengths
                )
            except TypeError:
                pred, loss = self.model(texts, texts_mask, imgs, labels=labels)

        return pred, loss, labels

    def _accumulate(self, running, pred, loss, labels):
        bs = labels.size(0)
        running['loss'] += float(loss.item()) * bs
        running['correct'] += (pred == labels).sum().item()
        running['count'] += bs
        running['y_true'].extend(labels.tolist())
        running['y_pred'].extend(pred.tolist())

    def _summarize_epoch(self, split, running):
        y_true, y_pred = running['y_true'], running['y_pred']
        loss = running['loss'] / max(1, running['count'])
        acc = running['correct'] / max(1, running['count'])

        # === 关键：用 labelvocab 取正类 id，默认回退到 1 ===
        try:
            pos_idx = self.processor.labelvocab.label_to_id('positive')
            if pos_idx is None:
                pos_idx = 1
        except Exception:
            pos_idx = 1

        # Binary（以指定正类为准）
        p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=pos_idx, zero_division=0
        )
        # Macro
        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        metrics = {
            "loss": loss, "acc": acc,
            "precision": p_bin, "recall": r_bin, "f1": f1_bin,  # binary
            "precision_macro": p_mac, "recall_macro": r_mac, "f1_macro": f1_mac  # macro
        }

        # （可选）如果你 processor.metric 还有自定义指标
        try:
            extra = self.processor.metric(y_true, y_pred)
            if isinstance(extra, dict):
                metrics.update(extra)
            elif isinstance(extra, (int, float)):
                metrics["processor_metric"] = float(extra)
        except Exception:
            pass

        print(f"{self._fmt_epoch()}[{split}] loss={loss:.4f} acc={acc:.4f} "
              f"| Bin(P/R/F1)={p_bin:.4f}/{r_bin:.4f}/{f1_bin:.4f} "
              f"| MacroF1={f1_mac:.4f}")
        print(f"{self._fmt_epoch()}[{split}] metrics:\n" +
              json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2))

        print("true:", Counter(y_true), "pred:", Counter(y_pred))
        return loss, metrics

    # ============== 新增：早停状态更新 ==============
    def update_early_stopping(self, val_metrics):
        """
        在 epoch 结束后调用：
          improved, should_stop = trainer.update_early_stopping(v_metrics)
        """
        # 从 val_metrics 里取监控值
        cur = None
        if self.es_monitor in val_metrics:
            cur = float(val_metrics[self.es_monitor])
        else:
            # 兜底优先顺序
            for k in ['f1_macro', 'f1', 'acc', 'loss']:
                if k in val_metrics:
                    cur = float(val_metrics[k])
                    break
        if cur is None or math.isnan(cur):
            # 指标不可用则不早停
            return False, False

        improved = False
        if self.es_mode == 'max':
            if cur > self.es_best + self.es_min_delta:
                improved = True
                self.es_best = cur
                self.es_bad_epochs = 0
            else:
                self.es_bad_epochs += 1
        else:  # min
            if cur < self.es_best - self.es_min_delta:
                improved = True
                self.es_best = cur
                self.es_bad_epochs = 0
            else:
                self.es_bad_epochs += 1

        print(f"{self._fmt_epoch()}[EarlyStop] monitor={self.es_monitor} mode={self.es_mode} "
              f"best={self.es_best:.6f} bad_epochs={self.es_bad_epochs}/{self.es_patience}")

        should_stop = self.es_bad_epochs >= self.es_patience
        return improved, should_stop

    # ============== 带实时 tqdm 的训练与验证 ==============
    def train(self, train_loader):
        self.model.train()
        running = {'loss':0.0,'correct':0,'count':0,'y_true':[],'y_pred':[]}
        pbar = tqdm(train_loader, desc='----- [Training] ', ncols=100, leave=False)
        for batch in pbar:
            step = self._step_batch(batch, train=True)
            if step is None:
                continue
            pred, loss, labels = step

            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                clip_grad_norm_(self.model.parameters(), max_norm=getattr(self.config, 'max_grad_norm', 1.0))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=getattr(self.config, 'max_grad_norm', 1.0))
                self.optimizer.step()

            self._accumulate(running, pred, loss, labels)

            # 实时显示“累计均值”，更稳定
            avg_loss = running['loss'] / max(1, running['count'])
            avg_acc  = running['correct'] / max(1, running['count'])
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
        pbar.close()
        return self._summarize_epoch("Train", running)

    @torch.no_grad()
    def valid(self, val_loader):
        self.model.eval()
        running = {'loss':0.0,'correct':0,'count':0,'y_true':[],'y_pred':[]}
        pbar = tqdm(val_loader, desc='----- [Validating] ', ncols=100, leave=False)
        for batch in pbar:
            step = self._step_batch(batch, train=False)
            if step is None:
                continue
            pred, loss, labels = step
            self._accumulate(running, pred, loss, labels)
            avg_loss = running['loss'] / max(1, running['count'])
            avg_acc  = running['correct'] / max(1, running['count'])
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
        pbar.close()
        return self._summarize_epoch("Valid", running)

    @torch.no_grad()
    def predict(self, test_loader, output_path):
        import os, json
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from tqdm import tqdm
        from sklearn.metrics import (
            roc_curve, auc, precision_recall_curve, average_precision_score,
            classification_report, precision_recall_fscore_support, confusion_matrix,
            roc_auc_score
        )
        from sklearn.preprocessing import label_binarize

        os.makedirs(output_path, exist_ok=True)
        self.model.eval()

        y_true, y_pred = [], []
        prob_chunks = []

        # ===== 推理 =====
        with torch.inference_mode():
            for batch in tqdm(test_loader, desc='----- [Predicting] ', ncols=100, leave=False):
                guids, texts, texts_mask, imgs, labels, rois, tok_embeds, tok_lengths = self._unpack_batch(batch)

                texts = texts.to(self.device, non_blocking=True)
                texts_mask = texts_mask.to(self.device, non_blocking=True)
                imgs = imgs.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                labels = labels.to(self.device, non_blocking=True)
                rois = rois.to(self.device, non_blocking=True) if rois is not None else None
                tok_embeds = tok_embeds.to(self.device, non_blocking=True) if tok_embeds is not None else None
                tok_lengths = tok_lengths.to(self.device, non_blocking=True) if tok_lengths is not None else None

                # 文本分支（优先用预计算）
                try:
                    H_sent, S_doc, mask_sent = self.model.text_model(
                        texts, texts_mask, token_embeds=tok_embeds, token_lengths=tok_lengths
                    )
                except TypeError:
                    H_sent, S_doc, mask_sent = self.model.text_model(texts, texts_mask)

                # 图像分支
                P, r_img = self.model.img_model(imgs)
                if rois is not None and hasattr(self.model.img_model,
                                                "proj_roi") and self.model.img_model.proj_roi is not None:
                    r = self.model.img_model.act(self.model.img_model.proj_roi(rois))
                else:
                    r = r_img

                # 融合 + 分类头
                c = self.model.align(H_sent, mask_sent, P)
                O = self.model.lowrank(c, r)
                rho4, _ = self.model.hdn(V=r, O=O, S=S_doc)
                logits = self.model.cls(self.model.dropout(rho4))  # (B,) or (B,1) or (B,C)

                # === 统一得到概率矩阵 probs: (B,2)/(B,C) ===
                if logits.dim() == 1 or logits.size(-1) == 1:
                    # 单输出：视作正类 logit，过 sigmoid，再拼两列
                    p_pos = torch.sigmoid(logits.view(-1))
                    probs = torch.stack([1.0 - p_pos, p_pos], dim=1)
                else:
                    probs = torch.softmax(logits, dim=-1)

                pred = torch.argmax(probs, dim=-1)

                y_true.extend(labels.tolist())
                y_pred.extend(pred.tolist())
                prob_chunks.append(probs.detach().cpu().numpy())

        if len(y_true) == 0:
            print("[Predict] 空数据，跳过可视化。")
            return {}

        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)
        y_prob = np.vstack(prob_chunks)  # (N, C?)

        # === 再次兜底：把 y_prob 规整为 >= 2 列 ===
        if y_prob.ndim == 1:
            y_prob = y_prob.reshape(-1, 1)
        if y_prob.shape[1] == 1:
            p_pos = y_prob[:, 0].astype(np.float64)
            if not np.all((p_pos >= 0.0) & (p_pos <= 1.0)):  # 不是概率就当 logit 过 sigmoid
                p_pos = 1.0 / (1.0 + np.exp(-p_pos))
            p_pos = np.clip(p_pos, 0.0, 1.0)
            y_prob = np.column_stack([1.0 - p_pos, p_pos])

        n_classes = y_prob.shape[1]
        classes = np.arange(n_classes)
        cls_names = [str(c) for c in classes]

        # ===== 分类报告（Argmax 阈值） =====
        report_txt = classification_report(y_true, y_pred, labels=classes, target_names=cls_names,
                                           digits=4, zero_division=0)
        with open(os.path.join(output_path, "classification_report@argmax.txt"), "w", encoding="utf-8") as f:
            f.write(report_txt + "\n")
        report_dict = classification_report(y_true, y_pred, labels=classes, target_names=cls_names,
                                            digits=4, zero_division=0, output_dict=True)
        with open(os.path.join(output_path, "classification_report@argmax.json"), "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)

        # ===== 混淆矩阵 =====
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=cls_names, yticklabels=cls_names,
                    annot_kws={'size': 20}, ax=ax)
        ax.set_xlabel('Predicted', fontsize=13)
        ax.set_ylabel('True', fontsize=13)
        ax.set_title('Confusion Matrix (Counts)', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(output_path, "matrix_counts.png"), dpi=600)
        plt.close(fig)

        cm_norm = confusion_matrix(y_true, y_pred, labels=classes, normalize='true')
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                    xticklabels=cls_names, yticklabels=cls_names,
                    annot_kws={'size': 20}, ax=ax)
        ax.set_xlabel('Predicted', fontsize=13);
        ax.set_ylabel('True', fontsize=13)
        ax.set_title('Confusion Matrix (Normalized)', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(output_path, "matrix_norm.png"), dpi=600)
        plt.close(fig)

        # ===== ROC / PR（稳健宏曲线 + 二分类专属图） =====
        metrics = {}

        # ---------- 二分类：保留阈值/二元指标 ----------
        if n_classes == 2:
            p1 = y_prob[:, 1]
            # 概率分布诊断
            stats = {
                "p1_min": float(np.min(p1)),
                "p1_max": float(np.max(p1)),
                "p1_mean": float(np.mean(p1)),
                "p1_std": float(np.std(p1)),
            }
            with open(os.path.join(output_path, "prob_stats.json"), "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            if stats["p1_std"] < 1e-3:
                print("[Warn] 正类概率几乎常数（std<1e-3）。AUC/AP 可能接近随机。")

            # 二元 AUC/AP
            auc_bin = roc_auc_score(y_true, p1)
            ap_bin = average_precision_score(y_true, p1)

            # per-class AUC/AP（若某类在 y_true 中缺失会跳过）
            y0 = (y_true == 0).astype(int)
            y1 = (y_true == 1).astype(int)
            auc0 = roc_auc_score(y0, y_prob[:, 0]) if (y0.min() == 0 and y0.max() == 1) else np.nan
            auc1 = roc_auc_score(y1, y_prob[:, 1]) if (y1.min() == 0 and y1.max() == 1) else np.nan
            ap0 = average_precision_score(y0, y_prob[:, 0]) if y0.sum() > 0 else np.nan
            ap1 = average_precision_score(y1, y_prob[:, 1]) if y1.sum() > 0 else np.nan
            auc_macro = np.nanmean([auc0, auc1])
            ap_macro = np.nanmean([ap0, ap1])

            # per-class ROC
            if not np.isnan(auc0) and not np.isnan(auc1):
                fpr0, tpr0, _ = roc_curve(y0, y_prob[:, 0])
                fpr1, tpr1, _ = roc_curve(y1, y_prob[:, 1])
                plt.figure(figsize=(8, 6))
                plt.plot(fpr0, tpr0, lw=2, label=f'Class 0 (AUC={auc0:.3f})')
                plt.plot(fpr1, tpr1, lw=2, label=f'Class 1 (AUC={auc1:.3f})')
                plt.plot([0, 1], [0, 1],  color='gray', lw=1)
                plt.xlim([0.0, 1.0]);
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate');
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curves | binary AUC={auc_bin:.3f}, macro AUC={auc_macro:.3f}')
                plt.legend(loc='lower right', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, "roc_per_class.png"), dpi=600)
                plt.close()

            # per-class PR
            if (y0.sum() > 0) and (y1.sum() > 0):
                pr0, rc0, _ = precision_recall_curve(y0, y_prob[:, 0])
                pr1, rc1, _ = precision_recall_curve(y1, y_prob[:, 1])
                plt.figure(figsize=(8, 6))
                plt.plot(rc0, pr0, lw=2, label=f'Class 0 (AP={ap0:.3f})')
                plt.plot(rc1, pr1, lw=2, label=f'Class 1 (AP={ap1:.3f})')
                plt.xlim([0.0, 1.0]);
                plt.ylim([0.0, 1.05])
                plt.xlabel('Recall');
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall | binary AP={ap_bin:.3f}, macro AP={ap_macro:.3f}')
                plt.legend(loc='lower left', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, "pr_per_class.png"), dpi=600)
                plt.close()

            # 阈值搜索（可选）
            def _find_best_t(y, p):
                cand = np.unique(np.round(p, 6))
                if len(cand) > 2000:
                    cand = np.quantile(p, np.linspace(0, 1, 2001))
                best_t, best_m = 0.5, -1.0
                for t in cand:
                    yp = (p >= t).astype(np.int32)
                    m = precision_recall_fscore_support(y, yp, average='macro', zero_division=0)[2]
                    if m > best_m:
                        best_m, best_t = m, t
                return float(best_t), float(best_m)

            best_t, best_macro_f1 = _find_best_t(y_true, p1)
            y_pred_opt = (p1 >= best_t).astype(np.int32)
            rep_opt = classification_report(y_true, y_pred_opt, labels=[0, 1], target_names=['0', '1'],
                                            digits=4, zero_division=0)
            with open(os.path.join(output_path, "classification_report@best_threshold.txt"), "w",
                      encoding="utf-8") as f:
                f.write(f"[best_threshold] = {best_t:.6f}\n\n{rep_opt}\n")

            # 阈值=0.5 的 P/R/F1
            p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
                y_true, (p1 >= 0.5).astype(np.int32), labels=[0, 1], average='binary', zero_division=0
            )

            metrics.update({
                'precision': float(p_bin), 'recall': float(r_bin), 'f1': float(f1_bin),
                'auc_binary': float(auc_bin), 'ap_binary': float(ap_bin),
                'roc_auc_class0': float(auc0), 'roc_auc_class1': float(auc1), 'roc_auc_macro': float(auc_macro),
                'ap_class0': float(ap0), 'ap_class1': float(ap1), 'ap_macro': float(ap_macro),
                'best_threshold': float(best_t), 'best_macro_f1': float(best_macro_f1),
                'p1_min': float(stats["p1_min"]), 'p1_max': float(stats["p1_max"]),
                'p1_mean': float(stats["p1_mean"]), 'p1_std': float(stats["p1_std"]),
            })

        # ---------- 宏 ROC/PR（适配多分类；也适用于二分类） ----------
        # 注意：label_binarize 在二分类时**只返回 1 列**，需要手动扩展为两列以对齐 y_prob 的两列
        Y_bin = label_binarize(y_true, classes=classes)  # (N, 1 or N, C)
        if Y_bin.ndim == 1:
            Y_bin = Y_bin.reshape(-1, 1)
        if n_classes == 2 and Y_bin.shape[1] == 1:
            y_pos = (y_true == classes[1]).astype(np.int32)  # 以 classes[1] 作为“正类”
            Y_bin = np.column_stack([1 - y_pos, y_pos])  # (N,2) -> [neg, pos]

        # 仅选择“正负样本都出现”的有效类别
        valid = []
        per_fpr, per_tpr = [], []
        per_prec, per_rec, per_ap = [], [], []
        for i in range(Y_bin.shape[1]):  # 用 Y_bin 的列数更稳妥
            yi = Y_bin[:, i]
            if yi.min() == yi.max():  # 全 0 或全 1 无法画曲线
                continue
            try:
                fpr_i, tpr_i, _ = roc_curve(yi, y_prob[:, i])
                pr_i, rc_i, _ = precision_recall_curve(yi, y_prob[:, i])
                ap_i = average_precision_score(yi, y_prob[:, i])

                per_fpr.append(fpr_i)
                per_tpr.append(tpr_i)
                per_prec.append(pr_i)
                per_rec.append(rc_i)
                per_ap.append(ap_i)
                valid.append(i)
            except Exception:
                continue

        # 宏 ROC
        if len(valid) > 0:
            all_fpr = np.unique(np.concatenate(per_fpr))
            mean_tpr = np.zeros_like(all_fpr)
            for fpr_i, tpr_i in zip(per_fpr, per_tpr):
                mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
            mean_tpr /= len(per_fpr)
            macro_auc_curve = auc(all_fpr, mean_tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(all_fpr, mean_tpr, lw=3, label=f"Macro ROC (AUC={macro_auc_curve:.2f})")
            plt.fill_between(all_fpr, mean_tpr, 0, alpha=0.5, color='lightblue', edgecolor="none")
            plt.plot([0, 1], [0, 1], color='gray', lw=2, label='Chance (AUC=0.50)')
            plt.xlim([0.0, 1.0]);
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Macro-average ROC")
            plt.legend(loc="lower right",fontsize=15)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "roc_macro.png"), dpi=600)
            plt.close()

            metrics['roc_auc_macro_curve'] = float(macro_auc_curve)
        else:
            metrics['roc_auc_macro_curve'] = float('nan')

        # 宏 PR
        if len(valid) > 0:
            grid = np.linspace(0, 1, 500)
            mean_prec = np.zeros_like(grid)
            for pr_i, rc_i in zip(per_prec, per_rec):
                mean_prec += np.interp(grid, rc_i, pr_i)
            mean_prec /= len(per_prec)
            macro_ap_curve = float(np.nanmean(per_ap)) if len(per_ap) > 0 else float('nan')

            pos_rate = Y_bin[:, valid].mean() if len(valid) > 0 else Y_bin.mean()
            baseline = np.full_like(grid, pos_rate)

            plt.figure(figsize=(8, 6))
            plt.plot(grid, mean_prec, lw=3, label=f"Macro PR (AP={macro_ap_curve:.2f})")
            above = mean_prec >= baseline
            plt.fill_between(
                grid, mean_prec, baseline,
                where=above, interpolate=True,
                alpha=0.5,color='lightblue', edgecolor="none"
            )

            plt.hlines(pos_rate, 0, 1, color='gray', lw=2, label=f"Chance (pos={pos_rate:.2f})")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Macro-average Precision–Recall")
            plt.legend(loc="lower left",fontsize=15)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "pr_macro.png"), dpi=600)
            plt.close()

            metrics['ap_macro_curve'] = float(macro_ap_curve)
        else:
            metrics['ap_macro_curve'] = float('nan')

        # ===== 汇总指标（写文件） =====
        if n_classes == 2:
            # 保持/补充关键指标
            metrics.setdefault('roc_auc_macro', float(np.nanmean([
                metrics.get('roc_auc_class0', np.nan),
                metrics.get('roc_auc_class1', np.nan)
            ])))
            metrics.setdefault('ap_macro', float(np.nanmean([
                metrics.get('ap_class0', np.nan),
                metrics.get('ap_class1', np.nan)
            ])))
        else:
            # 多分类：返回宏 AUC / 宏 AP（基于 one-vs-rest）
            try:
                metrics['roc_auc_macro'] = float(
                    roc_auc_score(label_binarize(y_true, classes=classes), y_prob, average="macro"))
            except Exception:
                metrics['roc_auc_macro'] = float('nan')
            try:
                metrics['ap_macro'] = float(
                    average_precision_score(label_binarize(y_true, classes=classes), y_prob, average="macro"))
            except Exception:
                metrics['ap_macro'] = float('nan')

        print(f"[Predict] 输出宏曲线：roc_macro.png, pr_macro.png 已保存到 {output_path}")
        with open(os.path.join(output_path, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        return metrics

