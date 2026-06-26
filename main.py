# main.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('./utils')
sys.path.append('./utils/APIs')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
import random

from torch.utils.data import DataLoader, WeightedRandomSampler

from Config import config
from utils.common import (
    data_format, read_from_file, save_model, train_val_split, _label_to_id
)
from utils.DataProcess import Processor
from Trainer import Trainer
from Models.model import Model
# from Models.CMA import Model


# ------------------ 性能/复现相关 ------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------ 参数 ------------------
def init_argparse():
    parser = argparse.ArgumentParser()
    # 运行模式
    parser.add_argument('--do_train', default=True,action='store_true', help='训练模型')
    parser.add_argument('--do_test',default=True, action='store_true', help='评测测试集（需先有最优模型）')

    # 采样器（只作用于训练集）
    parser.add_argument('--sampler', type=str, default='balanced', choices=['none', 'balanced'],
                        help="训练集采样策略：none=不采样；balanced=按类别逆频率加权采样")

    # 训练/评测常规
    parser.add_argument('--text_pretrained_model', default='bert-base-chinese', type=str)
    parser.add_argument('--fuse_model_type', default='Model', type=str)
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=5e-2, type=float)
    parser.add_argument('--epoch', default=70, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--load_model_path', default=None, type=str)
    parser.add_argument('--text_only', action='store_true')
    parser.add_argument('--img_only', action='store_true')

    # ==== Faster R-CNN 相关（用于数据加载阶段提 ROI）====
    parser.add_argument('--use_frcnn_regions', action='store_true', default=True,
                        help='在数据加载阶段用 Faster R-CNN 提取 ROI 向量')
    parser.add_argument('--frcnn_topk', type=int, default=16,
                        help='每图取前 top-k proposals 聚合 ROI')
    parser.add_argument('--frcnn_device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='FRCNN 推理所用设备')
    parser.add_argument('--augment_hflip', action='store_true',
                        help='与 ROI 同步的随机水平翻转（建议开）')
    return parser


# ============== 新增：测试集序列化/读取工具 ==============
import os
import json
import numbers
import numpy as np
from pathlib import Path
from PIL import Image

def _to_py(v):
    """把 numpy / Path / 非原生对象安全转换为 json 友好的 Python 基本类型"""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, Path):
        return str(v)
    if v is None or isinstance(v, (str, bytes, bool, int, float)):
        return v
    if isinstance(v, (list, tuple)):
        return [_to_py(x) for x in v]
    if isinstance(v, dict):
        return {str(_to_py(k)): _to_py(val) for k, val in v.items()}
    return str(v)


def _ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _normalize_img_field(guid, img, data_root_dir):
    """
    返回 (img_str, img_abs_path)：
    - img_str：写回 JSON 的 `img` 字段（字符串；可为绝对路径）
    - img_abs_path：写回 JSON 的 `img_path` 字段（绝对路径，便于调试/兜底）
    规则：
      * 若 img 是 str：
          - 绝对路径：直接使用
          - 相对路径：认为相对 data_root_dir，转为绝对
      * 若 img 是 PIL.Image.Image：
          - 若带 filename 且存在：用其绝对路径
          - 否则把图另存到 data_root_dir/<guid>.jpg
      * 其它类型：转成字符串；若不是绝对路径，则不构造 abs（留空）
    """
    _ensure_dir(data_root_dir)

    # case 1: 字符串
    if isinstance(img, str):
        if os.path.isabs(img):
            return img, img
        else:
            abs_path = os.path.join(data_root_dir, img)
            return abs_path, abs_path

    # case 2: PIL.Image.Image
    if isinstance(img, Image.Image):
        # 如果原始 image 有 filename 且文件存在，直接用
        fn = getattr(img, 'filename', None)
        if isinstance(fn, str) and fn and os.path.isfile(fn):
            absp = os.path.abspath(fn)
            return absp, absp
        # 否则将其另存为 data_root_dir/<guid>.jpg
        save_path = os.path.join(data_root_dir, f"{guid}.jpg")
        try:
            # 确保是 RGB
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            img.save(save_path, format='JPEG', quality=95)
            absp = os.path.abspath(save_path)
            return absp, absp
        except Exception as e:
            # 保存失败则退化：给一个纯字符串描述，img_path 置空
            desc = f"{guid}.jpg"
            return desc, ""

    # case 3: 其它类型
    s = _to_py(img)
    if isinstance(s, str) and os.path.isabs(s):
        return s, s
    # 非绝对路径就仅返回字符串，img_path 留空
    return str(s), ""


def _serialize_test_item(item, data_root_dir):
    """
    item 结构为 [guid, text, img, label]
    把非 JSON 友好类型全部规整，并确保 img 最终可用于 Dataset（绝对路径更稳）。
    """
    guid  = str(_to_py(item[0]))
    text  = _to_py(item[1])
    raw_img = item[2]
    label = _to_py(item[3] if len(item) > 3 else item[-1])

    img_str, img_abs = _normalize_img_field(guid, raw_img, data_root_dir)

    return {
        "guid": guid,
        "text": text,
        "img": img_str,      # 允许是绝对路径；Dataset 若再 join(root, abs) 也没问题（abs 会覆盖前缀）
        "label": label,
        "img_path": img_abs  # 调试/兜底字段
    }


def save_test_data_full(test_data, save_path, data_root_dir):
    """
    将完整测试集样本写到 save_path（JSON 数组），每条包含 guid/text/img/label（以及 img_path）。
    """
    _ensure_dir(os.path.dirname(save_path))
    payload = [_serialize_test_item(x, data_root_dir) for x in test_data]
    payload = _to_py(payload)  # 再整体规整一遍，防止残留 numpy 类型
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Split] Saved FULL test dataset to {save_path} (n={len(payload)})")


def load_test_data_full(load_path):
    """
    读取由 save_test_data_full 写出的 JSON，并转换回 Processor 可消费的 list[item] 结构：
      item = [guid, text, img, label]
    这里优先使用绝对的 img_path；否则使用 img 字段。
    """
    with open(load_path, "r", encoding="utf-8") as f:
        arr = json.load(f)

    restored = []
    for d in arr:
        guid  = str(d.get("guid", ""))
        text  = d.get("text", "")
        img_p = d.get("img_path", "")  # 优先绝对路径
        img   = img_p if (isinstance(img_p, str) and img_p) else d.get("img", "")

        label = d.get("label", 0)
        try:
            # 把数字字符串/np 数字转回 int
            if isinstance(label, str) and label.isdigit():
                label = int(label)
            elif isinstance(label, numbers.Number):
                label = int(label)
        except Exception:
            pass

        restored.append([guid, text, img, label])
    return restored

def main(args):
    # 固定随机种子
    set_seed(args.seed)

    # 写回 config（注意 text_backbone 字段）
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.epoch = args.epoch
    config.text_backbone = args.text_pretrained_model   # ← 模型读取的是 text_backbone
    config.bert_name = args.text_pretrained_model       # ← 仅用于打印兼容
    config.fuse_model_type = args.fuse_model_type
    config.load_model_path = args.load_model_path

    config.only = 'img' if args.img_only else None
    config.only = 'text' if args.text_only else config.only

    # ==== FRCNN 设定注入到 config（Processor/Dataset 会读取）====
    config.use_frcnn_regions = args.use_frcnn_regions
    config.frcnn_topk = args.frcnn_topk
    config.frcnn_device = args.frcnn_device
    config.augment_hflip = args.augment_hflip

    # 与模型图像归一化空间保持一致（数据侧默认用 ImageNet 归一化即可）
    config.image_input_space = getattr(config, "image_input_space", "imagenet")

    print('TextModel: {}, ImageModel: {}, FuseModel: {}'.format(
        config.text_backbone, config.image_backbone, config.fuse_model_type))

    # 初始化
    processor = Processor(config)
    model = Model(config)
    if torch.cuda.is_available():
        idx = 1 if torch.cuda.device_count() > 1 else 0  # 有两块以上卡就用 cuda:1，否则用 cuda:0
        device = torch.device(f"cuda:{idx}")
    else:
        device = torch.device("cpu")
    trainer = Trainer(config, processor, model, device)

    # =============== 训练 ===============
    def train():
        # 1) 准备数据（统一先生成 train.json）
        data_format(r"data/data/label.txt", r"data/data", r"data/train.json")
        all_data = read_from_file(r"data/train.json", r"data/data", config.only)
        train_data, val_data, test_data = train_val_split(all_data)

        # 训练集类别统计（使用 label_to_id 而不是 int(y)）
        def _lab(x):
            return _label_to_id(x[3] if len(x) > 3 else x[-1])

        y_train = [_lab(item) for item in train_data]
        cnt = Counter(y_train)
        print(f"[class dist] train={cnt}")

        # 自动 class_weight：两类都存在才启用
        if len(cnt) == 2:
            total = sum(cnt.values())
            w_neg = total / (2.0 * cnt.get(0, 1))
            w_pos = total / (2.0 * cnt.get(1, 1))
            config.class_weight = [w_neg, w_pos]
        else:
            config.class_weight = None

        # ★ 立刻把 class_weight 生效到模型的损失函数（否则初始化时未带权重）
        if config.class_weight is not None:
            w = torch.tensor(config.class_weight, dtype=torch.float32, device=device)
            model.crit = torch.nn.CrossEntropyLoss(
                weight=w,
                label_smoothing=float(getattr(config, "label_smoothing", 0.05))
            )

        # —— 新：把“完整测试集样本”序列化到输出目录（替代旧的 data/test.json GUID 列表）——
        out_dir = os.path.join(config.output_path, config.fuse_model_type)
        os.makedirs(out_dir, exist_ok=True)
        test_full_fp = os.path.join(out_dir, "test_data.json")
        # 与 Processor 一致的图片根目录（你项目中就是 data/data）
        data_root_dir = os.path.join("data", "data")
        if not os.path.isfile(test_full_fp):
            save_test_data_full(test_data, test_full_fp, data_root_dir)
        else:
            print(f"[Split] Reusing existing FULL test: {test_full_fp}")

        # 2) DataLoader：训练集可选采样；验证不增广、不采样
        config.augment = True
        if args.sampler == 'balanced':
            train_dataset = processor.to_dataset(train_data)
            labels_np = np.array(train_dataset.labels, dtype=np.int64)
            class_counts = np.bincount(labels_np, minlength=2)
            class_weights = 1.0 / np.maximum(class_counts, 1)
            sample_weights = class_weights[labels_np]
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).double(),
                num_samples=len(labels_np),
                replacement=True
            )

            # —— 关键：清理可能重复的关键字 —— #
            dl_kwargs = dict(config.train_params)  # 复制一份，避免改到原对象
            for k in ['shuffle', 'sampler', 'batch_sampler', 'collate_fn']:
                dl_kwargs.pop(k, None)

            train_loader = DataLoader(
                dataset=train_dataset,
                sampler=sampler,  # 用 sampler 时必须禁用 shuffle
                shuffle=False,
                collate_fn=train_dataset.collate_fn,
                **dl_kwargs
            )
            print("[Train] Using WeightedRandomSampler (balanced).")
        else:
            train_loader = processor(train_data, config.train_params)

        # 验证集
        config.augment = False
        val_loader = processor(val_data, config.val_params)

        rows = []
        df = None

        for e in range(1, config.epoch + 1):
            since = time.time()
            print('-' * 20 + f' Epoch {e} ' + '-' * 20)
            trainer.set_epoch(e, config.epoch)

            t_loss, t_metrics = trainer.train(train_loader)
            v_loss, v_metrics = trainer.valid(val_loader)

            # 取数值
            t_acc = float(t_metrics.get("acc", t_metrics.get("accuracy", np.nan)))
            v_acc = float(v_metrics.get("acc", v_metrics.get("accuracy", np.nan)))
            t_f1  = float(t_metrics.get("f1",  t_metrics.get("f1_macro", np.nan)))
            v_f1  = float(v_metrics.get("f1",  v_metrics.get("f1_macro", np.nan)))

            rows.append({
                "epoch": e,
                "train_loss": float(t_loss),
                "train_acc": t_acc,
                "train_macro_f1": t_f1,
                "val_loss": float(v_loss),
                "val_acc": v_acc,
                "val_macro_f1": v_f1,
            })

            # 写 TSV（保存在输出目录下）
            df = pd.DataFrame(rows, columns=[
                "epoch", "train_loss", "train_acc", "train_macro_f1", "val_loss", "val_acc", "val_macro_f1"
            ])
            df.to_csv(os.path.join(out_dir, "results.txt"), sep="\t", index=False)

            elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(elapsed // 60, elapsed % 60))

            # === 早停逻辑（由 Trainer 维护 best 与 patience）===
            improved, stop = trainer.update_early_stopping(v_metrics)
            if improved:
                save_model(config.output_path, config.fuse_model_type, model)
                with open(os.path.join(out_dir, "early_stop_best.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "best_epoch": e,
                        "monitor": getattr(config, "early_stop_monitor", "f1_macro"),
                        "mode": getattr(config, "early_stop_mode", "max"),
                        "best_metrics": v_metrics
                    }, f, ensure_ascii=False, indent=2)
                print('Update best model!')
            if stop:
                print(f"Early stopping at epoch {e}.")
                break

        # 4) 画图（若没有 df 说明 0 轮，跳过）
        if df is not None and len(df):
            epochs = df["epoch"].to_numpy()
            train_loss = df["train_loss"].to_numpy()
            val_loss = df["val_loss"].to_numpy()
            train_acc = df["train_acc"].to_numpy()
            val_acc = df["val_acc"].to_numpy()

            # Loss
            plt.figure(figsize=(10, 8))
            plt.plot(epochs, train_loss, label='Train Loss')
            plt.plot(epochs, val_loss, label='Validation Loss')
            plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Loss Curve')
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "loss.png"), dpi=600)

            # Acc
            plt.figure(figsize=(10, 8))
            plt.plot(epochs, train_acc, label='Train Accuracy')
            plt.plot(epochs, val_acc, label='Validation Accuracy')
            plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.title('Accuracy Curve')
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "acc.png"), dpi=600)

    # =============== 测试 ===============
    def test():
        # 统一不做增广
        config.augment = False

        # 先尝试读取“完整测试集样本”JSON（训练阶段已写在输出目录）
        out_dir = os.path.join(config.output_path, config.fuse_model_type)
        test_full_fp = os.path.join(out_dir, "test_data.json")

        if os.path.isfile(test_full_fp):
            test_data = load_test_data_full(test_full_fp)
            print(f"[Split] Loaded FULL test dataset from {test_full_fp} (n={len(test_data)})")
        else:
            # 如果没有新的完整 JSON，则回退到老逻辑
            # 始终从 train.json 读取全集数据
            data_format(os.path.join(config.root_path, 'data/label.txt'),
                        os.path.join(config.root_path, 'data/data'),
                        os.path.join(config.root_path, 'data/train.json'))
            all_data = read_from_file(os.path.join(config.root_path, 'data/train.json'),
                                      os.path.join(config.root_path, 'data/data'),
                                      config.only)

            # 优先复用 data/test.json（里面存的是 guid 列表）
            test_id_file = os.path.join("data", "test.json")
            if os.path.isfile(test_id_file):
                with open(test_id_file, "r", encoding="utf-8") as f:
                    test_ids = set(json.load(f))
                test_data = [item for item in all_data if str(item[0]) in test_ids]
                print(f"[Split] Reused {len(test_data)} test samples from {test_id_file}")
            else:
                # 没有固定测试集就临时再切一次（兜底）
                _, _, test_data = train_val_split(all_data)
                print(f"[Split] No saved test set; using a fresh split with {len(test_data)} samples")

        test_loader = processor(test_data, config.test_params)

        # 加载最优模型并评估
        load_model_path = os.path.join(config.output_path, config.fuse_model_type, 'pytorch_model.bin')
        map_loc = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load(load_model_path, map_location=map_loc))

        outputs = trainer.predict(test_loader, config.output_path)
        print(outputs)

    # ====== 执行 ======
    ran_any = False
    if args.do_train or (not args.do_train and not args.do_test):
        # 默认只训练，不自动测试（避免协议泄露）
        train()
        ran_any = True
    if args.do_test:
        test()
        ran_any = True
    if not ran_any:
        print("Nothing to do. Pass --do_train and/or --do_test.")

if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()
    main(args)
