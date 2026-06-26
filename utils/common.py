'''
普通的常用工具
'''

import os
import json
import chardet
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import os, csv, json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import chardet

# 多编码智能解码
def smart_decode(raw: bytes) -> str:
    enc_guess = chardet.detect(raw).get("encoding")
    tried = set()
    for enc in [enc_guess, "utf-8", "utf-8-sig", "gb18030", "gbk", "big5", "cp1252", "latin1"]:
        if not enc:
            continue
        enc = enc.lower()
        if enc in tried:
            continue
        tried.add(enc)
        try:
            return raw.decode(enc)
        except Exception:
            pass
    # 最后兜底：不再报错
    return raw.decode("utf-8", errors="replace")

# 将文本和标签格式化成一个 json（避免 “error 3190” 一类解码异常）
def data_format(input_path, data_dir, output_path):
    data_dir = Path(data_dir)
    assert data_dir.exists(), f"文本目录不存在：{data_dir}"

    # 读标签文件（csv: guid,label 或 guid,tag）
    raw = Path(input_path).read_bytes()
    text = smart_decode(raw).replace("\r\n", "\n")
    rows = list(csv.reader(text.splitlines(), skipinitialspace=True))
    if not rows:
        raise ValueError("标签文件为空")

    header = [ (h or "").strip().lower() for h in rows[0] ]
    try:
        gid_idx = header.index("guid")
    except ValueError:
        gid_idx = header.index("id")
    lab_idx = None
    for k in ("label", "tag"):
        if k in header:
            lab_idx = header.index(k)
            break
    if lab_idx is None:
        raise ValueError(f"未找到 label/tag 列，表头为：{header}")

    out = []
    drop_no_text = drop_empty_text = 0

    for r in tqdm(rows[1:], desc='----- [Formating]'):
        if not r or len(r) <= max(gid_idx, lab_idx):
            continue
        guid = str(r[gid_idx]).strip()
        if not guid or guid.lower() == "guid":
            continue
        label = str(r[lab_idx]).strip()

        txt_path = data_dir / f"{guid}.txt"
        if not txt_path.exists():
            drop_no_text += 1
            # print("missing", guid)
            continue

        raw_txt = txt_path.read_bytes()
        text = smart_decode(raw_txt).strip()
        if not text:
            drop_empty_text += 1
            # print("empty", guid)
            continue

        out.append({"guid": guid, "label": label, "text": text})

    with open(output_path, "w", encoding="utf-8") as wf:
        json.dump(out, wf, ensure_ascii=False, indent=2)

    print(f"[OK] 写入 {output_path}")
    print(f"[STAT] 成功: {len(out)} | 无文本: {drop_no_text} | 空文本: {drop_empty_text}")

# 读取数据，返回 [(guid, text, img, label)]，对 JSON/图片都做容错
def read_from_file(path, data_dir, only=None):
    data_dir = Path(data_dir)

    # JSON 多编码容错
    try:
        with open(path, "r", encoding="utf-8") as f:
            json_file = json.load(f)
    except UnicodeDecodeError:
        raw = Path(path).read_bytes()
        json_file = json.loads(smart_decode(raw))

    data = []
    for d in tqdm(json_file, desc='----- [Loading]'):
        guid = str(d.get('guid', '')).strip()
        label = d.get('label', '')
        text  = d.get('text', '')

        if not guid or guid.lower() == 'guid':
            continue

        # 图片
        if only == 'text':
            img = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))
        else:
            img_path = data_dir / f"{guid}.jpg"
            try:
                img = Image.open(img_path)
                img = img.convert("RGB")
                img.load()
            except Exception:
                # 找不到或损坏 -> 占位图（也可选择 continue 跳过）
                img = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))

        # 只图模式
        if only == 'img':
            text = ''

        data.append((guid, text, img, label))

    return data

# 分离训练集和验证集
from sklearn.model_selection import train_test_split
from collections import Counter

def _label_to_id(lbl):
    """把各种写法统一成 0/1。未知/异常时默认 0。"""
    s = str(lbl).strip().lower()
    if s in {"1", "pos", "positive", "true", "yes"}:
        return 1
    if s in {"0", "neg", "negative", "false", "no"}:
        return 0
    try:
        return 1 if int(s) == 1 else 0
    except Exception:
        return 0

def train_val_split(data, val_ratio=0.2, test_ratio=0.1, seed=42, stratify=True):
    """
    返回: train, val, test
    - 支持标签为字符串('negative'/'positive')或数字
    - 类别极端不平衡或只有单类时，自动退化为非分层切分（避免 sklearn 报错）
    """
    if not data:
        return [], [], []

    # 索引与标签
    X = list(range(len(data)))
    # 标签通常在 item[3]；若某些数据结构不同，退而取最后一个
    y = [_label_to_id(item[3] if len(item) > 3 else item[-1]) for item in data]

    cnt = Counter(y)
    # 分层切分要求每个被切分的集合里每个类至少有 2 个样本
    use_strat = stratify and len(cnt) >= 2 and min(cnt.values()) >= 2

    # 先拆出 test
    X_tmp, X_test, _, _ = train_test_split(
        X, y,
        test_size=test_ratio,
        stratify=y if use_strat else None,
        random_state=seed
    )
    y_tmp = [y[i] for i in X_tmp]
    cnt_tmp = Counter(y_tmp)
    use_strat2 = use_strat and len(cnt_tmp) >= 2 and min(cnt_tmp.values()) >= 2

    # 再从剩余里拆出 val
    X_train, X_val, _, _ = train_test_split(
        X_tmp, y_tmp,
        test_size=val_ratio / (1.0 - test_ratio),
        stratify=y_tmp if use_strat2 else None,
        random_state=seed
    )

    train = [data[i] for i in X_train]
    val   = [data[i] for i in X_val]
    test  = [data[i] for i in X_test]
    return train, val, test


# 写入数据
def write_to_file(path, outputs):
    with open(path, 'w') as f:
        for line in tqdm(outputs, desc='----- [Writing]'):
            f.write(line)
            f.write('\n')
        f.close()


# 保存模型
def save_model(output_path, model_type, model):
    output_model_dir = os.path.join(output_path, model_type)
    if not os.path.exists(output_model_dir): os.makedirs(output_model_dir)    # 没有文件夹则创建
    model_to_save = model.module if hasattr(model, 'module') else model     # Only save the model it-self
    output_model_file = os.path.join(output_model_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)