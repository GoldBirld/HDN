# === save as: tools/compute_class_weights.py（随便放，能运行即可） ===
from collections import Counter
import numpy as np

# 按你的工程导入
from common import read_from_file, train_val_split

# ====== 路径按你的工程填写 ======
# json_path = r'../redata/data/single_train.json'
# img_dir   = r'../data/single_data/MVSA_Single/data'

# json_path = r'../data/tw15_data/train.json'
# img_dir   = r'../data/tw15_data/tw15_data'

# json_path = r'../data//MVSA-multiple/MVSA/train.json'
# img_dir   = r'../data/MVSA-multiple/MVSA/data'

# json_path = r'../data/tw17_data/train.json'
# img_dir   = r'../data/tw17_data/tw17_data'

json_path = r'../data/train.json'
img_dir   = r'../data/data'

# json_path = r'../amazon_datapro/processed_amazon/train.json'
# img_dir   = r'../amazon_datapro/processed_amazon/data'

num_labels = 2   # <- 对应 Config.num_labels

# 扫描目录找出损坏图片
# from pathlib import Path
# from PIL import Image
# bad = []
# for p in Path(img_dir).rglob("*.*"):
#     try:
#         with Image.open(p) as im:
#             im.verify()
#     except Exception as e:
#         bad.append((str(p), str(e)))
# print(f"坏图数量: {len(bad)}")
# 之后按 bad 列表重下或删除

# 读取与切分（只用训练集统计）
data = read_from_file(json_path, img_dir)
train_data, val_data, test_data = train_val_split(data)

# 取出训练集标签（train_data 形如 (guid, text, img, label)）
raw_labels = [lab for (_, _, _, lab) in train_data]

# 如果标签是字符串，做个映射; 若已是 int 会在下步直接用
# str2id = {'positive': 0, 'neutral': 1, 'negative': 2}  # 不含 'null'
str2id = {'positive': 1,  'negative': 0}  # 不含 'null'
labels = []
for lab in raw_labels:
    if isinstance(lab, int):
        labels.append(lab)
    else:
        if lab in str2id:
            labels.append(str2id[lab])
        # 其他字符串（如 'null'）直接跳过，不纳入 3 类统计
        # else: pass

# 统计每类数量
cnt = Counter(labels)
K = num_labels
N = len(labels)

eps = 1e-6
counts = np.array([cnt.get(c, 0) for c in range(K)], dtype=float)
if (counts == 0).any():
    missing = [i for i, v in enumerate(counts) if v == 0]
    print('[警告] 训练集中以下类别计数为 0，将以极小值代替：', missing)
counts = np.clip(counts, eps, None)

# ---------- 方案 A：Balanced（频率倒数），均值归一到 1 ----------
weights_bal = N / (K * counts)
weights_bal = weights_bal * (K / weights_bal.sum())  # 均值=1
weights_bal = np.clip(weights_bal, 0.1, 10.0)        # 可选裁剪，防过大/过小

# ---------- 方案 B：Effective Number（长尾更友好） ----------
beta = 0.99  # 长尾更重可调到 0.999
eff_num = (1 - np.power(beta, counts)) / (1 - beta)
weights_eff = 1.0 / eff_num
weights_eff = weights_eff * (K / weights_eff.sum())  # 均值=1
weights_eff = np.clip(weights_eff, 0.1, 10.0)

def fmt(arr):
    return '[' + ', '.join(f'{x:.6f}' for x in arr.tolist()) + ']'

print('\n=== 统计结果 ===')
print('Class counts:', counts.tolist())
print('\n建议粘贴到 Config.py（任选其一）:')
print('loss_weight（Balanced）        =', fmt(weights_bal))
print('loss_weight（EffectiveNumber） =', fmt(weights_eff))
