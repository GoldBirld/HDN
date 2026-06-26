from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

def _label_to_name(label) -> str:
    s = str(label).strip().lower()
    if s in {"1","pos","positive","true","yes"}:
        return "positive"
    # 其余一律按负类
    return "negative"

def get_resize(image_size):
    side = 1
    for _ in range(20):
        if side >= image_size:
            return side
        side *= 2
    return image_size

def api_encode(data, labelvocab, config):
    # 只注册二分类
    labelvocab.add_label('positive')
    labelvocab.add_label('negative')

    tokenizer = AutoTokenizer.from_pretrained(config.bert_name)

    img_transform = transforms.Compose([
        transforms.Resize(get_resize(config.image_size)),
        transforms.CenterCrop(config.image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

    max_len = int(getattr(config, "max_seq_len", 128))

    guids, encoded_texts, encoded_imgs, encoded_labels = [], [], [], []
    for guid, text, img, label in tqdm(data, desc='----- [Encoding]'):
        guids.append(str(guid))

        # 文本
        text = ("" if text is None else str(text)).replace('#', '')
        input_ids = tokenizer.encode(text, add_special_tokens=True,
                                     truncation=True, max_length=max_len)
        encoded_texts.append(input_ids)

        # 图像
        if not isinstance(img, Image.Image):
            if isinstance(img, np.ndarray):
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)
                img = Image.fromarray(img.astype("uint8"))
            else:
                img = Image.new("RGB", (224,224), (0,0,0))
        encoded_imgs.append(img_transform(img))

        # 标签：先映射成名字，再取 id
        name = _label_to_name(label)
        encoded_labels.append(labelvocab.label_to_id(name))

        # print("labelvocab size =", len(labelvocab))
        # print("id2label =", getattr(labelvocab, "id2label", None))

    return guids, encoded_texts, encoded_imgs, encoded_labels
