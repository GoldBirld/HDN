'''
data process: 数据处理, 包括 标签Vocab 和 数据处理类
    tips:
        其中标签Vocab实例化对象必须在api_encode中被调用(add_label)
'''

from torch.utils.data import DataLoader

from APIs.APIDataset import APIDataset
# from APIs.APIEncode import api_encode
from APIs.newAPIEncode import api_encode
from APIs.APIDecode import api_decode
from APIs.APIMetric import api_metric


class LabelVocab:
    UNK = 'UNK'

    def __init__(self) -> None:
        self.label2id = {}
        self.id2label = {}

    def __len__(self):
        return len(self.label2id)

    def add_label(self, label):
        if label not in self.label2id:
            self.label2id.update({label: len(self.label2id)})
            self.id2label.update({len(self.id2label): label})

    def label_to_id(self, label):
        return self.label2id.get(label)
    
    def id_to_label(self, id):
        return self.id2label.get(id)


class Processor:

    def __init__(self, config) -> None:
        self.config = config
        self.labelvocab = LabelVocab()
        pass

    def __call__(self, data, params):
        return self.to_loader(data, params)

    def encode(self, data):
        return api_encode(
            data, self.labelvocab, self.config,
            use_frcnn=getattr(self.config, "use_frcnn_regions", False),
            frcnn_topk=getattr(self.config, "frcnn_topk", 16),
            roi_cache_dir=getattr(self.config, "roi_cache_dir", None),
            frcnn_device=getattr(self.config, "frcnn_device", None),
            roi_cache_load_only=getattr(self.config,"roi_cache_load_only",None),
            use_text_precompute=getattr(self.config, "precompute_text", False),
            text_cache_dir=getattr(self.config, "text_cache_dir", None),
            text_device=getattr(self.config,"text_device",None),
            text_cache_load_only=getattr(self.config,"text_cache_load_only",None)

        )

    def decode(self, outputs):
        return api_decode(outputs, self.labelvocab)

    def metric(self, inputs, outputs):
        return api_metric(inputs, outputs)
    
    def to_dataset(self, data):
        dataset_inputs = self.encode(data)
        return APIDataset(*dataset_inputs)

    def to_loader(self, data, params):
        dataset = self.to_dataset(data)
        return DataLoader(dataset=dataset, **params, collate_fn=dataset.collate_fn)