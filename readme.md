一、如何运行

    1.安装环境 pip install requirements.txt
    2.运行 python main.py或者直接右键
    P.S：运行过程中生成的modeloutput文件夹存放了roi（图像区域特征）和textcache(文本向量）两个文件夹，以便后续复用提升速度。

二、模型结构

    HDN/
        ├── comModels/ 对比实验
        │   ├── __init__.py
        │   ├── bert_model.py 文本建模部分替换为BERT做情感分类基线模型
        │   ├── bertBilstm_model.py 文本建模部分使用BERT提取特征和BiLSTM进行编码
        │   ├── bilstm_model.py 文本建模部分使用纯BiLSTM模型
        │   ├── biSRU_model.py 文本建模部分使用双向SRU模型
        │   ├── bridgeTower_model.py 融合方法部分使用BridgeTower模型
        │   ├── drf.py 融合方法部分使用DRF模型
        │   ├── esafn.py 融合方法部分使用ESAFN模型
        │   ├── inceptionV3_model.py 图像建模部分使用inceptionV3模型
        │   ├── itin.py 融合方法部分使用ITIN模型
        │   ├── mvan.py 融合方法部分使用MVAN模型
        │   ├── rcnn_model.py 文本建模部分使用RCNN模型
        │   ├── repVGG_model.py 图像建模部分使用RepVGG模型
        │   ├── vault_model.py 融合方法部分使用VauLT模型
        │   └── vggNet_model.py 图像建模部分使用VGGNet模型
        ├── data/
        │   └── data/ 数据集，label.txt记录了(guid,tag)，guid是某个样本的唯一编号，tag是情感标签（positive,negative)
        ├── Models/
        │   ├── __init__.py
        │   └── HDN.py 本文模型(Hierarchical Dynamic Neighborhood,HDN):文本编码器+图像编码器→对齐注意力交互→低秩张量融合→分层动态邻域模块→分类头
        ├── utils/ 工具类
        │   ├── APIs/ 数据加载
        │   │   ├── __init__.py
        │   │   ├── APIDataset.py 数据集定义：返回训练需要的张量
        │   │   ├── APIDecode.py 解码工：将模型输出转回文本，把存档数据还原为可读形式
        │   │   ├── APIEncode.py 编码工具（这个是不包含图像区域图像的编码过程，可以不使用这个）
        │   │   ├── APIMetric.py 评估指标计算（打印一些测试报告）
        │   │   └── newAPIEncode.py 包含原始的文本和图片，按指定的骨干模型与训练/验证规范，统一变成模型能直接张量；并且可选地提前算好图像 ROI 特征和文本 token 向量，还能落盘缓存，下次用更快。
        │   ├── __init__.py
        │   ├── class_weight.py 当类别不平衡时计算可以放在损失函数内的class_weight
        │   ├── common.py 常用的一些工具
        │   ├── DataProcess.py 数据预处理脚本
        │   └── FocalLoss.py 焦点损失函数
        ├── __init__.py
        ├── Config.py 模型配置
        ├── main.py 程序入口
        ├── readme.md 使用说明
        ├── requirements.txt 模型依赖的安装包
        └── Trainer.py 模型训练/验证/测试过程

