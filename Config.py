import os

class config:
    # 根目录
    root_path = os.getcwd()
    data_dir = os.path.join(root_path, './data/data/')
    train_data_path = os.path.join(root_path, 'data/train.json')
    val_data_path = os.path.join(root_path, 'data/val.json')
    test_data_path = os.path.join(root_path, 'data/test.json')
    output_path = os.path.join(root_path, 'HDN_output')
    output_test_path = os.path.join(output_path, 'test.txt')
    load_model_path = None

    # 一般超参
    epoch = 20
    learning_rate = 3e-4
    weight_decay = 0
    num_labels = 2

    # loss = "focal"
    focal_gamma = 2.0
    focal_alpha = [0.25,0.75] # [a_neg, a_pos]，用“有效样本数”法（β=0.999）得到
    label_smoothing = 0.0
    class_prior = [0.014489, 0.985511]
    decision_threshold = 0.5  # 训练后在验证集扫阈值
    class_weight=[0.768571, 1.231429]
    loss_weight = [0.768571, 1.231429]

    attn_dropout=0.2
    attn_temperature=None
    # loss_weight每个元素对应一个类别的权重。权重越高，损失函数对该类别错误分类的惩罚就越大。样本数量越少的类别权重越高
    # 建议Single使用[0.1, 0.6, 0.3]，Multiple使用[0.09, 0.68, 0.23]

    num_classes=2
    # text_backbone= "openai/clip-vit-base-patch32"
    # text_backbone:str = "bert-base-uncased"
    text_backbone = "bert-base-chinese"
    text_hidden= 768
    text_gru_hidden= 256
    hdn_hidden= 256
    lowrank_out = 256
    lowrank_rank= 2
    temperature = 1.0
    dropout = 0.3
    use_frcnn_regions= True
    frcnn_topk = 8
    api_roi_cache_dir = "modeloutput/roi"
    text_cache_dir='modeloutput/textcache'
    frcnn_device = "cuda"
    text_device="cuda"

    precompute_text_embeds=True
    precompute_text = False

    roi_cache_load_only=False
    text_cache_load_only=False


    early_stop_monitor = 'f1_macro'  # 可选: 'acc', 'f1', 'f1_macro', 'loss'
    early_stop_mode = 'max'  # 'max' 或 'min'；不写的话会自动根据 monitor 推断
    early_stop_patience = 10  # 容忍多少个 epoch 无提升
    early_stop_min_delta = 1e-4  # 提升的最小幅度

    # —— ImageModel 可控项 ——
    # image_backbone = "openai/clip-vit-base-patch32"  # 或 "resnet34"/"resnet50"
    image_backbone = "resnet152"  # 或 "resnet34"/"resnet50"
    image_global_only =False  # 只用全局特征，强烈建议先开以提速
    image_region_heads = 8  # 用区域注意时再调小，如 4/8
    image_region_pool = 7  # 用区域注意时开启下采样（7/5/4）；不用就设 None
    image_activation = "relu"  # "relu" 比 "gelu" 更快

    train_bert_last_n_layers = 1

    # 训练 ResNet 的哪些层（匹配 named_parameters 的前缀），例如只训 layer4
    train_resnet_layers = ('layer4',)

    # —— PyTorch 2.x 可选编译（对稳定模块尝试加速；不支持就静默跳过）——
    compile_submodules = False

    # Fuse相关
    fuse_model_type = 'model' # 模型使用类型
    only = None
    middle_hidden_size = 64
    attention_nhead = 8
    attention_dropout = 0.4
    fuse_dropout = 0.5
    out_hidden_size = 128

    # BERT相关
    fixed_text_model_params = False
    bert_name = r"C:\Users\huangxuan\model\bert-base-chinese"
    # bert_name = r"C:\Users\huangxuan\model\bert-base-uncased"
    #bert_name = 'bert-base-uncased'
    #bert_name = 'roberta-base'
    bert_learning_rate = 5e-6
    bert_dropout = 0.2
    text_max_length = 128

    # ResNet相关
    resnet_name = r"C:\Users\huangxuan\model\resnet-50"
    resnet_model_name="resnet-50"
    fixed_img_model_params = False
    image_size = 224
    fixed_image_model_params = True
    resnet_learning_rate = 5e-6
    resnet_dropout = 0.2
    img_hidden_seq = 64
    image_input_space = "imagenet"


    # Dataloader params
    checkout_params = {'batch_size': 4, 'shuffle': False}
    train_params = {'batch_size': 64, 'shuffle': True, 'num_workers': 0}
    val_params = {'batch_size': 64, 'shuffle': False, 'num_workers': 0}
    test_params =  {'batch_size': 64, 'shuffle': False, 'num_workers': 0}

    
    