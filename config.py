# config.py
import torch

# 数据配置
DATA_CONFIG = {
    'data_path': "F:\PlantDoc",
    'batch_size': 16,
    'test_size': 0.2,
    'random_state': 42,
    'image_size': 224
}

# 模型配置
MODEL_CONFIG = {
    'default_model': 'google/vit-base-patch16-224',
    'num_classes': 27,  # PlantDoc数据集的27个类别
    'pretrained': True
}

# 训练配置
TRAINING_CONFIG = {
    'epochs': 20,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'save_dir': './checkpoints',
    'early_stopping_patience': 5
}

# 实验配置
EXPERIMENT_CONFIG = {
    'enable_mixed_precision': True,
    'use_amp': torch.cuda.is_available()
}