"""
PlantDoc Transformer 项目工具模块
提供数据处理、模型训练和可视化的工具函数
"""

# 数据工具导入
from .data_utils import (
    explore_dataset,
    PlantDocDataset,
    prepare_data_loaders,
    get_transforms
)

# 模型工具导入
from .model_utils import (
    set_seed,
    get_device,
    cleanup_memory,
    calculate_accuracy,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    enable_mixed_precision
)

# 定义公开的API
__all__ = [
    # 数据工具
    'explore_dataset',
    'PlantDocDataset',
    'prepare_data_loaders',
    'get_transforms',

    # 模型工具
    'set_seed',
    'get_device',
    'cleanup_memory',
    'calculate_accuracy',
    'save_checkpoint',
    'load_checkpoint',
    'count_parameters',
    'enable_mixed_precision'
]

# 版本信息
__version__ = "1.0.0"
__author__ = "PlantDoc Transformer Project"
__description__ = "工具模块包含数据处理和模型训练的工具函数"

print(f"PlantDoc Transformer 工具模块 v{__version__} 已加载")