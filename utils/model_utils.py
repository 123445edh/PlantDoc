import torch
import random
import numpy as np
import os
import gc


def set_seed(seed=42):
    """
    设置随机种子以确保结果可复现

    参数:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"设置随机种子: {seed}")


def cleanup_memory():
    """
    清理GPU内存
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("内存清理完成")


def calculate_accuracy(outputs, labels):
    """
    计算分类准确率

    参数:
        outputs: 模型输出
        labels: 真实标签

    返回:
        accuracy: 准确率
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


def save_checkpoint(model, optimizer, epoch, accuracy, class_names, filepath):
    """
    保存模型检查点

    参数:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        accuracy: 当前准确率
        class_names: 类别名称列表
        filepath: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'class_names': class_names
    }

    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    torch.save(checkpoint, filepath)
    print(f"检查点已保存: {filepath} (Epoch {epoch}, Accuracy: {accuracy:.2f}%)")


def load_checkpoint(model, optimizer, filepath):
    """
    加载模型检查点

    参数:
        model: 模型
        optimizer: 优化器
        filepath: 检查点路径

    返回:
        epoch: 训练的epoch
        accuracy: 准确率
        class_names: 类别名称
    """
    if not os.path.exists(filepath):
        print(f"检查点文件不存在: {filepath}")
        return 0, 0.0, None

    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    accuracy = checkpoint.get('accuracy', 0.0)
    class_names = checkpoint.get('class_names', None)

    print(f"检查点已加载: {filepath}")
    print(f"Epoch: {epoch}, Accuracy: {accuracy:.2f}%")

    return epoch, accuracy, class_names


def get_device():
    """
    获取可用设备 (GPU/CPU)

    返回:
        device: 设备
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")

    return device


def count_parameters(model):
    """
    计算模型参数数量

    参数:
        model: 模型

    返回:
        total_params: 总参数数量
        trainable_params: 可训练参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    return total_params, trainable_params


def enable_mixed_precision():
    """
    启用混合精度训练（需要torch>=1.6）

    返回:
        scaler: GradScaler对象
    """
    if torch.cuda.is_available():
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print("混合精度训练已启用")
        return scaler
    else:
        print("混合精度训练需要GPU")
        return None


def model_summary(model, input_size=(3, 224, 224)):
    """
    打印模型摘要

    参数:
        model: 模型
        input_size: 输入尺寸
    """
    try:
        from torchsummary import summary
        device = get_device()
        summary(model.to(device), input_size=input_size)
    except ImportError:
        print("请安装 torchsummary: pip install torchsummary")
        count_parameters(model)