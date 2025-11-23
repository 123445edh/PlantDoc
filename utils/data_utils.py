import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import ViTImageProcessor
import config


def explore_dataset(data_path=None):
    """探索数据集结构"""
    if data_path is None:
        data_path = config.DATA_CONFIG['data_path']

    class_counts = {}
    total_images = 0

    print("数据集结构:")
    for class_name in os.listdir(data_path):
        class_dir = os.path.join(data_path, class_name)
        if os.path.isdir(class_dir):
            images = [f for f in os.listdir(class_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[class_name] = len(images)
            total_images += len(images)
            print(f"  {class_name}: {len(images)} 张图像")

    print(f"\n总统计: {len(class_counts)} 个类别, {total_images} 张图像")

    # 绘制类别分布
    plt.figure(figsize=(15, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('各类别图像数量分布')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return class_counts


class PlantDocDataset(Dataset):
    def __init__(self, image_paths, labels, processor, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        inputs = self.processor(image, return_tensors="pt")

        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }


def prepare_data_loaders(data_path=None, batch_size=None, test_size=None, random_state=42):
    """准备数据加载器"""
    if data_path is None:
        data_path = config.DATA_CONFIG['data_path']
    if batch_size is None:
        batch_size = config.DATA_CONFIG['batch_size']
    if test_size is None:
        test_size = config.DATA_CONFIG['test_size']

    # 收集图像路径和标签
    image_paths = []
    labels = []
    class_names = sorted([d for d in os.listdir(data_path)
                          if os.path.isdir(os.path.join(data_path, d))])

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}

    for class_name in class_names:
        class_dir = os.path.join(data_path, class_name)
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, img_file))
                labels.append(class_to_idx[class_name])

    # 划分数据集
    from sklearn.model_selection import train_test_split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    print(f"训练集: {len(train_paths)} 张图像")
    print(f"验证集: {len(val_paths)} 张图像")

    # 初始化处理器
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    # 创建数据集和数据加载器
    train_dataset = PlantDocDataset(train_paths, train_labels, processor)
    val_dataset = PlantDocDataset(val_paths, val_labels, processor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, class_names, class_to_idx, idx_to_class


# 在 data_utils.py 文件末尾添加：

def get_transforms(augmentation_strategy='basic'):
    """
    获取数据增强变换

    参数:
        augmentation_strategy: 增强策略 ('basic', 'advanced', 'heavy')

    返回:
        transform: 数据变换组合
    """
    if augmentation_strategy == 'basic':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    elif augmentation_strategy == 'advanced':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    elif augmentation_strategy == 'heavy':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
            transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    else:
        # 默认基础变换
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])