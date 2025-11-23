import sys
import os

sys.path.append('./utils')
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import ViTImageProcessor

sys.path.append('..')  # 添加项目根目录

from utils.data_utils import explore_dataset, prepare_data_loaders
from utils.model_utils import set_seed, get_device
import config

# 设置中文字体
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
set_seed(42)

print("环境设置完成！")


# %%
def explore_dataset(data_path):
    """探索数据集结构"""
    total_images = 0
    train_images = 0
    test_images = 0
    train_class_counts = {}
    test_class_counts = {}

    print("数据集结构:")

    # 检查 train 目录
    train_path = os.path.join(data_path, "train")
    if os.path.exists(train_path):
        print(f"\nTRAIN 目录:")
        for class_name in os.listdir(train_path):
            class_path = os.path.join(train_path, class_name)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                train_class_counts[class_name] = len(images)
                train_images += len(images)
                print(f"  {class_name}: {len(images)} 张图像")

    # 检查 test 目录
    test_path = os.path.join(data_path, "test")
    if os.path.exists(test_path):
        print(f"\nTEST 目录:")
        for class_name in os.listdir(test_path):
            class_path = os.path.join(test_path, class_name)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                test_class_counts[class_name] = len(images)
                test_images += len(images)
                print(f"  {class_name}: {len(images)} 张图像")

    total_images = train_images + test_images
    total_classes = len(set(list(train_class_counts.keys()) + list(test_class_counts.keys())))

    print(f"\n总统计: {total_classes} 个类别, {total_images} 张图像")
    print(f"训练集: {train_images} 张图像")
    print(f"测试集: {test_images} 张图像")

    # 绘制类别分布图
    plt.figure(figsize=(15, 6))

    # 训练集分布
    if train_class_counts:
        plt.subplot(1, 2, 1)
        plt.bar(train_class_counts.keys(), train_class_counts.values(), color='skyblue', alpha=0.7)
        plt.title('训练集各类别图像数量分布')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('图像数量')

        # 在柱状图上显示数字
        for i, (class_name, count) in enumerate(train_class_counts.items()):
            plt.text(i, count + 0.1, str(count), ha='center', va='bottom')

    # 测试集分布
    if test_class_counts:
        plt.subplot(1, 2, 2)
        plt.bar(test_class_counts.keys(), test_class_counts.values(), color='lightcoral', alpha=0.7)
        plt.title('测试集各类别图像数量分布')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('图像数量')

        # 在柱状图上显示数字
        for i, (class_name, count) in enumerate(test_class_counts.items()):
            plt.text(i, count + 0.1, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # 如果有数据，绘制饼图显示训练/测试分布
    if total_images > 0:
        plt.figure(figsize=(10, 5))

        # 训练测试分布
        plt.subplot(1, 2, 1)
        sizes = [train_images, test_images]
        labels = [f'训练集\n{train_images}张', f'测试集\n{test_images}张']
        colors = ['lightblue', 'lightcoral']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('训练集 vs 测试集分布')

        # 类别数量分布（前10个类别）
        plt.subplot(1, 2, 2)
        all_class_counts = {}
        for class_name in set(list(train_class_counts.keys()) + list(test_class_counts.keys())):
            all_class_counts[class_name] = train_class_counts.get(class_name, 0) + test_class_counts.get(class_name, 0)

        # 按数量排序，取前10个
        sorted_classes = sorted(all_class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        class_names = [x[0] for x in sorted_classes]
        class_counts = [x[1] for x in sorted_classes]

        plt.barh(class_names, class_counts, color='lightgreen')
        plt.title('前10个类别的图像数量')
        plt.xlabel('图像数量')

        # 在条形图上显示数字
        for i, count in enumerate(class_counts):
            plt.text(count + 0.1, i, str(count), va='center')

        plt.tight_layout()
        plt.show()

    return train_class_counts, test_class_counts


# 使用示例
data_path = "F:\PlantDoc"  # 修改为您的实际路径
train_counts, test_counts = explore_dataset(data_path)


# %%
class PlantDocDataset(Dataset):
    def __init__(self, image_paths, labels, processor, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        image = Image.open(self.image_paths[idx]).convert('RGB')

        # 应用变换（如果有）
        if self.transform:
            image = self.transform(image)

        # 使用ViT处理器
        inputs = self.processor(image, return_tensors="pt")

        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }


# %%
def prepare_data_loaders(data_path, batch_size=16, test_size=0.2):
    """准备训练和验证数据加载器"""

    # 收集训练集图像路径和标签
    train_image_paths = []
    train_labels = []
    train_path = os.path.join(data_path, "train")

    class_names = []
    class_to_idx = {}

    if os.path.exists(train_path):
        class_names = sorted([d for d in os.listdir(train_path)
                              if os.path.isdir(os.path.join(train_path, d))])
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = os.path.join(train_path, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    train_image_paths.append(os.path.join(class_dir, img_file))
                    train_labels.append(class_to_idx[class_name])

    if len(train_image_paths) == 0:
        print("错误: 在训练目录中没有找到任何图像文件！")
        print("请检查数据集路径和目录结构")
        return None, None, None

    print(f"原始训练集: {len(train_image_paths)} 张图像")
    print(f"类别数量: {len(class_names)}")

    # 检查标签长度是否匹配
    if len(train_image_paths) != len(train_labels):
        print(f"警告: 图像路径数量({len(train_image_paths)})和标签数量({len(train_labels)})不匹配!")
        # 取较小值
        min_len = min(len(train_image_paths), len(train_labels))
        train_image_paths = train_image_paths[:min_len]
        train_labels = train_labels[:min_len]

    # 划分训练集和验证集
    train_paths, val_paths, train_labels_list, val_labels = train_test_split(
        train_image_paths, train_labels, test_size=test_size, random_state=42, stratify=train_labels
    )

    print(f"训练集: {len(train_paths)} 张图像")
    print(f"验证集: {len(val_paths)} 张图像")
    print(f"总类别数: {len(class_names)}")
    print("类别列表:")
    for i, cls in enumerate(class_names):
        print(f"  {i + 1}: {cls}")

    # 使用 torchvision transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 修改 PlantDocDataset 类
    class PlantDocDataset(Dataset):
        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            try:
                # 加载图像
                image = Image.open(self.image_paths[idx]).convert('RGB')

                # 应用变换
                if self.transform:
                    image = self.transform(image)

                return {
                    'pixel_values': image,
                    'labels': torch.tensor(self.labels[idx], dtype=torch.long)
                }
            except Exception as e:
                print(f"加载图像出错: {self.image_paths[idx]}, 错误: {e}")
                # 返回一个空图像或跳过
                return self.__getitem__((idx + 1) % len(self.image_paths))

    # 创建数据集
    train_dataset = PlantDocDataset(train_paths, train_labels_list, transform)
    val_dataset = PlantDocDataset(val_paths, val_labels, transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, class_names


# 使用修正版本
train_loader, val_loader, class_names = prepare_data_loaders(data_path)