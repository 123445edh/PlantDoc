# train_swin_offline.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import swin_b, Swin_B_Weights
from PIL import Image
import os
from sklearn.model_selection import train_test_split


class PlantDocDataset(Dataset):
    """简化的植物病害数据集类"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(self.labels[idx], dtype=torch.long)
        except Exception as e:
            print(f"加载图像出错: {self.image_paths[idx]}, 错误: {e}")
            return self.__getitem__((idx + 1) % len(self.image_paths))


def load_dataset_paths(dataset_path):
    """加载数据集路径"""
    dataset_path = os.path.normpath(dataset_path)

    train_image_paths = []
    train_labels = []
    train_path = os.path.join(dataset_path, "TRAIN")

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
        return None, None, None

    print(f"原始训练集: {len(train_image_paths)} 张图像")
    print(f"类别数量: {len(class_names)}")

    return train_image_paths, train_labels, class_names


def create_swin_model_offline(num_classes=27):
    """使用 torchvision 的 Swin Transformer，避免 huggingface 下载"""
    try:
        # 方法1: 使用 torchvision 的 Swin Transformer
        print("尝试使用 torchvision 的 Swin Transformer...")
        weights = Swin_B_Weights.IMAGENET1K_V1
        model = swin_b(weights=weights)

        # 修改分类头
        model.head = nn.Linear(model.head.in_features, num_classes)
        print("成功加载 torchvision Swin Transformer")
        return model

    except Exception as e:
        print(f"torchvision Swin 加载失败: {e}")
        print("尝试创建简单的 CNN 模型作为替代...")
        return create_simple_cnn(num_classes)


def create_simple_cnn(num_classes=27):
    """创建简单的 CNN 模型作为备选"""
    print("创建简单的 CNN 模型...")

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512 * 14 * 14, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    return SimpleCNN(num_classes)


def train_model():
    """训练模型 - 自动选择可用模型"""

    # 超参数
    learning_rate = 2e-5
    batch_size = 16
    weight_decay = 5e-5
    epochs = 5

    # 数据集路径
    dataset_path = r"F:\PlantDoc"

    print("=" * 60)
    print("植物病害分类模型训练")
    print("=" * 60)

    # 加载数据
    train_image_paths, train_labels, class_names = load_dataset_paths(dataset_path)
    if train_image_paths is None:
        return

    # 划分训练验证集
    train_paths, val_paths, train_labels_list, val_labels = train_test_split(
        train_image_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )

    print(f"训练集: {len(train_paths)} 张图像")
    print(f"验证集: {len(val_paths)} 张图像")
    print(f"类别数: {len(class_names)}")

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = PlantDocDataset(train_paths, train_labels_list, train_transform)
    val_dataset = PlantDocDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    print("创建模型...")
    model = create_swin_model_offline(num_classes=len(class_names))
    model = model.to(device)

    # 显示模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型类型: {type(model).__name__}")
    print(f"模型参数: 总计 {total_params:,}，可训练 {trainable_params:,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # 学习率调度
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 训练循环
    print(f"\n开始训练...")
    print(f"学习率: {learning_rate}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {epochs}")

    best_acc = 0.0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 20 == 0:
                acc = 100. * correct / total
                print(f'Epoch [{epoch + 1}/{epochs}], 批次 [{batch_idx}/{len(train_loader)}], '
                      f'损失: {loss.item():.4f}, 准确率: {acc:.2f}%')

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # 计算指标
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        val_acc = 100. * val_correct / val_total

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f'\nEpoch {epoch + 1}/{epochs} 完成:')
        print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'  验证准确率: {val_acc:.2f}%')
        print(f'  当前学习率: {current_lr:.2e}')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
                'model_type': type(model).__name__
            }, 'best_model.pth')
            print(f'  ★ 保存最佳模型，验证准确率: {val_acc:.2f}%')

    print(f"\n训练完成!")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"模型已保存到: best_model.pth")

    # 保存类别信息
    with open('class_names.txt', 'w', encoding='utf-8') as f:
        for i, class_name in enumerate(class_names):
            f.write(f"{i}: {class_name}\n")
    print(f"类别信息已保存到: class_names.txt")


if __name__ == "__main__":
    train_model()