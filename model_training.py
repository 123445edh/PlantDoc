import torch
import torch.nn as nn
from torch import optim
import torchvision.models as models
import json

from data_preparation import class_names, val_loader, train_loader

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# 使用 torchvision ViT 模型
class PlantDiseaseViT:
    def __init__(self, num_classes=27, model_name='vit_b_16'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载 torchvision ViT 模型
        if model_name == 'vit_b_16':
            self.model = models.vit_b_16(pretrained=True)
        elif model_name == 'vit_b_32':
            self.model = models.vit_b_32(pretrained=True)
        else:
            self.model = models.vit_b_16(pretrained=True)

        # 修改分类头
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

        self.model.to(self.device)
        print(f"加载模型: torchvision/{model_name}")


# 重新定义训练器类，适配 torchvision 模型
class ViTTrainer:
    def __init__(self, model, train_loader, val_loader, class_names, learning_rate=1e-4, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.learning_rate = learning_rate
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # 调试：打印batch的类型和内容
            if batch_idx == 0:
                print(f"Batch类型: {type(batch)}")
                print(f"Batch长度: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")

            # 处理不同的数据格式
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                # 标准格式: (images, labels)
                images, labels = batch
            elif isinstance(batch, dict):
                # 字典格式: {'pixel_values': ..., 'labels': ...}
                images = batch.get('pixel_values', batch.get('images'))
                labels = batch.get('labels')
            else:
                print(f"未知的batch格式: {type(batch)}")
                continue

            # 确保数据是张量
            if not isinstance(images, torch.Tensor):
                print(f"图像不是张量: {type(images)}")
                continue
            if not isinstance(labels, torch.Tensor):
                print(f"标签不是张量: {type(labels)}")
                continue

            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        """验证模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # 处理不同的数据格式
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, labels = batch
                elif isinstance(batch, dict):
                    images = batch.get('pixel_values', batch.get('images'))
                    labels = batch.get('labels')
                else:
                    continue

                # 确保数据是张量
                if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
                    continue

                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def train(self, epochs=10, save_path='model.pth'):
        """完整训练过程"""
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        best_acc = 0.0

        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 50)

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.scheduler.step()

            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
            print(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
            print(f'学习率: {self.optimizer.param_groups[0]["lr"]:.6f}')

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'epoch': epoch
                }, save_path)
                print(f'保存最佳模型，验证准确率: {val_acc:.2f}%')

        return history


# 完整的训练流程
if __name__ == "__main__":
    print("开始训练...")

    # 初始化模型
    vit_model = PlantDiseaseViT(num_classes=len(class_names))
    trainer = ViTTrainer(vit_model.model, train_loader, val_loader, class_names)

    # 开始训练
    history = trainer.train(epochs=10, save_path='baseline_vit_model.pth')

    # 保存训练历史
    with open('training_history.json', 'w') as f:
        json.dump(history, f)

    print("训练完成！")