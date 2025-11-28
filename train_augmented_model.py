# train_augmented_model.py
import torch
import torch.nn as nn
from torch import optim
import json
from tqdm import tqdm
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data_preparation import class_names, val_loader, train_loader
from advanced_augmentation import AdvancedAugmentation
from model_training import PlantDiseaseViT, ViTTrainer


class AugmentedViTTrainer(ViTTrainer):
    """使用数据增强的ViT训练器"""

    def __init__(self, model, train_loader, val_loader, class_names,
                 learning_rate=2e-5, augmentation_strategy='medium'):
        super().__init__(model, train_loader, val_loader, class_names, learning_rate)
        self.augmenter = AdvancedAugmentation(augmentation_strategy)
        self.use_cutmix = True
        self.use_mixup = True

        print(f"使用数据增强训练器，策略: {augmentation_strategy}")

    def train_epoch(self):
        """训练一个epoch，包含数据增强"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc='Training')):
            # 应用高级数据增强
            if self.use_cutmix and torch.rand(1) < 0.5:
                batch = self.augmenter.cutmix(batch)
            elif self.use_mixup and torch.rand(1) < 0.5:
                batch = self.augmenter.mixup(batch)

            # 处理不同的数据格式
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
            elif isinstance(batch, dict):
                images = batch.get('pixel_values', batch.get('images'))
                labels = batch.get('labels')
            else:
                continue

            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            # 处理增强后的损失计算
            if 'lam' in batch and 'labels_b' in batch:
                # CutMix或MixUp批次
                labels_b = batch['labels_b'].to(self.device)
                lam = batch['lam']
                loss = self._mixed_loss(outputs, labels, labels_b, lam)
            else:
                loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def _mixed_loss(self, outputs, labels_a, labels_b, lam):
        """混合损失函数，用于CutMix和MixUp"""
        criterion = nn.CrossEntropyLoss()
        return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


def train_augmented_model(epochs=10, augmentation_strategy='medium', save_path='augmented_model.pth'):
    """训练使用数据增强的模型"""
    print(f"开始训练数据增强模型，策略: {augmentation_strategy}")

    # 初始化模型
    vit_model = PlantDiseaseViT(num_classes=len(class_names))

    # 创建增强训练器
    trainer = AugmentedViTTrainer(
        vit_model.model, train_loader, val_loader, class_names,
        learning_rate=2e-5, augmentation_strategy=augmentation_strategy
    )

    # 开始训练
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    best_acc = 0.0

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        print('-' * 50)

        # 训练阶段（使用数据增强）
        train_loss, train_acc = trainer.train_epoch()

        # 验证阶段（不使用数据增强）
        val_loss, val_acc = trainer.validate()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_accuracy': val_acc,
                'epoch': epoch,
                'augmentation_strategy': augmentation_strategy
            }, save_path)
            print(f'保存最佳增强模型，验证准确率: {val_acc:.2f}%')

    # 保存训练历史
    with open('augmented_training_history.json', 'w') as f:
        json.dump(history, f)

    print(f"\n数据增强模型训练完成！最佳准确率: {best_acc:.2f}%")
    return history, best_acc


if __name__ == "__main__":
    # 训练数据增强模型
    history, best_acc = train_augmented_model(
        epochs=10,
        augmentation_strategy='medium',
        save_path='augmented_model.pth'
    )

    print(f"\n训练完成！现在可以运行 attention_analysis_augmented.py 来分析增强模型")