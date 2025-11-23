"""
模型训练模块 - 提供模型定义和训练功能
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from transformers import ViTForImageClassification
import json
from tqdm import tqdm

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.model_utils import set_seed, get_device, save_checkpoint, calculate_accuracy
import config


class PlantDiseaseViT:
    """
    Vision Transformer 模型封装类
    """

    def __init__(self, num_classes=None, model_name=None):
        if num_classes is None:
            num_classes = config.MODEL_CONFIG['num_classes']
        if model_name is None:
            model_name = config.MODEL_CONFIG['default_model']

        self.num_classes = num_classes
        self.model_name = model_name
        self.device = get_device()

        # 加载预训练模型
        print(f"正在加载模型: {model_name}")
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)

        # 打印模型信息
        self._print_model_info()

    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"模型加载完成: {self.model_name}")
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"类别数: {self.num_classes}")
        print(f"设备: {self.device}")

    def get_model(self):
        """返回模型实例"""
        return self.model

    def save_model(self, filepath):
        """保存模型"""
        torch.save(self.model.state_dict(), filepath)
        print(f"模型已保存: {filepath}")

    def load_model(self, filepath):
        """加载模型权重"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"模型权重已加载: {filepath}")


class ViTTrainer:
    """
    ViT 模型训练器
    """

    def __init__(self, model, train_loader, val_loader, class_names, learning_rate=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.device = model.device

        if learning_rate is None:
            learning_rate = config.TRAINING_CONFIG['learning_rate']

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=config.TRAINING_CONFIG['weight_decay']
        )
        self.criterion = nn.CrossEntropyLoss()

        # 训练历史记录
        self.history = {
            'epochs': [],
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }

        # 最佳模型记录
        self.best_accuracy = 0.0
        self.best_epoch = 0

        print(f"训练器初始化完成")
        print(f"学习率: {learning_rate}")
        print(f"优化器: AdamW")
        print(f"损失函数: CrossEntropyLoss")

    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch} Training')

        for batch_idx, batch in enumerate(progress_bar):
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(pixel_values=pixel_values)
            loss = self.criterion(outputs.logits, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计信息
            total_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条
            current_acc = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self):
        """在验证集上评估模型"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(pixel_values=pixel_values)
                loss = self.criterion(outputs.logits, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100. * correct / total
        avg_loss = val_loss / len(self.val_loader)

        return avg_loss, accuracy, all_predictions, all_labels

    def train(self, epochs=None, save_dir=None):
        """完整训练流程"""
        if epochs is None:
            epochs = config.TRAINING_CONFIG['epochs']
        if save_dir is None:
            save_dir = config.TRAINING_CONFIG['save_dir']

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n开始训练，共 {epochs} 个 epochs")
        print("=" * 60)

        for epoch in range(1, epochs + 1):
            # 训练一个 epoch
            train_loss, train_acc = self.train_epoch(epoch)

            # 验证
            val_loss, val_acc, predictions, true_labels = self.validate()

            # 记录历史
            self.history['epochs'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # 打印结果
            print(f'\nEpoch {epoch}/{epochs}:')
            print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
            print(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')

            # 保存最佳模型
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.best_epoch = epoch

                best_model_path = save_dir / 'best_model.pth'
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_acc,
                    self.class_names, best_model_path
                )
                print(f'  ✅ 保存最佳模型，准确率: {val_acc:.2f}%')

        # 保存训练历史
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        # 保存最终模型
        final_model_path = save_dir / 'final_model.pth'
        save_checkpoint(
            self.model, self.optimizer, epochs, val_acc,
            self.class_names, final_model_path
        )

        self._print_training_summary()

        return self.history

    def _print_training_summary(self):
        """打印训练总结"""
        print("\n" + "=" * 60)
        print("训练总结")
        print("=" * 60)
        print(f"最佳 epoch: {self.best_epoch}")
        print(f"最佳验证准确率: {self.best_accuracy:.2f}%")
        print(f"最终训练准确率: {self.history['train_acc'][-1]:.2f}%")
        print(f"最终验证准确率: {self.history['val_acc'][-1]:.2f}%")

        if len(self.history['epochs']) > 1:
            improvement = self.history['val_acc'][-1] - self.history['val_acc'][0]
            print(f"准确率提升: {improvement:+.2f}%")

    def get_training_history(self):
        """返回训练历史"""
        return self.history

    def predict(self, dataloader):
        """在给定数据上进行预测"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting'):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(pixel_values=pixel_values)
                probabilities = torch.softmax(outputs.logits, dim=1)
                _, predicted = torch.max(outputs.logits, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        return all_predictions, all_labels, all_probabilities


# 使用示例
if __name__ == "__main__":
    # 示例用法
    set_seed(42)

    # 加载数据
    from data_loader import PlantDocDataLoader

    loader = PlantDocDataLoader()
    train_loader, val_loader, class_names = loader.load_data(batch_size=8)

    # 初始化模型
    vit_model = PlantDiseaseViT(num_classes=len(class_names))

    # 创建训练器
    trainer = ViTTrainer(
        vit_model.model,
        train_loader,
        val_loader,
        class_names,
        learning_rate=2e-5
    )

    # 开始训练（只训练2个epoch作为示例）
    history = trainer.train(epochs=2)