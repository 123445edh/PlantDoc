import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import json
import os
import torch.nn as nn
import torchvision.models as models
from torch.optim import AdamW
import import_ipynb

try:
    # 导入01 notebook
    from data_preparation import (
        data_path,
        class_names,
        train_loader,
        val_loader,
        prepare_data_loaders
    )

    print("成功从01 notebook导入数据相关变量和函数")

    # 验证变量是否存在
    if 'class_names' not in globals() or class_names is None:
        print("class_names未定义或为空，重新加载数据...")
        train_loader, val_loader, class_names = prepare_data_loaders(data_path)

except Exception as e:
    print(f"从01 notebook导入失败: {e}")
    print("使用独立数据加载方案...")

    # 定义备用数据路径
    data_path = "F:\PlantDoc"


    # 重新定义数据加载函数
    def prepare_data_loaders(data_path, batch_size=16, test_size=0.2):
        # 这里放入之前的数据加载代码
        # ... (省略详细实现，使用之前提供的代码)
        pass


    # 加载数据
    train_loader, val_loader, class_names = prepare_data_loaders(data_path)

# 单元格3：从02 notebook导入模型相关类
try:
    # 导入02 notebook
    from model_training import PlantDiseaseViT, ViTTrainer

    print("成功从02 notebook导入模型类")

except Exception as e:
    print(f"从02 notebook导入失败: {e}")
    print("重新定义模型类...")


    # 重新定义模型类
    class PlantDiseaseViT:
        def __init__(self, num_classes=27, model_name='vit_b_16'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"使用设备: {self.device}")

            # 加载 torchvision ViT 模型
            if model_name == 'vit_b_16':
                self.model = models.vit_b_16(pretrained=True)
            elif model_name == 'vit_b_32':
                self.model = models.vit_b_32(pretrained=True)
            elif model_name == 'vit_l_16':
                self.model = models.vit_l_16(pretrained=True)
            else:
                self.model = models.vit_b_16(pretrained=True)

            # 修改分类头
            in_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(in_features, num_classes)

            self.model.to(self.device)
            print(f"加载模型: torchvision/{model_name}")


    class ViTTrainer:
        def __init__(self, model, train_loader, val_loader, class_names, learning_rate=1e-4):
            self.model = model
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.class_names = class_names
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.CrossEntropyLoss()

        def train(self, epochs=10, save_path='model.pth'):
            history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'train_accuracy': []}

            for epoch in range(epochs):
                # 训练阶段
                self.model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0

                for batch_idx, batch in enumerate(self.train_loader):
                    pixel_values = batch['pixel_values'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(pixel_values)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()

                    # 计算训练准确率
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                    if batch_idx % 50 == 0:
                        print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

                # 验证阶段
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch in self.val_loader:
                        pixel_values = batch['pixel_values'].to(self.device)
                        labels = batch['labels'].to(self.device)

                        outputs = self.model(pixel_values)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item()

                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                # 计算准确率
                train_accuracy = 100 * train_correct / train_total
                val_accuracy = 100 * val_correct / val_total

                # 记录历史
                avg_train_loss = train_loss / len(self.train_loader)
                avg_val_loss = val_loss / len(self.val_loader)

                history['train_loss'].append(avg_train_loss)
                history['val_loss'].append(avg_val_loss)
                history['train_accuracy'].append(train_accuracy)
                history['val_accuracy'].append(val_accuracy)

                print(f'Epoch {epoch + 1}/{epochs}:')
                print(f'  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%')
                print(f'  验证损失: {avg_val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%')
                print('-' * 50)

            # 保存模型
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'class_names': self.class_names
            }, save_path)
            print(f"模型已保存到: {save_path}")

            return history


# %%
def learning_rate_ablation(data_path, class_names, lr_list=[1e-5, 2e-5, 5e-5, 1e-4]):
    """比较不同学习率的效果"""

    lr_results = {}

    for lr in lr_list:
        print(f"\n{'=' * 50}")
        print(f"测试学习率: {lr}")
        print(f"{'=' * 50}")

        # 重新初始化模型（确保公平比较）
        model = PlantDiseaseViT(num_classes=len(class_names))
        train_loader, val_loader, _ = prepare_data_loaders(data_path)

        trainer = ViTTrainer(model.model, train_loader, val_loader, class_names, learning_rate=lr)

        # 快速训练3个epochs进行评估
        history = trainer.train(epochs=3, save_path=f'vit_lr_{lr}.pth')

        lr_results[f'lr_{lr}'] = {
            'final_val_acc': history['val_accuracy'][-1],
            'val_acc_history': history['val_accuracy'],
            'train_acc_history': history['train_accuracy']
        }

        print(f"学习率 {lr} 最终验证准确率: {history['val_accuracy'][-1]:.2f}%")

    return lr_results


# %%
lr_results = learning_rate_ablation(data_path, class_names)


# 单元格4：可视化学习率结果
def plot_lr_comparison(lr_results):
    """绘制学习率对比图"""
    lr_names = list(lr_results.keys())
    final_accs = [lr_results[lr]['final_val_acc'] for lr in lr_names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(lr_names, final_accs, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    plt.title('不同学习率对验证准确率的影响')
    plt.ylabel('验证准确率 (%)')
    plt.xlabel('学习率')

    # 在柱子上添加数值
    for bar, acc in zip(bars, final_accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{acc:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


plot_lr_comparison(lr_results)


# %%
def model_architecture_ablation(data_path, class_names):
    """比较不同ViT架构"""

    model_configs = {
        'ViT-Base': 'google/vit-base-patch16-224',
        'ViT-Large': 'google/vit-large-patch16-224',
        'DeiT-Base': 'facebook/deit-base-patch16-224'
    }

    arch_results = {}

    for name, model_name in model_configs.items():
        print(f"\n{'=' * 50}")
        print(f"测试模型: {name}")
        print(f"{'=' * 50}")

        try:
            model = PlantDiseaseViT(num_classes=len(class_names), model_name=model_name)
            train_loader, val_loader, _ = prepare_data_loaders(data_path)

            trainer = ViTTrainer(model.model, train_loader, val_loader, class_names)

            # 训练5个epochs
            history = trainer.train(epochs=5, save_path=f'{name}_model.pth')

            arch_results[name] = {
                'final_val_acc': history['val_accuracy'][-1],
                'val_acc_history': history['val_accuracy'],
                'params': sum(p.numel() for p in model.model.parameters())
            }

            print(f"{name} 最终验证准确率: {history['val_accuracy'][-1]:.2f}%")
            print(f"{name} 参数量: {arch_results[name]['params']:,}")

        except Exception as e:
            print(f"训练 {name} 时出错: {e}")
            arch_results[name] = None

    return arch_results


# 运行架构消融
arch_results = model_architecture_ablation(data_path, class_names)