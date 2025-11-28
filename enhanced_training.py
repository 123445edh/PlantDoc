# enhanced_training.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import sys
from pathlib import Path
import random

from data_preparation import prepare_data_loaders, data_path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    # 尝试导入离线版本的模型改进分析
    from model_improvement_analysis import OfflineViTModel, ModelAnalyzer

    IMPROVEMENT_ANALYSIS_AVAILABLE = True
    print("使用离线模型改进分析模块")
except ImportError:
    print("警告: 无法导入模型改进分析模块")
    IMPROVEMENT_ANALYSIS_AVAILABLE = False

try:
    # 导入高级数据增强
    from advanced_augmentation import AdvancedAugmentation, AdaptiveAugmentation

    AUGMENTATION_AVAILABLE = True
    print("使用高级数据增强模块")
except ImportError:
    print("警告: 无法导入高级数据增强模块")
    AUGMENTATION_AVAILABLE = False


class EnhancedViTTrainer:
    """
    增强的ViT训练器，集成高级数据增强和模型改进
    """

    def __init__(self, base_trainer, augmentation_strategy='adaptive'):
        self.base_trainer = base_trainer

        if AUGMENTATION_AVAILABLE:
            if augmentation_strategy == 'adaptive':
                self.augmenter = AdaptiveAugmentation()
            else:
                self.augmenter = AdvancedAugmentation(augmentation_strategy)
        else:
            self.augmenter = None
            print("警告: 数据增强不可用，使用基础训练")

    def train_with_advanced_augmentation(self, epochs=10, use_cutmix=True, use_mixup=True):
        """使用高级数据增强进行训练"""

        if self.augmenter is None:
            print("数据增强不可用，使用基础训练方法")
            return self.base_trainer.train(epochs=epochs)

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 50)

            # 训练阶段
            self.base_trainer.model.train()
            running_loss = 0.0
            running_corrects = 0

            for batch_idx, batch in enumerate(tqdm(self.base_trainer.train_loader, desc='Training')):
                # 应用高级数据增强
                if use_cutmix and random.random() < 0.5:
                    batch = self.augmenter.cutmix(batch)
                elif use_mixup and random.random() < 0.5:
                    batch = self.augmenter.mixup(batch)

                # 正常训练步骤
                self.base_trainer.optimizer.zero_grad()

                if 'lam' in batch:  # CutMix或MixUp批次
                    outputs = self.base_trainer.model(batch['pixel_values'])
                    loss = self._mixed_loss(
                        outputs.logits,
                        batch['labels'],
                        batch['labels_b'],
                        batch['lam']
                    )
                else:
                    outputs = self.base_trainer.model(
                        pixel_values=batch['pixel_values'],
                        labels=batch['labels']
                    )
                    loss = outputs.loss

                loss.backward()
                self.base_trainer.optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs.logits, 1)
                running_corrects += torch.sum(preds == batch['labels'].data)

            epoch_loss = running_loss / len(self.base_trainer.train_loader)
            epoch_acc = running_corrects.double() / len(self.base_trainer.train_loader.dataset)

            print(f'训练损失: {epoch_loss:.4f}, 训练准确率: {epoch_acc:.4f}')

            # 验证阶段
            val_loss, val_acc = self.base_trainer.validate()
            print(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}')

            # 更新自适应增强强度
            if hasattr(self.augmenter, 'update_strength'):
                self.augmenter.update_strength(epoch_acc, val_acc)

    def _mixed_loss(self, outputs, labels_a, labels_b, lam):
        """混合损失函数，用于CutMix和MixUp"""
        criterion = nn.CrossEntropyLoss()
        return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

    def evaluate_improvements(self, test_loader):
        """评估不同改进的效果"""
        if not IMPROVEMENT_ANALYSIS_AVAILABLE:
            print("模型改进分析不可用")
            return None

        print("开始模型改进评估...")

        improver = OfflineViTModel(num_classes=len(self.base_trainer.class_names))
        analyzer = ModelAnalyzer()

        # 创建不同改进版本的模型
        models = {
            'baseline': self.base_trainer.model,
            'multi_scale': improver.create_attention_improvement('multi_scale'),
            'residual_attention': improver.create_attention_improvement('residual'),
            'channel_wise': improver.create_attention_improvement('channel_wise')
        }

        # 分析每个模型
        results = {}
        for name, model in models.items():
            print(f"评估 {name}...")
            model.to(self.base_trainer.device)
            model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_loader:
                    outputs = model(pixel_values=batch['pixel_values'].to(self.base_trainer.device))
                    _, predicted = torch.max(outputs.logits, 1)
                    total += batch['labels'].size(0)
                    correct += (predicted == batch['labels'].to(self.base_trainer.device)).sum().item()

            accuracy = 100 * correct / total
            results[name] = {'accuracy': accuracy}

        # 复杂度分析
        dummy_input = torch.randn(1, 3, 224, 224).to(self.base_trainer.device)
        complexity_results = analyzer.compare_architectures(models, dummy_input)

        # 合并结果
        for name in results:
            if name in complexity_results:
                results[name].update(complexity_results[name])

        # 生成报告
        report = analyzer.generate_improvement_report(results)

        with open('enhanced_training_analysis.json', 'w') as f:
            json.dump(report, f, indent=2)

        print("改进评估完成！结果已保存到 enhanced_training_analysis.json")
        return report


# 使用示例和测试函数
def test_enhanced_training():
    """测试增强训练功能"""
    print("测试增强训练功能...")

    try:
        # 尝试导入必要的模块
        from src.data_loader import PlantDocDataLoader
        from model_training import PlantDiseaseViT, ViTTrainer

        # 加载数据
        data_loader = PlantDocDataLoader()
        train_loader, val_loader, class_names = prepare_data_loaders(data_path)

        # 创建基础模型和训练器
        model = PlantDiseaseViT(num_classes=len(class_names))
        base_trainer = ViTTrainer(
            model.model, train_loader, val_loader, class_names,
            learning_rate=2e-5, device=model.device
        )

        # 创建增强训练器
        enhanced_trainer = EnhancedViTTrainer(base_trainer, augmentation_strategy='advanced')

        print("增强训练器创建成功！")

        # 测试增强训练（只运行1个epoch作为测试）
        print("测试增强训练...")
        enhanced_trainer.train_with_advanced_augmentation(epochs=1)

        # 测试改进评估
        print("测试改进评估...")
        enhanced_trainer.evaluate_improvements(val_loader)

        print("所有测试完成！")

    except Exception as e:
        print(f"测试失败: {e}")
        return False

    return True


if __name__ == "__main__":
    # 运行测试
    success = test_enhanced_training()

    if success:
        print("\n增强训练模块测试成功！")
        print("现在你可以使用以下方式运行完整训练:")
        print("1. 创建基础训练器")
        print("2. 创建EnhancedViTTrainer实例")
        print("3. 调用train_with_advanced_augmentation()进行训练")
        print("4. 调用evaluate_improvements()评估模型改进")
    else:
        print("\n增强训练模块测试失败，请检查依赖模块")