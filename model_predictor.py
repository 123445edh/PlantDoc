# test_set_evaluator.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import swin_b
import torch.nn as nn
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd
import datetime


class TestSetEvaluator:
    """测试集评估器 - 专门评估模型在测试集上的表现"""

    def __init__(self, model_path='best_model.pth', class_names_path='class_names.txt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载类别名称
        self.class_names = self.load_class_names(class_names_path)
        self.num_classes = len(self.class_names)

        # 创建模型架构
        self.model = self.create_model()

        # 加载训练好的权重
        self.load_model(model_path)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"✅ 模型加载成功! 类别数: {self.num_classes}")

    def load_class_names(self, class_names_path):
        """加载类别名称"""
        try:
            with open(class_names_path, 'r', encoding='utf-8') as f:
                class_names = [line.strip().split(': ')[1] for line in f.readlines()]
            print(f"加载 {len(class_names)} 个病害类别")
            return class_names
        except Exception as e:
            print(f"加载类别名称失败: {e}")
            return [f"Class_{i}" for i in range(27)]

    def create_model(self):
        """创建模型架构"""
        model = swin_b(weights=None)
        model.head = nn.Linear(model.head.in_features, self.num_classes)
        return model.to(self.device)

    def load_model(self, model_path):
        """加载训练好的模型权重"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"✅ 模型权重加载成功")
            if 'val_acc' in checkpoint:
                print(f"📈 模型验证准确率: {checkpoint['val_acc']:.2f}%")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    def load_test_set(self, test_dir='F:\\PlantDoc\\TEST'):
        """加载测试集数据"""
        print(f"📁 加载测试集: {test_dir}")

        test_image_paths = []
        test_labels = []

        # 按照类别目录结构加载
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(test_dir, class_name)
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        test_image_paths.append(os.path.join(class_dir, img_file))
                        test_labels.append(class_idx)
            else:
                print(f"⚠️  警告: 测试集中找不到类别目录 '{class_name}'")

        print(f"📊 测试集统计:")
        print(f"   - 总图像数: {len(test_image_paths)}")
        print(f"   - 类别数: {len(set(test_labels))}")

        # 统计每个类别的图像数量
        class_counts = {}
        for label in test_labels:
            class_name = self.class_names[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print(f"   - 各类别图像分布:")
        for class_name, count in class_counts.items():
            print(f"     {class_name}: {count} 张")

        return test_image_paths, test_labels

    def evaluate_test_set(self, test_dir='F:\\PlantDoc\\TEST'):
        """在测试集上进行完整评估"""
        print("\n" + "=" * 60)
        print("🧪 开始在测试集上进行评估")
        print("=" * 60)

        # 加载测试集
        test_image_paths, true_labels = self.load_test_set(test_dir)

        if len(test_image_paths) == 0:
            print("❌ 测试集为空，无法进行评估")
            return

        # 进行预测
        predictions = []
        predicted_labels = []
        confidences = []

        print(f"\n🔍 进行预测...")
        for i, image_path in enumerate(test_image_paths):
            try:
                # 加载和预处理图像
                image = Image.open(image_path).convert('RGB')
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)

                # 预测
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = F.softmax(outputs[0], dim=0)
                    predicted_idx = torch.argmax(probabilities).item()
                    confidence = probabilities[predicted_idx].item()

                predictions.append({
                    'image_path': image_path,
                    'true_label': true_labels[i],
                    'true_class': self.class_names[true_labels[i]],
                    'predicted_label': predicted_idx,
                    'predicted_class': self.class_names[predicted_idx],
                    'confidence': confidence,
                    'is_correct': (predicted_idx == true_labels[i])
                })

                predicted_labels.append(predicted_idx)
                confidences.append(confidence)

                if (i + 1) % 50 == 0:
                    print(f"   已处理 {i + 1}/{len(test_image_paths)} 张图像")

            except Exception as e:
                print(f"❌ 处理图像失败 {image_path}: {e}")
                continue

        # 计算准确率
        correct_predictions = sum(1 for p in predictions if p['is_correct'])
        test_accuracy = correct_predictions / len(predictions) * 100

        print(f"\n🎯 测试集评估结果:")
        print(f"   - 总图像数: {len(predictions)}")
        print(f"   - 正确预测: {correct_predictions}")
        print(f"   - 测试准确率: {test_accuracy:.2f}%")

        return predictions, test_accuracy, true_labels, predicted_labels, confidences

    def generate_detailed_report(self, predictions, test_accuracy, true_labels, predicted_labels,
                                 output_dir='test_results'):
        """生成详细评估报告"""

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n📊 生成详细评估报告...")

        # 1. 保存预测结果到JSON
        results_file = os.path.join(output_dir, 'test_predictions.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            # 转换numpy类型为Python原生类型
            def convert_types(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                else:
                    return obj

            json.dump(convert_types(predictions), f, indent=2, ensure_ascii=False)

        # 2. 生成分类报告
        class_report = classification_report(true_labels, predicted_labels,
                                             target_names=self.class_names,
                                             output_dict=True)

        # 3. 生成混淆矩阵
        cm = confusion_matrix(true_labels, predicted_labels)

        # 绘制混淆矩阵
        plt.figure(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=self.class_names)
        disp.plot(cmap='Blues', ax=plt.gca(), values_format='d')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.title(f'混淆矩阵 - 测试准确率: {test_accuracy:.2f}%', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 4. 生成准确率分布图（按类别）
        class_accuracies = {}
        for class_idx, class_name in enumerate(self.class_names):
            class_predictions = [p for p in predictions if p['true_label'] == class_idx]
            if class_predictions:
                class_correct = sum(1 for p in class_predictions if p['is_correct'])
                class_accuracies[class_name] = class_correct / len(class_predictions) * 100

        # 绘制类别准确率
        plt.figure(figsize=(14, 8))
        classes = list(class_accuracies.keys())
        accs = list(class_accuracies.values())

        colors = ['green' if acc >= 70 else 'orange' if acc >= 50 else 'red' for acc in accs]

        bars = plt.bar(classes, accs, color=colors, alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('准确率 (%)')
        plt.title('各类别测试准确率', fontsize=16, pad=20)
        plt.axhline(y=test_accuracy, color='red', linestyle='--',
                    label=f'平均准确率: {test_accuracy:.2f}%')
        plt.legend()

        # 在柱状图上添加数值
        for bar, acc in zip(bars, accs):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_accuracy.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 5. 生成置信度分布图
        plt.figure(figsize=(10, 6))
        correct_confidences = [p['confidence'] for p in predictions if p['is_correct']]
        incorrect_confidences = [p['confidence'] for p in predictions if not p['is_correct']]

        plt.hist(correct_confidences, bins=20, alpha=0.7, label='正确预测', color='green')
        plt.hist(incorrect_confidences, bins=20, alpha=0.7, label='错误预测', color='red')
        plt.xlabel('预测置信度')
        plt.ylabel('频次')
        plt.title('正确 vs 错误预测的置信度分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 6. 生成详细报告
        report_file = os.path.join(output_dir, 'test_evaluation_report.md')

        # 计算各类别统计
        class_stats = {}
        for class_idx, class_name in enumerate(self.class_names):
            class_predictions = [p for p in predictions if p['true_label'] == class_idx]
            if class_predictions:
                class_correct = sum(1 for p in class_predictions if p['is_correct'])
                class_accuracy = class_correct / len(class_predictions) * 100
                class_stats[class_name] = {
                    'total': len(class_predictions),
                    'correct': class_correct,
                    'accuracy': class_accuracy
                }

        # 找出最容易混淆的类别对
        error_analysis = self.analyze_errors(predictions)

        report = f"""
# Swin Transformer 测试集评估报告

## 总体性能
- **测试集准确率**: {test_accuracy:.2f}%
- **总图像数量**: {len(predictions)}
- **正确预测数量**: {sum(1 for p in predictions if p['is_correct'])}
- **平均置信度**: {np.mean([p['confidence'] for p in predictions]):.3f}

## 各类别性能

| 类别 | 图像数量 | 正确预测 | 准确率 | 状态 |
|------|----------|----------|--------|------|
"""

        for class_name, stats in class_stats.items():
            status = "✅ 优秀" if stats['accuracy'] >= 80 else "⚠️ 良好" if stats['accuracy'] >= 60 else "❌ 需改进"
            report += f"| {class_name} | {stats['total']} | {stats['correct']} | {stats['accuracy']:.1f}% | {status} |\n"

        report += f"""
## 错误分析

### 最常见的错误预测
"""

        for i, (true_class, pred_class, count) in enumerate(error_analysis[:10]):
            report += f"{i + 1}. **{true_class}** → **{pred_class}**: {count}次\n"

        report += f"""
## 可视化结果

![混淆矩阵](confusion_matrix.png)
![类别准确率](class_accuracy.png)  
![置信度分布](confidence_distribution.png)

## 详细数据
- 完整预测结果: `test_predictions.json`
- 分类报告: 包含精确率、召回率、F1分数
- 混淆矩阵: 显示各类别间的混淆情况

---
*评估时间: { datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✅ 评估报告已生成: {report_file}")
        print(f"✅ 预测结果已保存: {results_file}")
        print(f"✅ 可视化图表已保存到: {output_dir}/")

        return report_file

    def analyze_errors(self, predictions):
        """分析错误预测模式"""
        errors = [p for p in predictions if not p['is_correct']]

        error_patterns = {}
        for error in errors:
            pattern = f"{error['true_class']} -> {error['predicted_class']}"
            error_patterns[pattern] = error_patterns.get(pattern, 0) + 1

        # 按出现次数排序
        sorted_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)

        return [(pattern.split(' -> ')[0], pattern.split(' -> ')[1], count)
                for pattern, count in sorted_errors]

    def run_complete_evaluation(self, test_dir='F:\\PlantDoc\\TEST'):
        """运行完整的测试集评估"""
        print("🚀 开始完整测试集评估流程...")

        # 1. 评估测试集
        predictions, test_accuracy, true_labels, predicted_labels, confidences = \
            self.evaluate_test_set(test_dir)

        # 2. 生成详细报告
        report_file = self.generate_detailed_report(
            predictions, test_accuracy, true_labels, predicted_labels
        )

        # 3. 显示关键结果
        print(f"\n🎉 评估完成!")
        print(f"📈 测试集准确率: {test_accuracy:.2f}%")
        print(f"📁 结果保存目录: test_results/")
        print(f"📄 详细报告: {report_file}")

        return test_accuracy


def quick_evaluation():
    """快速评估测试集"""
    print("🌿 Swin Transformer 测试集评估")
    print("=" * 60)

    evaluator = TestSetEvaluator()

    # 运行完整评估
    test_accuracy = evaluator.run_complete_evaluation()

    # 与验证集准确率比较
    print(f"\n📊 性能对比:")
    print(f"   验证集准确率: 70.13%")
    print(f"   测试集准确率: {test_accuracy:.2f}%")

    if test_accuracy >= 70:
        print("   ✅ 测试集表现优秀，模型泛化能力良好!")
    elif test_accuracy >= 65:
        print("   ⚠️  测试集表现良好，略有下降但可接受")
    else:
        print("   ❌ 测试集表现不佳，可能存在过拟合")


if __name__ == "__main__":
    quick_evaluation()