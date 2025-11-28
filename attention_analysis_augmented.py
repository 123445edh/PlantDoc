# attention_analysis_augmented.py
import os
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from data_preparation import class_names, val_loader
    from model_training import PlantDiseaseViT
except ImportError as e:
    print(f"导入失败: {e}")
    # 备用定义
    class_names = [f"Class_{i}" for i in range(27)]

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AugmentedModelAttentionAnalyzer:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        self.device = next(model.parameters()).device

    def analyze_predictions(self, val_loader):
        """分析模型预测结果"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_confidences = []
        all_attention_maps = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
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

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                confidence = torch.softmax(outputs, dim=1).max(dim=1)[0]

                # 转换为 Python 原生类型以避免序列化问题
                all_predictions.extend(predicted.cpu().numpy().astype(int).tolist())
                all_labels.extend(labels.cpu().numpy().astype(int).tolist())
                all_confidences.extend(confidence.cpu().numpy().astype(float).tolist())

        return all_predictions, all_labels, all_confidences

    def analyze_failure_cases(self, predictions, true_labels, confidences):
        """分析失败案例"""
        incorrect_indices = [i for i, (pred, true) in enumerate(zip(predictions, true_labels))
                             if pred != true]

        print(f"总样本数: {len(predictions)}")
        print(f"错误预测数: {len(incorrect_indices)}")
        accuracy = (1 - len(incorrect_indices) / len(predictions)) * 100
        print(f"准确率: {accuracy:.2f}%")

        # 分析错误模式
        error_patterns = {}
        for idx in incorrect_indices:
            true_class = self.class_names[true_labels[idx]]
            pred_class = self.class_names[predictions[idx]]
            pattern = f"{true_class} → {pred_class}"
            error_patterns[pattern] = error_patterns.get(pattern, 0) + 1

        print("\n常见错误模式:")
        for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {pattern}: {count}次")

        return incorrect_indices, error_patterns

    def compare_with_baseline(self, baseline_results_path='attention_analysis_results.json'):
        """与基线模型比较"""
        try:
            with open(baseline_results_path, 'r') as f:
                baseline_results = json.load(f)

            baseline_accuracy = (1 - len(baseline_results.get('incorrect_indices', [])) /
                                 len(baseline_results.get('predictions', []))) * 100

            current_accuracy = (1 - len(self.incorrect_indices) / len(self.predictions)) * 100

            comparison = {
                'baseline_accuracy': baseline_accuracy,
                'augmented_accuracy': current_accuracy,
                'improvement': current_accuracy - baseline_accuracy,
                'baseline_error_patterns': baseline_results.get('error_patterns', {}),
                'augmented_error_patterns': self.error_patterns
            }

            return comparison
        except FileNotFoundError:
            print("基线结果文件不存在，跳过比较")
            return None


def load_augmented_model(model_path='augmented_model.pth', num_classes=27):
    """加载数据增强训练的模型"""
    print(f"加载数据增强模型: {model_path}")

    # 创建模型结构
    model = PlantDiseaseViT(num_classes=num_classes)

    # 加载训练好的权重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.model.load_state_dict(checkpoint)
        print(f"数据增强模型权重加载成功!")
    else:
        # 尝试其他可能的模型文件
        alternative_models = ['enhanced_model.pth', 'best_augmented_model.pth']
        for alt_model in alternative_models:
            if os.path.exists(alt_model):
                print(f"使用替代模型: {alt_model}")
                checkpoint = torch.load(alt_model, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.model.load_state_dict(checkpoint)
                break
        else:
            print(f"警告: 数据增强模型文件 {model_path} 不存在")
            return None

    return model


# 自定义 JSON 编码器处理 NumPy 类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def analyze_augmented_model_performance():
    """分析数据增强模型的性能"""
    print("开始分析数据增强模型...")

    # 加载数据增强模型
    model = load_augmented_model('augmented_model.pth', len(class_names))
    if model is None:
        print("无法加载数据增强模型，退出分析")
        return

    # 创建分析器
    analyzer = AugmentedModelAttentionAnalyzer(model.model, class_names)

    # 分析预测结果
    predictions, true_labels, confidences = analyzer.analyze_predictions(val_loader)

    # 分析失败案例
    incorrect_indices, error_patterns = analyzer.analyze_failure_cases(
        predictions, true_labels, confidences
    )

    # 保存分析器实例的属性用于比较
    analyzer.predictions = predictions
    analyzer.true_labels = true_labels
    analyzer.incorrect_indices = incorrect_indices
    analyzer.error_patterns = error_patterns

    # 与基线模型比较
    comparison = analyzer.compare_with_baseline()

    # 保存结果
    results = {
        'model_type': 'augmented',
        'predictions': predictions,
        'true_labels': true_labels,
        'confidences': confidences,
        'incorrect_indices': incorrect_indices,
        'error_patterns': error_patterns,
        'comparison_with_baseline': comparison,
        'analysis_timestamp': np.datetime64('now').astype(str)
    }

    with open('attention_analysis_augmented_results.json', 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print("数据增强模型注意力分析完成！结果已保存")

    # 生成混淆矩阵
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    accuracy = (1 - len(incorrect_indices) / len(predictions)) * 100
    plt.title(f'数据增强模型混淆矩阵 (准确率: {accuracy:.2f}%)', fontsize=16)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('augmented_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印详细分类报告
    print("\n详细分类报告:")
    print(classification_report(true_labels, predictions, target_names=class_names))

    # 保存分类报告到文本文件
    report = classification_report(true_labels, predictions, target_names=class_names, output_dict=True)
    with open('augmented_classification_report.txt', 'w') as f:
        f.write("数据增强模型分类报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"准确率: {accuracy:.2f}%\n")
        f.write(f"验证集样本数: {len(predictions)}\n")
        f.write(f"错误预测数: {len(incorrect_indices)}\n")

        if comparison:
            f.write(f"相对于基线的改进: {comparison['improvement']:.2f}%\n")

        f.write("\n详细报告:\n")
        f.write(classification_report(true_labels, predictions, target_names=class_names))

    # 如果存在比较结果，显示改进情况
    if comparison:
        print(f"\n与基线模型比较:")
        print(f"  基线准确率: {comparison['baseline_accuracy']:.2f}%")
        print(f"  增强准确率: {comparison['augmented_accuracy']:.2f}%")
        print(f"  改进: {comparison['improvement']:+.2f}%")

    print("分类报告已保存到 augmented_classification_report.txt")


# 主执行代码
if __name__ == "__main__":
    analyze_augmented_model_performance()