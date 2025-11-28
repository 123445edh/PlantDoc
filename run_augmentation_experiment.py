# run_augmentation_experiment.py
import json
import torch
from pathlib import Path
import sys
from datetime import datetime
import os

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from advanced_augmentation import AdvancedAugmentation, AdaptiveAugmentation


def prepare_data_loaders(data_path):
    """直接使用 data_preparation.py 中的函数"""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import torchvision.transforms as transforms
    from sklearn.model_selection import train_test_split
    import os

    # 收集图像路径和标签
    image_paths = []
    labels = []
    class_names = []

    train_path = os.path.join(data_path, "train")

    if os.path.exists(train_path):
        class_names = sorted([d for d in os.listdir(train_path)
                              if os.path.isdir(os.path.join(train_path, d))])
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = os.path.join(train_path, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, img_file))
                    labels.append(class_to_idx[class_name])

    if len(image_paths) == 0:
        raise ValueError("没有找到训练图像文件")

    print(f"找到 {len(image_paths)} 张图像，{len(class_names)} 个类别")

    # 划分训练集和验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"训练集: {len(train_paths)} 张图像")
    print(f"验证集: {len(val_paths)} 张图像")

    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据集类
    class PlantDocDataset(Dataset):
        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return {
                'pixel_values': image,
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }

    # 创建数据加载器
    train_dataset = PlantDocDataset(train_paths, train_labels, transform)
    val_dataset = PlantDocDataset(val_paths, val_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    return train_loader, val_loader, class_names


def run_augmentation_experiment():
    """运行数据增强实验并生成JSON结果"""
    print("开始数据增强实验...")

    try:
        # 直接使用数据路径
        data_path = "F:/PlantDoc"  # 根据你的实际路径修改

        # 准备数据
        train_loader, val_loader, class_names = prepare_data_loaders(data_path)

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return {
            'error': f'数据加载失败: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }

    # 初始化增强器
    augmenters = {
        'basic': AdvancedAugmentation('light'),
        'advanced': AdvancedAugmentation('medium'),
        'heavy': AdvancedAugmentation('heavy'),
        'adaptive': AdaptiveAugmentation()
    }

    experiment_results = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'dataset': 'PlantDoc',
            'num_classes': len(class_names),
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'status': 'success'
        },
        'augmentation_strategies': {},
        'performance_comparison': {}
    }

    # 测试增强器功能
    print("\n测试增强器功能...")

    for strategy_name, augmenter in augmenters.items():
        print(f"\n测试增强策略: {strategy_name}")

        try:
            # 获取一个测试批次
            test_batch = next(iter(train_loader))

            # 测试各种增强方法
            test_results = {}

            # 测试CutMix
            try:
                cutmix_batch = augmenter.cutmix(test_batch)
                test_results['cutmix'] = 'success'
                print(f"  ✅ CutMix测试成功")
            except Exception as e:
                test_results['cutmix'] = f'failed: {str(e)}'
                print(f"  ❌ CutMix测试失败: {e}")

            # 测试MixUp
            try:
                mixup_batch = augmenter.mixup(test_batch)
                test_results['mixup'] = 'success'
                print(f"  ✅ MixUp测试成功")
            except Exception as e:
                test_results['mixup'] = f'failed: {str(e)}'
                print(f"  ❌ MixUp测试失败: {e}")

            # 测试Albumentations
            try:
                test_image = test_batch['pixel_values'][0]
                augmented_image = augmenter.albumentations_augment(test_image)
                test_results['albumentations'] = 'success'
                print(f"  ✅ Albumentations测试成功")
            except Exception as e:
                test_results['albumentations'] = f'failed: {str(e)}'
                print(f"  ❌ Albumentations测试失败: {e}")

            # 记录结果
            experiment_results['augmentation_strategies'][strategy_name] = {
                'status': 'success',
                'augmentation_params': get_augmentation_params(augmenter),
                'test_results': test_results
            }

        except Exception as e:
            print(f"  ❌ {strategy_name} 测试失败: {e}")
            experiment_results['augmentation_strategies'][strategy_name] = {
                'status': 'failed',
                'error': str(e)
            }

    # 性能比较分析
    analyze_augmentation_performance(experiment_results)

    # 保存结果
    output_file = 'augmentation_experiment_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_results, f, indent=2, ensure_ascii=False)

    print(f"\n数据增强实验完成！结果已保存到: {output_file}")
    return experiment_results


def get_augmentation_params(augmenter):
    """获取增强器参数"""
    params = {
        'level': getattr(augmenter, 'augmentation_level', 'unknown'),
        'type': augmenter.__class__.__name__
    }

    if hasattr(augmenter, 'strength'):
        params.update({
            'strength': augmenter.strength,
            'max_strength': augmenter.max_strength
        })

    return params


def analyze_augmentation_performance(results):
    """分析增强性能"""
    strategies = results['augmentation_strategies']
    successful_strategies = [k for k, v in strategies.items() if v.get('status') == 'success']

    # 统计成功率
    total_tests = 0
    passed_tests = 0

    for strategy_name, strategy_data in strategies.items():
        if strategy_data.get('status') == 'success':
            test_results = strategy_data.get('test_results', {})
            for test_name, result in test_results.items():
                total_tests += 1
                if result == 'success':
                    passed_tests += 1

    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    results['performance_comparison'] = {
        'total_strategies': len(strategies),
        'successful_strategies': len(successful_strategies),
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': f"{success_rate:.1f}%",
        'recommendation': '所有增强策略都可用于后续训练' if success_rate > 80 else '部分增强策略需要调试'
    }


if __name__ == "__main__":
    print("=" * 60)
    print("数据增强实验")
    print("=" * 60)

    results = run_augmentation_experiment()

    # 打印摘要
    print("\n" + "=" * 50)
    print("数据增强实验摘要")
    print("=" * 50)

    if 'error' in results:
        print(f"实验失败: {results['error']}")
    else:
        comparison = results.get('performance_comparison', {})
        print(f"测试策略数量: {comparison.get('total_strategies', 'N/A')}")
        print(f"成功策略数量: {comparison.get('successful_strategies', 'N/A')}")
        print(f"测试通过率: {comparison.get('success_rate', 'N/A')}")
        print(f"建议: {comparison.get('recommendation', 'N/A')}")

        # 显示各策略的测试结果
        print("\n各策略详细结果:")
        for strategy_name, strategy_data in results.get('augmentation_strategies', {}).items():
            status_icon = "✅" if strategy_data.get('status') == 'success' else "❌"
            print(f"  {status_icon} {strategy_name}: {strategy_data.get('status', 'unknown')}")
            if 'test_results' in strategy_data:
                for test_name, result in strategy_data['test_results'].items():
                    test_icon = "✅" if result == 'success' else "❌"
                    print(f"    {test_icon} {test_name}: {result}")