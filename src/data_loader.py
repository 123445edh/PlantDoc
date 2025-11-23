import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from utils.data_utils import explore_dataset, prepare_data_loaders, get_transforms
import config


class PlantDocDataLoader:
    """
    高级数据加载器，提供数据管理和分析功能
    """

    def __init__(self, data_path=None):
        if data_path is None:
            self.data_path = config.DATA_CONFIG['data_path']
        else:
            self.data_path = Path(data_path)

        self.train_loader = None
        self.val_loader = None
        self.class_names = None
        self.class_to_idx = None
        self.idx_to_class = None

    def load_data(self, batch_size=None, test_size=None):
        """
        加载并准备数据

        返回:
            train_loader, val_loader, class_names
        """
        print("正在加载 PlantDoc 数据集...")

        # 数据探索
        class_counts = explore_dataset(self.data_path)

        # 准备数据加载器
        if batch_size is None:
            batch_size = config.DATA_CONFIG['batch_size']
        if test_size is None:
            test_size = config.DATA_CONFIG['test_size']

        # 修正：移除 random_state 参数
        self.train_loader, self.val_loader, self.class_names, self.class_to_idx, self.idx_to_class = \
            prepare_data_loaders(
                self.data_path,
                batch_size=batch_size,
                test_size=test_size
                # 移除 random_state=config.DATA_CONFIG['random_state']
            )

        print("数据加载完成！")
        self._print_data_summary()

        return self.train_loader, self.val_loader, self.class_names

    def _print_data_summary(self):
        """打印数据摘要"""
        if self.train_loader is None:
            print("请先调用 load_data() 方法加载数据")
            return

        print("\n" + "=" * 50)
        print("数据摘要")
        print("=" * 50)
        print(f"类别数量: {len(self.class_names)}")
        print(f"训练样本: {len(self.train_loader.dataset)}")
        print(f"验证样本: {len(self.val_loader.dataset)}")
        print(f"批次大小: {self.train_loader.batch_size}")
        print(f"总批次 (训练): {len(self.train_loader)}")
        print(f"总批次 (验证): {len(self.val_loader)}")

    def get_class_distribution(self):
        """获取类别分布信息"""
        if self.train_loader is None:
            print("请先调用 load_data() 方法加载数据")
            return None

        # 统计训练集类别分布
        train_labels = []
        for batch in self.train_loader:
            train_labels.extend(batch['labels'].numpy())

        # 统计验证集类别分布
        val_labels = []
        for batch in self.val_loader:
            val_labels.extend(batch['labels'].numpy())

        train_counts = np.bincount(train_labels)
        val_counts = np.bincount(val_labels)

        distribution = {
            'train': {self.class_names[i]: count for i, count in enumerate(train_counts)},
            'val': {self.class_names[i]: count for i, count in enumerate(val_counts)}
        }

        return distribution

    def visualize_sample_batch(self, num_samples=8):
        """可视化一个批次的样本"""
        if self.train_loader is None:
            print("请先调用 load_data() 方法加载数据")
            return

        # 获取一个批次
        batch = next(iter(self.train_loader))
        images = batch['pixel_values']
        labels = batch['labels']

        # 转换为适合显示的格式
        images = images.cpu().numpy()
        labels = labels.cpu().numpy()

        # 显示图像
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()

        for i in range(min(num_samples, len(images))):
            # 反标准化图像
            img = images[i].transpose(1, 2, 0)
            img = img * 0.5 + 0.5  # 反标准化
            img = np.clip(img, 0, 1)

            axes[i].imshow(img)
            axes[i].set_title(f"{self.class_names[labels[i]]}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def get_data_info(self):
        """获取完整的数据信息"""
        if self.train_loader is None:
            return {"status": "数据未加载"}

        distribution = self.get_class_distribution()

        info = {
            "status": "数据已加载",
            "data_path": str(self.data_path),
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "train_samples": len(self.train_loader.dataset),
            "val_samples": len(self.val_loader.dataset),
            "batch_size": self.train_loader.batch_size,
            "class_distribution": distribution
        }

        return info


# 使用示例
if __name__ == "__main__":
    loader = PlantDocDataLoader()
    train_loader, val_loader, class_names = loader.load_data()
    print(loader.get_data_info())