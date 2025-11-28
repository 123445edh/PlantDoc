# advanced_augmentation.py
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("警告: albumentations 库不可用，将使用基础增强")


class AdvancedAugmentation:
    """
    高级数据增强模块，集成多种现代增强技术
    完全修复了 albumentations API 问题
    """

    def __init__(self, augmentation_level='medium'):
        """
        初始化增强器

        参数:
            augmentation_level: 增强强度 ('light', 'medium', 'heavy')
        """
        self.augmentation_level = augmentation_level

        if ALBUMENTATIONS_AVAILABLE:
            self.albumentations_transform = self._get_albumentations_transform()
        else:
            self.albumentations_transform = None
            print("使用基础数据增强 (albumentations 不可用)")

    def _get_albumentations_transform(self):
        """获取Albumentations增强流水线 - 完全修复API问题"""
        if not ALBUMENTATIONS_AVAILABLE:
            return None

        try:
            if self.augmentation_level == 'light':
                return A.Compose([
                    A.HorizontalFlip(p=0.3),
                    A.RandomRotate90(p=0.3),
                    A.RandomBrightnessContrast(p=0.2),
                    A.GaussNoise(p=0.2),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])

            elif self.augmentation_level == 'medium':
                return A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.3),
                    A.RandomRotate90(p=0.5),
                    A.Affine(
                        translate_percent=(0.05, 0.05),
                        scale=(0.9, 1.1),
                        rotate=(-15, 15),
                        p=0.5
                    ),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
                    A.GaussNoise(p=0.3),
                    # 使用安全的变换替代 Dropout
                    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])

            else:  # heavy
                return A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Affine(
                        translate_percent=(0.1, 0.1),
                        scale=(0.8, 1.2),
                        rotate=(-30, 30),
                        p=0.7
                    ),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                    A.GaussNoise(p=0.4),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                    A.Blur(blur_limit=3, p=0.3),
                    A.MotionBlur(blur_limit=5, p=0.2),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
        except Exception as e:
            print(f"Albumentations 变换创建失败: {e}")
            return None

    def albumentations_augment(self, image):
        """使用Albumentations进行增强"""
        if self.albumentations_transform is None:
            return self._basic_augment(image)

        try:
            if isinstance(image, torch.Tensor):
                # 如果是tensor，先转换为PIL图像
                image = transforms.ToPILImage()(image)

            if isinstance(image, Image.Image):
                image = np.array(image)

            augmented = self.albumentations_transform(image=image)
            return augmented['image']
        except Exception as e:
            print(f"Albumentations 增强失败: {e}")
            return self._basic_augment(image)

    def _basic_augment(self, image):
        """基础增强替代方案"""
        if isinstance(image, torch.Tensor):
            return image  # 已经是tensor，直接返回

        if isinstance(image, Image.Image):
            # 转换为tensor
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return transform(image)

        return image

    def cutmix(self, batch, alpha=1.0):
        """
        CutMix数据增强
        参考: https://arxiv.org/abs/1905.04899
        """
        images, labels = batch['pixel_values'], batch['labels']

        indices = torch.randperm(images.size(0))
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]

        lam = np.random.beta(alpha, alpha)

        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = shuffled_images[:, :, bbx1:bbx2, bby1:bby2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

        return {
            'pixel_values': images,
            'labels': labels,
            'labels_b': shuffled_labels,
            'lam': lam
        }

    def mixup(self, batch, alpha=1.0):
        """
        MixUp数据增强
        参考: https://arxiv.org/abs/1710.09412
        """
        images, labels = batch['pixel_values'], batch['labels']

        indices = torch.randperm(images.size(0))
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]

        lam = np.random.beta(alpha, alpha)

        mixed_images = lam * images + (1 - lam) * shuffled_images

        return {
            'pixel_values': mixed_images,
            'labels': labels,
            'labels_b': shuffled_labels,
            'lam': lam
        }

    def _rand_bbox(self, size, lam):
        """生成随机边界框"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def gridmask(self, image, d1=10, d2=20, rotate=1, ratio=0.6):
        """
        GridMask数据增强
        参考: https://arxiv.org/abs/2001.04086
        """
        if isinstance(image, torch.Tensor):
            # 如果是tensor，先转换为PIL图像
            image = transforms.ToPILImage()(image)

        if isinstance(image, Image.Image):
            image = np.array(image)

        h, w = image.shape[:2]

        # 创建网格掩码
        mask = np.ones((h, w), np.float32)

        d = np.random.randint(d1, d2)
        r = int(d * ratio)

        for _ in range(rotate):
            for x in range(0, w, d):
                for y in range(0, h, d):
                    x1 = max(0, x - r)
                    y1 = max(0, y - r)
                    x2 = min(w, x + r)
                    y2 = min(h, y + r)
                    mask[y1:y2, x1:x2] = 0

        # 应用掩码
        if len(image.shape) == 3:
            mask = np.expand_dims(mask, axis=2)

        augmented_image = image * mask
        return Image.fromarray(augmented_image.astype(np.uint8))


class AdaptiveAugmentation:
    """
    自适应数据增强 - 根据模型训练状态动态调整增强强度
    修复了方法委托问题
    """

    def __init__(self, initial_strength=0.1, max_strength=0.8):
        self.strength = initial_strength
        self.max_strength = max_strength
        self.augmenter = AdvancedAugmentation('medium')

    def update_strength(self, train_accuracy, val_accuracy):
        """根据准确率差距更新增强强度"""
        gap = train_accuracy - val_accuracy

        if gap > 0.1:  # 过拟合
            self.strength = min(self.strength + 0.05, self.max_strength)
        elif gap < 0.02:  # 欠拟合
            self.strength = max(self.strength - 0.05, 0.1)

    def get_current_transform(self):
        """获取当前强度的变换"""
        if self.strength < 0.3:
            return self.augmenter.albumentations_augment
        elif self.strength < 0.6:
            # 中等强度：Albumentations + 随机GridMask
            def medium_transform(image):
                if random.random() < 0.3:
                    image = self.augmenter.gridmask(image)
                return self.augmenter.albumentations_augment(image)

            return medium_transform
        else:
            # 高强度：Albumentations + 更多增强
            def heavy_transform(image):
                if random.random() < 0.5:
                    image = self.augmenter.gridmask(image)
                return self.augmenter.albumentations_augment(image)

            return heavy_transform

    # 委托方法给内部的augmenter
    def cutmix(self, batch, alpha=1.0):
        return self.augmenter.cutmix(batch, alpha)

    def mixup(self, batch, alpha=1.0):
        return self.augmenter.mixup(batch, alpha)

    def albumentations_augment(self, image):
        return self.augmenter.albumentations_augment(image)

    def gridmask(self, image, d1=10, d2=20, rotate=1, ratio=0.6):
        return self.augmenter.gridmask(image, d1, d2, rotate, ratio)


# 直接运行测试
def test_augmentation():
    """测试增强功能"""
    print("测试数据增强功能...")

    # 创建测试图像
    test_image = Image.new('RGB', (224, 224), color='red')

    # 测试各种增强策略
    strategies = ['light', 'medium', 'heavy']

    for strategy in strategies:
        print(f"\n测试 {strategy} 增强:")
        augmenter = AdvancedAugmentation(strategy)

        # 测试CutMix
        try:
            test_batch = {
                'pixel_values': torch.randn(4, 3, 224, 224),
                'labels': torch.tensor([0, 1, 2, 3])
            }
            cutmix_batch = augmenter.cutmix(test_batch)
            print(f"  ✅ CutMix 测试成功")
        except Exception as e:
            print(f"  ❌ CutMix 测试失败: {e}")

        # 测试MixUp
        try:
            mixup_batch = augmenter.mixup(test_batch)
            print(f"  ✅ MixUp 测试成功")
        except Exception as e:
            print(f"  ❌ MixUp 测试失败: {e}")

        # 测试Albumentations
        try:
            aug_img = augmenter.albumentations_augment(test_image)
            print(f"  ✅ Albumentations 测试成功")
        except Exception as e:
            print(f"  ❌ Albumentations 测试失败: {e}")

        # 测试GridMask
        try:
            grid_img = augmenter.gridmask(test_image)
            print(f"  ✅ GridMask 测试成功")
        except Exception as e:
            print(f"  ❌ GridMask 测试失败: {e}")

    print("\n测试自适应增强:")
    adaptive_aug = AdaptiveAugmentation()
    print(f"初始强度: {adaptive_aug.strength}")

    # 测试自适应增强的方法委托
    try:
        test_batch = {
            'pixel_values': torch.randn(4, 3, 224, 224),
            'labels': torch.tensor([0, 1, 2, 3])
        }
        cutmix_batch = adaptive_aug.cutmix(test_batch)
        print(f"  ✅ 自适应增强 CutMix 测试成功")
    except Exception as e:
        print(f"  ❌ 自适应增强 CutMix 测试失败: {e}")

    adaptive_aug.update_strength(0.9, 0.7)  # 模拟过拟合
    print(f"更新后强度: {adaptive_aug.strength}")


if __name__ == "__main__":
    test_augmentation()
    print("\n数据增强模块测试完成!")