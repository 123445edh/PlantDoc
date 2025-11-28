# model_improvement_analysis_offline.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List
import json

try:
    from transformers import ViTForImageClassification, ViTConfig

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers 库不可用，使用离线模式")


class OfflineViTModel:
    """
    离线ViT模型分析类
    避免网络连接问题
    """

    def __init__(self, num_classes=27):
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_baseline_model(self):
        """创建基础ViT模型"""
        if TRANSFORMERS_AVAILABLE:
            try:
                # 尝试在线加载
                config = ViTConfig.from_pretrained(
                    'google/vit-base-patch16-224',
                    num_labels=self.num_classes
                )
                return ViTForImageClassification(config)
            except:
                print("在线加载失败，使用离线模式")
                return self._create_offline_model()
        else:
            return self._create_offline_model()

    def _create_offline_model(self):
        """创建离线模型"""

        class OfflineViT(nn.Module):
            def __init__(self, num_classes=27):
                super().__init__()

                # 使用torchvision的ViT
                import torchvision.models as models
                self.vit = models.vit_b_16(pretrained=False)

                # 修改分类头
                in_features = self.vit.heads.head.in_features
                self.vit.heads.head = nn.Linear(in_features, num_classes)

            def forward(self, pixel_values, labels=None):
                outputs = self.vit(pixel_values)

                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(outputs, labels)
                    return type('Output', (), {'loss': loss, 'logits': outputs})
                else:
                    return type('Output', (), {'loss': None, 'logits': outputs})

        return OfflineViT(self.num_classes)

    def create_attention_improvement(self, attention_type='multi_scale'):
        """
        创建注意力机制改进版本 - 离线版
        """
        if attention_type == 'multi_scale':
            return self._create_multi_scale_attention()
        elif attention_type == 'residual':
            return self._create_residual_attention()
        elif attention_type == 'channel_wise':
            return self._create_channel_wise_attention()
        else:
            return self.create_baseline_model()

    def _create_multi_scale_attention(self):
        """创建多尺度注意力ViT - 离线版"""

        class MultiScaleViT(nn.Module):
            def __init__(self, num_classes=27):
                super().__init__()

                # 基础ViT
                import torchvision.models as models
                self.vit = models.vit_b_16(pretrained=False)

                # 多尺度特征融合
                self.multi_scale_conv = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((14, 14))
                )

                # 修改分类头
                in_features = self.vit.heads.head.in_features + 64 * 14 * 14
                self.vit.heads.head = nn.Linear(in_features, num_classes)

            def forward(self, pixel_values, labels=None):
                # 多尺度特征提取
                multi_scale_features = self.multi_scale_conv(pixel_values)
                multi_scale_features = multi_scale_features.flatten(1)

                # ViT特征
                vit_features = self.vit.conv_proj(pixel_values)
                vit_features = vit_features.flatten(2).transpose(1, 2)

                # 简单的transformer编码
                B, N, C = vit_features.shape
                cls_token = self.vit.class_token.expand(B, -1, -1)
                vit_features = torch.cat([cls_token, vit_features], dim=1)
                vit_features = vit_features + self.vit.encoder.pos_embedding

                # 简化版的transformer处理
                for block in self.vit.encoder.layers:
                    vit_features = block(vit_features)

                vit_features = self.vit.encoder.ln(vit_features)
                cls_features = vit_features[:, 0]

                # 特征融合
                fused_features = torch.cat([cls_features, multi_scale_features], dim=1)
                logits = self.vit.heads.head(fused_features)

                loss = None
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, labels)

                return type('Output', (), {'loss': loss, 'logits': logits})

        return MultiScaleViT(self.num_classes)

    def _create_residual_attention(self):
        """创建残差注意力ViT - 离线版"""

        class ResidualAttentionViT(nn.Module):
            def __init__(self, num_classes=27):
                super().__init__()

                import torchvision.models as models
                self.vit = models.vit_b_16(pretrained=False)

                # 残差注意力模块
                self.attention_enhancer = nn.MultiheadAttention(
                    embed_dim=768, num_heads=12, dropout=0.1
                )
                self.layer_norm = nn.LayerNorm(768)

                # 保持原有分类头
                in_features = self.vit.heads.head.in_features
                self.vit.heads.head = nn.Linear(in_features, num_classes)

            def forward(self, pixel_values, labels=None):
                # 基础ViT前向传播
                x = self.vit.conv_proj(pixel_values)
                x = x.flatten(2).transpose(1, 2)

                B, N, C = x.shape
                cls_token = self.vit.class_token.expand(B, -1, -1)
                x = torch.cat([cls_token, x], dim=1)
                x = x + self.vit.encoder.pos_embedding

                # 应用transformer层
                for block in self.vit.encoder.layers:
                    x = block(x)

                x = self.vit.encoder.ln(x)

                # 残差注意力增强
                cls_token_enhanced = x[:, 0:1, :]
                patch_tokens = x[:, 1:, :]

                enhanced_cls, _ = self.attention_enhancer(
                    cls_token_enhanced.transpose(0, 1),
                    patch_tokens.transpose(0, 1),
                    patch_tokens.transpose(0, 1)
                )
                enhanced_cls = enhanced_cls.transpose(0, 1)

                # 残差连接
                final_cls = self.layer_norm(cls_token_enhanced + enhanced_cls)

                logits = self.vit.heads.head(final_cls.squeeze(1))

                loss = None
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, labels)

                return type('Output', (), {'loss': loss, 'logits': logits})

        return ResidualAttentionViT(self.num_classes)

    def _create_channel_wise_attention(self):
        """创建通道注意力ViT - 离线版"""

        class ChannelWiseAttention(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.channel_attention = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 4, hidden_size),
                    nn.Sigmoid()
                )

            def forward(self, x):
                avg_pool = F.adaptive_avg_pool1d(x.transpose(1, 2), 1).view(x.size(0), -1)
                channel_weights = self.channel_attention(avg_pool).unsqueeze(1)
                return x * channel_weights

        class ChannelWiseViT(nn.Module):
            def __init__(self, num_classes=27):
                super().__init__()

                import torchvision.models as models
                self.vit = models.vit_b_16(pretrained=False)
                self.channel_attention = ChannelWiseAttention(768)

                in_features = self.vit.heads.head.in_features
                self.vit.heads.head = nn.Linear(in_features, num_classes)

            def forward(self, pixel_values, labels=None):
                # ViT前向传播
                x = self.vit.conv_proj(pixel_values)
                x = x.flatten(2).transpose(1, 2)

                B, N, C = x.shape
                cls_token = self.vit.class_token.expand(B, -1, -1)
                x = torch.cat([cls_token, x], dim=1)
                x = x + self.vit.encoder.pos_embedding

                # 应用通道注意力
                x = self.channel_attention(x)

                # Transformer层
                for block in self.vit.encoder.layers:
                    x = block(x)

                x = self.vit.encoder.ln(x)
                logits = self.vit.heads.head(x[:, 0])

                loss = None
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, labels)

                return type('Output', (), {'loss': loss, 'logits': logits})

        return ChannelWiseViT(self.num_classes)


# 使用示例
if __name__ == "__main__":
    print("使用离线模型改进分析...")

    # 创建离线模型改进器
    improver = OfflineViTModel(num_classes=27)

    models_to_analyze = {
        'baseline_vit': improver.create_baseline_model(),
        'multi_scale_vit': improver.create_attention_improvement('multi_scale'),
        'residual_attention_vit': improver.create_attention_improvement('residual'),
        'channel_wise_vit': improver.create_attention_improvement('channel_wise')
    }

    print("离线模型创建完成！")

    # 测试模型
    dummy_input = torch.randn(1, 3, 224, 224)
    for name, model in models_to_analyze.items():
        try:
            output = model(dummy_input)
            print(f"{name}: 前向传播成功，输出形状: {output.logits.shape}")
        except Exception as e:
            print(f"{name}: 错误 - {e}")