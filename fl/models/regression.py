# -*- coding: utf-8 -*-

import torch.nn as nn
from fl.models import model_registry


@model_registry
class lr(nn.Module):
    """逻辑回归模型：用于灰度图像的线性分类器。

    在联邦学习中经常作为轻量级基线，输入需展平成一维向量后接线性层输出 logits。
    """

    def __init__(self, input_dim=32*32, num_classes=10):
        """初始化逻辑回归模型并设置权重。

        参数:
            input_dim (int): 输入特征维度，默认 32×32 灰度图。
            num_classes (int): 分类类别数量。

        费曼学习法:
            (A) 构造线性层并对权重做正态初始化。
            (B) 类比把所有像素排成一行，再用一组权重计算各类得分。
        """
        super().__init__()
        self.input_dim, self.num_classes = input_dim, num_classes
        self.linear = nn.Linear(input_dim, num_classes)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

    def forward(self, xb):
        """前向传播：展平图像并计算分类 logits。

        参数:
            xb (Tensor): 输入张量，形状 `(batch, C, H, W)`，其中 `C*H*W = input_dim`。

        返回:
            Tensor: 模型输出的 logits，形状 `(batch, num_classes)`。

        费曼学习法:
            (A) 将图像摊平后乘以权重矩阵得到每类得分。
            (B) 类比把一张图片所有像素拼成一行，再分别与每个类别的“模板”做点积。
            (C) 步骤拆解:
                1. 使用 `reshape` 将图像展平为 `(batch, input_dim)`。
                2. 调用线性层得到 logits。
        """
        # flatten the image to vector for linear
        xb = xb.reshape(-1, self.input_dim)
        outputs = self.linear(xb)
        return outputs


# __AI_ANNOTATION_SUMMARY__
# 类 lr: 灰度图逻辑回归模型，线性层输出分类 logits。
