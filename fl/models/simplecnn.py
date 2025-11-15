# -*- coding: utf-8 -*-

"""SimpleCNN：联邦学习常见的轻量级卷积网络。

该模型在 FLTrust、FLDetector、FangAttack 等研究中常被用作基线，
包含两层卷积 + 池化以及两层全连接，适配 28×28 或 32×32 图像。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fl.models import model_registry

# 输入尺寸与模型配置映射：height -> [conv1_out, conv2_out, fc_hidden]
simple_config = {
    "28": [30, 50, 100],  # MNIST, FashionMNIST, FEMNIST
    "32": [32, 64, 512]   # CIFAR-10, CINIC-10
}


@model_registry
class simplecnn(nn.Module):
    """轻量卷积神经网络，支持灰度或彩色小尺寸图像分类。

    结构包含两层卷积与池化，随后接两层全连接，常用于联邦学习中的低成本基线。
    """

    def __init__(self, input_size=(3, 32, 32), num_classes=10):
        """根据输入尺寸选择配置，构建卷积与全连接层。

        参数:
            input_size (Tuple[int, int, int]): `(channels, height, width)`。
            num_classes (int): 分类类别数量。

        费曼学习法:
            (A) 启动时依据图像尺寸决定每层通道数，并建立卷积/池化/全连接结构。
            (B) 类比厨师根据锅大小调整配方，准备两道调味后的菜（卷积）和两道甜点（全连接）。
            (C) 步骤拆解:
                1. 从 `simple_config` 中读取适合输入尺寸的通道设置。
                2. 定义两组 `Conv2d+MaxPool` 提取特征。
                3. 调用 `_get_flattened_size` 推算展平向量长度。
                4. 构建两层全连接用于分类。
            (D) 示例:
                >>> model = simplecnn(input_size=(1,28,28), num_classes=10)
            (E) 边界条件与测试建议: 需确保 `input_size[1]` 为 28 或 32；建议编写 forward 单元测试确认输出维度正确。
        """
        super().__init__()
        self.model_config = simple_config[f"{input_size[1]}"]
        # 第一层卷积 + ReLU + 池化
        self.conv1 = nn.Conv2d(in_channels=input_size[0], out_channels=self.model_config[0], kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二层卷积 + ReLU + 池化
        self.conv2 = nn.Conv2d(in_channels=self.model_config[0], out_channels=self.model_config[1], kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 计算展平后的向量长度
        self.flattened_size = self._get_flattened_size(input_size)
        # 全连接层
        self.fc1 = nn.Linear(self.flattened_size, self.model_config[2])
        self.fc2 = nn.Linear(self.model_config[2], num_classes)

    def _get_flattened_size(self, input_size):
        """根据输入尺寸推算卷积+池化后特征向量长度。

        费曼学习法:
            (A) 按卷积与池化顺序更新高宽，最终乘上通道数得到展平长度。
            (B) 类比不断折叠纸张：每折一次尺寸减半、边缘缩小。
            (C) 步骤拆解:
                1. 应用第一层卷积的核大小与步幅公式。
                2. 应用第一次池化的尺寸变化。
                3. 重复上一步骤处理第二层卷积与池化。
                4. 返回 `(通道数 × 高 × 宽)`.
        """
        height, width = input_size[1], input_size[2]
        # conv1
        height = (height - 3) // 1 + 1
        width = (width - 3) // 1 + 1
        # pool1
        height = (height - 2) // 2 + 1
        width = (width - 2) // 2 + 1
        # conv2
        height = (height - 3) // 1 + 1
        width = (width - 3) // 1 + 1
        # pool2
        height = (height - 2) // 2 + 1
        width = (width - 2) // 2 + 1
        return self.model_config[1] * height * width

    def forward(self, x):
        """前向传播：卷积 → 池化 → 展平 → 全连接，输出 logits。

        参数:
            x (Tensor): 输入图像，形状 `(batch, channels, height, width)`。

        返回:
            Tensor: 分类 logits，形状 `(batch, num_classes)`。

        费曼学习法:
            (A) 逐层提取特征并映射到类别得分。
            (B) 类比拍照后用滤镜增强，再将结果交给裁判打分。
            (C) 步骤拆解:
                1. 依次通过 `conv1`、`pool1`、`conv2`、`pool2` 提取特征。
                2. 展平张量，便于进入全连接层。
                3. 经 `fc1` 激活后接 `fc2` 输出 logits。
            (D) 示例:
                >>> logits = model(torch.randn(4, 3, 32, 32))
            (E) 边界条件与测试建议: 输入尺寸需与 `input_size` 匹配；测试两种配置（28/32）确保流程正确。
        """
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# __AI_ANNOTATION_SUMMARY__
# 类 simplecnn: 基础卷积网络，构造与前向流程。
