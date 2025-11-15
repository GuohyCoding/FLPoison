# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
from fl.models import model_registry


@model_registry
class lenet(nn.Module):
    """LeNet-5 卷积神经网络的经典实现，主要用于灰度图像分类。

    结构包括两层卷积 + Pooling 和三层全连接，适合 MNIST 等小型数据集。
    """

    def __init__(self, num_channels=1, num_classes=10):
        """构造 LeNet-5 网络结构。

        参数:
            num_channels (int): 输入图像的通道数，灰度图为 1。
            num_classes (int): 分类类别数量。

        费曼学习法:
            (A) 初始化所有卷积层与全连接层。
            (B) 类比搭积木：先摆好两层卷积积木，再接三层全连接积木。
            (C) 步骤拆解:
                1. 创建 conv1/conv2 用于提取特征。
                2. 创建 fc1/fc2/fc3 用于分类决策。
        """
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """前向传播：卷积 → ReLU → Pooling → 展平 → 全连接。

        参数:
            x (Tensor): 形状 `(batch, num_channels, 32, 32)` 的输入图像张量。

        返回:
            Tensor: 未使用 Softmax 的分类 logits，形状 `(batch, num_classes)`。

        费曼学习法:
            (A) 数据经过两次“特征提取 + 降采样”，再进入全连接做分类。
            (B) 类比先用放大镜观察细节，再逐步总结成文字描述。
            (C) 步骤拆解:
                1. conv1 → ReLU → MaxPool 提取初级特征并降采样。
                2. conv2 → ReLU → MaxPool 提取更高层特征。
                3. 展平后依次通过 fc1、fc2（ReLU）与 fc3 得到输出。
            (D) 示例:
                >>> logits = lenet(num_channels=1, num_classes=10)(x)
            (E) 边界条件与测试建议: 输入大小需为 32×32；建议添加单元测试检查输出维度。
        """
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


@model_registry
class lenet_bn(nn.Module):
    """带批归一化（BatchNorm）的 LeNet-5 变体，适配灰度/RGB 图像输入。"""

    @staticmethod
    def weight_init(m):
        """针对不同层类型设置权重初始化策略。

        参数:
            m (nn.Module): 当前层模块。

        费曼学习法:
            (A) 根据层类型选择合适的初始化方法。
            (B) 类比不同食材需要不同火候：线性层用 Xavier，卷积层用 Kaiming。
            (C) 步骤拆解:
                1. 如果是线性层 → Xavier 初始化，偏置置零。
                2. 如果是卷积层 → Kaiming 初始化。
                3. 如果是批归一化 → 权重置 1，偏置置 0。
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self, num_channels=1, num_classes=10):
        """构建带 BatchNorm 的 LeNet 网络。

        参数:
            num_channels (int): 输入通道数，可为 1 或 3。
            num_classes (int): 分类类别数。

        费曼学习法:
            (A) 搭建卷积 + BatchNorm + ReLU + Pooling 的顺序模块，并接全连接层。
            (B) 类比每层卷积后加“温度调节器”（BatchNorm）保持稳定。
        """
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=6,
                      kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.BatchNorm2d(120),
            nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
            nn.LogSoftmax(dim=-1))
        # 对网络进行一次性权重初始化。
        self.apply(self.weight_init)

    def forward(self, x):
        """前向传播：卷积模块 + 全连接模块。

        参数:
            x (Tensor): 输入张量，形状 `(batch, num_channels, 32, 32)`。

        返回:
            Tensor: LogSoftmax 输出，形状 `(batch, num_classes)`。

        费曼学习法:
            (A) 数据先走卷积/归一化层，再展平进入全连接输出 LogSoftmax。
            (B) 类比先通过一系列滤镜提取特征，再交由分类器给出置信度。
            (C) 步骤拆解:
                1. 输入通过 `convnet` 提取特征。
                2. 展平后送入 `fc` 得到分类结果。
        """
        out = self.convnet(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out


# __AI_ANNOTATION_SUMMARY__
# 类 lenet: 原版 LeNet-5 卷积网络结构。
# 方法 __init__ (lenet): 构建卷积与全连接层。
# 方法 forward (lenet): 执行卷积、池化与全连接的前向传播。
# 类 lenet_bn: 加入 BatchNorm 的 LeNet 变体。
# 方法 weight_init: 依据层类型执行权重初始化。
# 方法 __init__ (lenet_bn): 构建带 BatchNorm 的卷积与全连接模块。
# 方法 forward (lenet_bn): 通过顺序网络获取 LogSoftmax 输出。
