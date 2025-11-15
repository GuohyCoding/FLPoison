# -*- coding: utf-8 -*-

"""ResNet 系列模型定义，移植自 PyTorch 官方实现并适配联邦学习场景。

参考论文：
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun,
        "Deep Residual Learning for Image Recognition." arXiv:1512.03385
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fl.models import model_registry


class BasicBlock(nn.Module):
    """ResNet 基础残差块，适用于 ResNet18/34 等浅层网络。"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """构造两层 3×3 卷积组成的残差单元，支持下采样。

        参数:
            in_planes (int): 输入通道数。
            planes (int): 输出通道数。
            stride (int): 第一层卷积的步幅，用于下采样。
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        """前向传播：残差分支 + Shortcut 相加后 ReLU。"""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet 瓶颈残差块，适用于 ResNet50/101/152 等深层网络。"""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        """构造“1×1→3×3→1×1”三层卷积的瓶颈结构。"""
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        """前向传播：瓶颈结构残差加和。"""
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class resnet(nn.Module):
    """通用 ResNet 构造器，参数化 block 类型与每层数量。"""

    def __init__(self, block, num_blocks, num_classes=10):
        """构建对应深度的 ResNet 网络。

        参数:
            block (nn.Module): 基础块类型（BasicBlock 或 Bottleneck）。
            num_blocks (List[int]): 每个 stage 的 block 数量。
            num_classes (int): 分类类别数。
        """
        super().__init__()
        self.in_planes = 64
        # 首层卷积采用 3×3 kernel、stride=1，适配 32×32 图像（如 CIFAR）。
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """堆叠指定数量的 residual block，首块可能下采样。"""
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """标准 ResNet 前向流程：conv→4 个 stage→平均池化→全连接。"""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# 注册常见 ResNet 架构
@model_registry
def resnet18(num_classes):
    """返回 ResNet-18 模型实例（BasicBlock，层配置 [2,2,2,2]）。"""
    return resnet(BasicBlock, [2, 2, 2, 2], num_classes)


@model_registry
def resnet34(num_classes):
    """返回 ResNet-34 模型实例（BasicBlock，层配置 [3,4,6,3]）。"""
    return resnet(BasicBlock, [3, 4, 6, 3], num_classes)


@model_registry
def resnet50(num_classes):
    """返回 ResNet-50 模型实例（Bottleneck，层配置 [3,4,6,3]）。"""
    return resnet(Bottleneck, [3, 4, 6, 3], num_classes)


@model_registry
def resnet101(num_classes):
    """返回 ResNet-101 模型实例（Bottleneck，层配置 [3,4,23,3]）。"""
    return resnet(Bottleneck, [3, 4, 23, 3], num_classes)


@model_registry
def resnet152(num_classes):
    """返回 ResNet-152 模型实例（Bottleneck，层配置 [3,8,36,3]）。"""
    return resnet(Bottleneck, [3, 8, 36, 3], num_classes)


def test():
    """简单测试函数：构建 resnet18 并以随机输入验证输出形状。"""
    net = resnet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# __AI_ANNOTATION_SUMMARY__
# 类 BasicBlock: ResNet 基础残差单元。
# 方法 forward (BasicBlock/Bottleneck/resnet): 定义残差块与网络整体前向路径。
# 函数 resnet18/resnet34/...: 返回指定深度的 ResNet 模型实例。
