# -*- coding: utf-8 -*-

"""TabularMLP：用于表格型网络流量数据的多层感知机。

结构参考 5G-NIDD 论文中的 MLP baseline：
    输入层 → 全连接(256) → ReLU → Dropout → 全连接(128) → ReLU → Dropout → 输出层

适用于 5G-NIDD 等以数值特征向量为输入的联邦学习任务。
"""

import torch.nn as nn
from fl.models import model_registry


@model_registry
class mlp_tabular(nn.Module):
    """表格数据多层感知机，支持任意输入维度与类别数。

    参数:
        input_dim (int): 输入特征维度（例如 5G-NIDD 选取 10 个特征则为 10）。
        num_classes (int): 分类类别数量（5G-NIDD 多分类为 9，二分类为 2）。
        hidden_dims (list): 隐藏层维度列表，默认 [256, 128]。
        dropout (float): Dropout 比率，默认 0.3。
    """

    def __init__(self, input_dim: int = 10, num_classes: int = 9,
                 hidden_dims=None, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),   # BatchNorm1d 在 batch_size=1 时报错，LayerNorm 无此限制
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """前向传播。

        参数:
            x (Tensor): 形状 (batch, input_dim) 的特征张量。

        返回:
            Tensor: 分类 logits，形状 (batch, num_classes)。
        """
        return self.net(x)
