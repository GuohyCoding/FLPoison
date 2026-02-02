"""
均值聚合器：计算客户端更新的简单平均值。

该模块实现联邦学习中最基本的聚合策略 —— 对所有客户端梯度或参数更新求算术平均，
也是 FedAvg 等算法的核心步骤之一。
"""
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
import torch
from aggregators import aggregator_registry


@aggregator_registry
class Mean(AggregatorBase):
    """
    均值聚合器：不带权重地对客户端更新求平均。
    """

    def __init__(self, args, **kwargs):
        """
        初始化均值聚合器。

        参数:
            args (argparse.Namespace | Any): 运行配置对象。
            **kwargs: 预留关键字参数，当前未使用。

        返回:
            None

        复杂度:
            时间复杂度 O(1)；空间复杂度 O(1)。
        """
        super().__init__(args)

    def aggregate(self, updates, **kwargs):
        """
        对客户端更新取平均并返回聚合结果。

        参数:
            updates (numpy.ndarray | list[numpy.ndarray]): 客户端更新集合，形状 (n, d)。
            **kwargs: 预留参数，当前未使用。

        返回:
            numpy.ndarray: 长度为 d 的均值向量。

        异常:
            ValueError: 当 updates 为空或维度不一致时 numpy 可能抛出异常。

        复杂度:
            时间复杂度 O(n * d)，空间复杂度 O(d)。
        """
        # 直接沿客户端维度取平均，得到全局更新。
        if torch.is_tensor(updates):
            return torch.mean(updates, dim=0)
        return np.mean(updates, axis=0)


# 费曼学习法解释 (Mean.aggregate)
# (A) 功能概述：把所有客户端的更新求平均，作为服务器端的新更新。
# (B) 类比说明：像收集多名学生的答案求平均分，作为班级的整体表现。
# (C) 步骤：
#     1. 接收所有客户端更新向量。
#     2. 使用 numpy.mean 在客户端维度上求均值。
#     3. 返回得到的平均向量供后续更新全局模型。
# (D) 最小示例：
#     >>> Mean(args).aggregate(np.array([[1, 2], [3, 4]]))
#     array([2., 3.])
# (E) 边界与测试建议：
#     - updates 为空或形状不一致会导致 numpy 报错；建议编写测试覆盖此情况。
#     - 测试建议：验证均值结果与手工计算一致。
# (F) 参考：FedAvg 原论文《Communication-Efficient Learning of Deep Networks from Decentralized Data》。


__AI_ANNOTATION_SUMMARY__ = """
Mean.aggregate: 对所有客户端更新向量执行算术平均，得到全局更新。
"""
