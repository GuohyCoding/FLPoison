"""
裁剪均值聚合器：对客户端更新执行双向截尾后再平均，提升鲁棒性。

该方法来自 ICML 2018《Byzantine-robust distributed learning: Towards optimal statistical rates》，
通过在每个坐标维度上剔除最小与最大的一定比例（beta）客户端更新，再对剩余部分取平均，
以降低极端值对聚合结果的影响。
"""
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
import torch
from aggregators import aggregator_registry


@aggregator_registry
class TrimmedMean(AggregatorBase):
    """
    裁剪均值聚合器：对每个坐标剔除双侧极值后求平均。
    """

    def __init__(self, args, **kwargs):
        """
        初始化裁剪均值聚合器并设定剪除比例。

        参数:
            args (argparse.Namespace | Any): 运行配置对象，应包含 defense_params。
            **kwargs: 预留关键字参数，当前未使用。

        返回:
            None

        异常:
            AttributeError: 当 args 缺少 defense_params 属性时可能抛出。

        复杂度:
            时间复杂度 O(1)；空间复杂度 O(1)。
        """
        super().__init__(args)
        """
        beta (float): 双侧裁剪比例，取值范围 [0, 0.5)，表示剔除的极值比例。
        """
        self.default_defense_params = {"beta": 0.1}
        self.update_and_set_attr()

    def aggregate(self, updates, **kwargs):
        """
        对客户端更新执行裁剪均值聚合。

        参数:
            updates (numpy.ndarray | list[numpy.ndarray]): 客户端上传的更新集合。
            **kwargs: 预留关键字参数，当前未使用。

        返回:
            numpy.ndarray: 裁剪并平均后的聚合向量。

        异常:
            ValueError: 当 beta 导致剔除数量超过客户端数时抛出。

        复杂度:
            时间复杂度 O(n * d log n)、空间复杂度 O(n * d)。
        """
        return trimmed_mean(updates, self.beta)


def trimmed_mean(updates, filter_frac):
    """
    裁剪均值函数：在每个坐标上剔除指定比例的最大与最小值后求平均。

    参数:
        updates (numpy.ndarray | list[numpy.ndarray]): 客户端更新集合，形状 (n, d)。
        filter_frac (float): 剔除比例，0 <= filter_frac < 0.5。

    返回:
        numpy.ndarray: 裁剪均值向量。

    异常:
        ValueError: 当 filter_frac * n >= 0.5 * n 时无法执行裁剪均值。

    复杂度:
        时间复杂度 O(n * d log n)（np.partition 均摊为 O(n)），空间复杂度 O(n * d)。
    """
    num_clients = len(updates)
    num_excluded = int(filter_frac * num_clients)
    if num_excluded * 2 >= num_clients:
        raise ValueError(
            f"filter_frac={filter_frac} 剔除过多客户端，需满足 2 * floor(beta * n) < n。"
        )

    if torch.is_tensor(updates):
        sorted_vals, _ = torch.sort(updates, dim=0)
        trimmed = sorted_vals[num_excluded: num_clients - num_excluded]
        return torch.mean(trimmed, dim=0)

    # 使用 np.partition 找到每个坐标的前 num_excluded 个最小值与最大值。
    smallest_excluded = np.partition(
        updates, kth=num_excluded, axis=0
    )[:num_excluded]
    biggest_excluded = np.partition(
        updates, kth=num_clients - num_excluded, axis=0
    )[num_clients - num_excluded :]

    # 通过加权抵消被剔除值，实现剩余元素的均值。
    weights = np.concatenate(
        (updates, -smallest_excluded, -biggest_excluded)
    ).sum(axis=0)
    weights /= num_clients - 2 * num_excluded
    return weights


# 费曼学习法解释（TrimmedMean.__init__）
# (A) 功能概述：设定裁剪比例 beta，准备执行裁剪均值策略。
# (B) 类比说明：像在评分前先决定要去掉每个人的最高分和最低分。
# (C) 步骤拆解：
#     1. 调用基类构造函数保存配置。
#     2. 设置默认 beta，并允许外部覆盖。
# (D) 最小示例：
#     >>> class Args: defense_params=None
#     >>> tm = TrimmedMean(Args())
# (E) 边界条件与测试建议：缺少 defense_params 会报错；建议测试自定义 beta 是否生效。
# (F) 参考：Trimmed Mean 聚合在鲁棒统计中的应用。


# 费曼学习法解释（TrimmedMean.aggregate）
# (A) 功能概述：调用裁剪均值函数对客户端更新执行鲁棒聚合。
# (B) 类比说明：像把每位评委的分数去掉极端值后求平均。
# (C) 步骤拆解：
#     1. 接收所有客户端更新。
#     2. 调用 `trimmed_mean` 进行裁剪均值计算。
#     3. 返回聚合结果。
# (D) 最小示例：
#     >>> agg = tm.aggregate(updates)
# (E) 边界条件与测试建议：beta 过大时需抛出异常；建议测试不同 beta 对鲁棒性的影响。
# (F) 参考：Trimmed Mean 相关研究。


# 费曼学习法解释（trimmed_mean）
# (A) 功能概述：在每个参数维度上剔除极值，平均剩余更新。
# (B) 类比说明：像对每道题的得分剔掉最高和最低若干份，再算班级平均。
# (C) 步骤拆解：
#     1. 根据 beta 计算需剔除的客户端数量。
#     2. 使用 `np.partition` 快速找到最小和最大元素。
#     3. 对剩余元素求平均。
# (D) 最小示例：
#     >>> trimmed_mean(np.array([[0,1],[2,3],[100,100]]), 1/3)
#     array([1., 2.])
# (E) 边界条件与测试建议：beta=0 时退化为普通均值；beta 过大时无剩余样本需抛错；建议测试极端值情况下的鲁棒性。
# (F) 参考：Trimmed Mean 在鲁棒统计与联邦学习中的应用。


__AI_ANNOTATION_SUMMARY__ = """
TrimmedMean.__init__: 初始化裁剪比例 beta 以支持裁剪均值聚合。
TrimmedMean.aggregate: 调用裁剪均值函数对客户端更新执行鲁棒聚合。
trimmed_mean: 在每个坐标剔除双侧极值后求平均，输出裁剪均值向量。
"""
