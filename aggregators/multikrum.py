"""
Multi-Krum 聚合器：在 Krum 的基础上选取多个最可信客户端再求平均。

算法源自 NeurIPS 2017《Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent》。
核心做法是先计算每个客户端与其他客户端的欧氏距离并得到 KRUM 得分，选出得分最小的 m 个客户端，
再对其更新进行均值聚合，从而兼顾鲁棒性与统计效率。
"""
from aggregators.aggregatorbase import AggregatorBase
from aggregators.aggregator_utils import L2_distances, krum_compute_scores
import numpy as np
from aggregators import aggregator_registry


@aggregator_registry
class MultiKrum(AggregatorBase):
    """
    Multi-Krum 聚合器：挑选多个得分最低的客户端更新平均，提升鲁棒性与效率。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 Multi-Krum 聚合器并配置平均比例与合法性检查开关。

        参数:
            args (argparse.Namespace | Any): 运行配置对象，需包含
                - num_adv (int): 恶意客户端数量上界。
                - defense_params (dict, optional): 用于覆盖默认参数。
            **kwargs: 预留关键字参数，当前未使用。

        返回:
            None

        异常:
            AttributeError: 当 args 缺少 defense_params 等字段时可能抛出。

        复杂度:
            时间复杂度 O(1); 空间复杂度 O(1)。
        """
        super().__init__(args)
        """
        avg_percentage (float): 选入平均的客户端比例（0-1 之间）。
        enable_check (bool): 是否启用 2f+2 < n 条件校验。
        """
        self.default_defense_params = {
            "avg_percentage": 0.2, "enable_check": False}
        self.update_and_set_attr()

    def aggregate(self, updates, **kwargs):
        """
        计算 Multi-Krum 聚合结果。

        参数:
            updates (numpy.ndarray | list[numpy.ndarray]): 客户端更新集合。
            **kwargs: 预留参数，当前未使用。

        返回:
            numpy.ndarray: 聚合后的更新向量。

        异常:
            ValueError: 当启用检查且 2f + 2 >= n，或平均比例过大导致 m <= 0。

        复杂度:
            时间复杂度 O(n^2 * d); 空间复杂度 O(n^2)。
        """
        return multi_krum(
            updates,
            self.args.num_adv,
            avg_percentage=self.avg_percentage,
            enable_check=self.enable_check,
        )


def multi_krum(updates, num_byzantine, avg_percentage, enable_check=False):
    """
    Multi-Krum 聚合函数：选出得分最小的 m 个客户端并取平均。

    参数:
        updates (Sequence[numpy.ndarray]): 客户端更新向量集合。
        num_byzantine (int): 恶意客户端数量上界 f。
        avg_percentage (float): 选入平均的客户端比例，0 < avg_percentage ≤ 1。
        enable_check (bool): 是否验证 2f + 2 < n 的条件。

    返回:
        numpy.ndarray: 聚合后的更新向量。

    异常:
        ValueError: 当启用检查且条件不满足，或 m_avg < 1 时抛出。

    复杂度:
        时间复杂度 O(n^2 * d); 空间复杂度 O(n^2)。
    """
    num_clients = len(updates)
    m_avg = int(avg_percentage * num_clients)
    if m_avg < 1:
        raise ValueError(
            f"avg_percentage={avg_percentage} 过小，导致 m_avg < 1，无法执行 Multi-Krum."
        )

    if enable_check:
        if num_clients <= 2 * num_byzantine + 2:
            raise ValueError(
                f"num_byzantine 应满足 2f+2 < n，当前 2*{num_byzantine}+2 >= {num_clients}."
            )

    # 计算客户端之间的欧氏距离。
    distances = L2_distances(updates)

    # 计算每个客户端的 KRUM 得分，即最近 n-f-1 个邻居的距离和。
    scores = [
        (i, krum_compute_scores(distances, i, num_clients, num_byzantine))
        for i in range(num_clients)
    ]

    # 按得分升序排序，选出最小的 m_avg 个客户端索引。
    sorted_scores = sorted(scores, key=lambda x: x[1])
    selected_indices = [sorted_scores[idx][0] for idx in range(m_avg)]

    # 对选出的客户端更新取均值。
    return np.mean(updates[selected_indices], axis=0)


# 费曼学习法解释 (MultiKrum.__init__)
# (A) 功能概述：设置选取比例与合法性检查开关，为后续聚合做准备。
# (B) 类比说明：像在评审前先决定要挑选多少份最佳方案，并确认参赛者足够多。
# (C) 步骤：
#     1. 调用基类构造函数保存配置。
#     2. 设置默认的 avg_percentage 与 enable_check。
#     3. 调用 update_and_set_attr 让外部配置覆盖默认值。
# (D) 示例：
#     >>> class Args: num_adv=1; defense_params={'avg_percentage':0.5}
#     >>> mk = MultiKrum(Args())
# (E) 边界/测试建议：avg_percentage 需在 (0,1]；建议测试自定义配置是否生效。
# (F) 参考：Multi-Krum 原论文；拜占庭鲁棒聚合综述。


# 费曼学习法解释 (MultiKrum.aggregate)
# (A) 功能概述：从所有客户端更新中挑选多个得分最低的向量并平均。
# (B) 类比说明：像在众多报告中挑选几份最接近主流意见的，再合成最终总结。
# (C) 步骤：
#     1. 接收更新集合。
#     2. 调用 multi_krum 函数计算聚合结果。
#     3. 返回平均后的更新向量。
# (D) 示例：
#     >>> result = mk.aggregate(updates)
# (E) 边界/测试建议：若 enable_check=True，需保证 2f+2 < n；建议测试 avg_percentage 不同取值对结果的影响。
# (F) 参考：Multi-Krum 与 Krum 的对比研究。


# 费曼学习法解释 (multi_krum 函数)
# (A) 功能概述：计算 KRUM 得分并选出多个得分最低的客户端，再求均值。
# (B) 类比说明：像让每位成员找出最接近自己的若干同伴，然后选出最不孤立的几人组成委员会，再求他们意见的平均。
# (C) 步骤：
#     1. 根据 avg_percentage 确定需要平均的客户端数量 m_avg。
#     2. 如启用检查，验证 2f+2 < n 以确保鲁棒性条件。
#     3. 计算两两欧氏距离矩阵。
#     4. 为每个客户端求 KRUM 得分（最近邻距离和）。
#     5. 选取得分最小的 m_avg 个客户端索引。
#     6. 对这些客户端更新取平均，返回聚合向量。
# (D) 示例：
#     >>> agg = multi_krum(updates, num_byzantine=1, avg_percentage=0.5)
# (E) 边界/测试建议：
#     - avg_percentage 太小会导致 m_avg=0，应在测试中覆盖。
#     - 建议测试：纯良性数据是否选择接近期望的客户端；含恶意更新时是否避开异常值。
# (F) 参考：Multi-Krum 原论文；鲁棒统计方法教材。


__AI_ANNOTATION_SUMMARY__ = """
MultiKrum.__init__: 初始化聚合器并设置平均比例与合法性检查开关。
MultiKrum.aggregate: 调用 multi_krum 函数挑选得分最低的多个客户端并返回均值。
multi_krum: 计算 KRUM 得分、筛选前 m 个客户端并对其更新取平均。
"""
