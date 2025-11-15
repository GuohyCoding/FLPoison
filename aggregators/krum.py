"""
Krum 聚合器：经典的拜占庭鲁棒梯度选择策略。

源自 NeurIPS 2017《Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent》。
算法通过计算每个客户端梯度与其他客户端梯度之间的欧氏距离，
选择与最多正常客户端最接近的那一个更新作为聚合结果，从而抑制异常值。
"""
from aggregators import aggregator_registry
from aggregators.aggregatorbase import AggregatorBase
from aggregators.aggregator_utils import L2_distances, krum_compute_scores


@aggregator_registry
class Krum(AggregatorBase):
    """
    Krum 聚合器：为拜占庭鲁棒场景挑选最可信的单个客户端更新。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 Krum 聚合器并设置拜占庭数量的合法性校验开关。

        参数:
            args (argparse.Namespace | Any): 联邦运行配置对象，应包含
                - num_adv (int): 恶意客户端数量估计。
                - defense_params (dict, optional): 用于覆盖默认防御参数。
            **kwargs: 预留关键字参数，当前未使用。

        返回:
            None

        异常:
            AttributeError: 当 args 未包含 defense_params 等字段时可能抛出。

        复杂度:
            时间复杂度 O(1); 空间复杂度 O(1)。
        """
        super().__init__(args)
        """
        enable_check (bool): 是否在聚合前检查 2f + 2 < n 条件，确保理论前提成立。
        """
        self.default_defense_params = {"enable_check": False}
        self.update_and_set_attr()

    def aggregate(self, updates, **kwargs):
        """
        执行 Krum 聚合：返回与多数客户端距离最近的单个更新。

        参数:
            updates (numpy.ndarray | list[numpy.ndarray]): 客户端上传的梯度或参数列表。
            **kwargs: 预留参数，当前未使用。

        返回:
            numpy.ndarray: 被选中客户端的更新向量。

        异常:
            ValueError: 当启用检查且 2f + 2 >= n 时抛出。

        复杂度:
            时间复杂度 O(n^2 * d)，空间复杂度 O(n^2)，n 为客户端数，d 为向量维度。
        """
        return krum(
            updates,
            self.args.num_adv,
            return_index=False,
            enable_check=self.enable_check,
        )


def krum(updates, num_byzantine=0, return_index=False, enable_check=False):
    """
    计算 Krum 聚合结果或其索引。

    参数:
        updates (Sequence[numpy.ndarray]): 客户端更新向量集合。
        num_byzantine (int): 允许的拜占庭客户端数量上界 f。
        return_index (bool): 若为 True，返回所选客户端索引；否则返回更新向量。
        enable_check (bool): 是否验证 2f + 2 < n 的约束。

    返回:
        numpy.ndarray | int: 聚合更新向量，或其对应的客户端索引。

    异常:
        ValueError: 当启用检查且 2f + 2 >= n 时抛出。

    复杂度:
        时间复杂度 O(n^2 * d)；空间复杂度 O(n^2)。
    """
    num_clients = len(updates)
    if enable_check:
        if 2 * num_byzantine + 2 >= num_clients:
            raise ValueError(
                f"num_byzantine should meet 2f+2 < n, got 2*{num_byzantine}+2 >= {num_clients}."
            )
    # 计算客户端两两之间的欧氏距离，形成对称矩阵。
    distances = L2_distances(updates)
    # 为每个客户端计算 KRUM 得分，即与最近 n-f-1 个邻居的距离和。
    scores = [
        (i, krum_compute_scores(distances, i, num_clients, num_byzantine))
        for i in range(num_clients)
    ]
    # 按得分升序排序，得分最小者最可信。
    sorted_scores = sorted(scores, key=lambda x: x[1])
    if return_index:
        return sorted_scores[0][0]
    else:
        return updates[sorted_scores[0][0]]


# 费曼学习法解释 (Krum.__init__)
# (A) 功能概述：设置是否在聚合前检查拜占庭数量约束，并继承通用配置。
# (B) 类比说明：像在开会前先确认参与者人数足够多，才能开展投票。
# (C) 步骤拆解：
#     1. 调用基类构造函数保存全局配置。
#     2. 设定默认的 `enable_check` 标志，允许外部覆盖。
#     3. 调用 `update_and_set_attr` 让配置生效。
# (D) 最小示例：
#     >>> class Args: num_adv=1; defense_params=None
#     >>> kr = Krum(Args())
# (E) 边界/测试建议：
#     - 未提供 defense_params 即尝试覆盖会触发 AttributeError。
# (F) 参考：Krum 原论文；拜占庭鲁棒聚合综述。


# 费曼学习法解释 (Krum.aggregate)
# (A) 功能概述：调用通用 krum 函数，从客户端更新中挑出最可信的一条。
# (B) 类比说明：像在多份报告中挑选与大多数观点最接近的那份作为正式结论。
# (C) 步骤拆解：
#     1. 接收所有客户端更新向量。
#     2. 将恶意客户端数量估计传入 `krum` 函数。
#     3. 返回 `krum` 选出的更新向量。
# (D) 最小示例：
#     >>> result = kr.aggregate(updates)
# (E) 边界/测试建议：
#     - 若 `enable_check` 为 True 时需保证 2f + 2 < n；可编写单元测试验证异常触发。
# (F) 参考：Krum 原论文；鲁棒聚合实验研究。


# 费曼学习法解释 (krum 函数)
# (A) 功能概述：计算每个客户端与最近邻的距离和，并选择得分最小的客户端。
# (B) 类比说明：像让每位同学找出最接近自己观点的几位同伴，观点差距总和最小的同学被视为最可靠。
# (C) 步骤拆解：
#     1. 若启用检查，验证客户端数量与拜占庭上限是否满足 2f + 2 < n。
#     2. 计算所有客户端之间的欧氏距离矩阵。
#     3. 对每个客户端求 n-f-1 个最近邻距离和作为得分。
#     4. 选取得分最小的客户端，返回其索引或更新向量。
# (D) 最小示例：
#     >>> idx = krum(updates, num_byzantine=1, return_index=True)
# (E) 边界/测试建议：
#     - 当 n <= 2f + 2 时算法无效，应及时抛异常；编写测试覆盖该情况。
#     - 建议测试：纯良性场景挑选最近邻正确；插入异常向量仍能避开。
# (F) 参考：Krum 原论文、分布式鲁棒优化教材。


__AI_ANNOTATION_SUMMARY__ = """
Krum.__init__: 初始化 Krum 聚合器并设置拜占庭数量合法性校验开关。
Krum.aggregate: 调用 krum 函数返回得分最小的客户端更新向量。
krum: 计算 KRUM 得分并选取得分最低的客户端或其索引。
"""
