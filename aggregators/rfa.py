"""
鲁棒聚合 RFA：利用几何中位数降低联邦学习中恶意客户端的影响。

算法基于 RFA（Robust Federated Aggregation），使用平滑的 Weiszfeld 迭代在客户端梯度之间
近似求解几何中位数，比算术平均更能抵御极端值。
"""
import numpy as np
from aggregators.aggregatorbase import AggregatorBase
from aggregators import aggregator_registry


@aggregator_registry
class RFA(AggregatorBase):
    """
    RFA 聚合器：通过平滑 Weiszfeld 算法近似几何中位数实现鲁棒聚合。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 RFA 聚合器，配置 Weiszfeld 迭代次数与数值稳定项。

        参数:
            args (argparse.Namespace | Any): 联邦运行配置对象，应包含
                - defense_params (dict, optional): 用于覆盖默认参数。
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
        num_iters (int): Weiszfeld 迭代次数，用于控制几何中位数近似精度。
        epsilon (float): 分母平滑项，防止除零。
        """
        self.default_defense_params = {"num_iters": 3, "epsilon": 1.0e-6}
        self.update_and_set_attr()
        self.algorithm = "FedAvg"

    def aggregate(self, updates, **kwargs):
        """
        使用平滑 Weiszfeld 算法对客户端更新求几何中位数近似。

        参数:
            updates (numpy.ndarray | list[numpy.ndarray]): 客户端上传的向量集合。
            **kwargs: 预留参数，当前未使用。

        返回:
            numpy.ndarray: 通过几何中位数近似得到的鲁棒聚合向量。

        复杂度:
            时间复杂度 O(num_iters * n * d)；空间复杂度 O(d)。
        """
        alphas = np.ones(len(updates), dtype=np.float32) / len(updates)
        # 使用平滑 Weiszfeld 迭代生成几何中位数近似。
        return smoothed_weiszfeld(updates, alphas, self.epsilon, self.num_iters)


def smoothed_weiszfeld(updates, alphas, epsilon, num_iters):
    """
    平滑 Weiszfeld 算法：迭代近似求解几何中位数。

    参数:
        updates (numpy.ndarray): 客户端向量集合，形状 (n, d)。
        alphas (numpy.ndarray): 权重向量，默认均匀分布。
        epsilon (float): 防止除零的平滑项。
        num_iters (int): 迭代次数。

    返回:
        numpy.ndarray: 几何中位数近似向量。

    复杂度:
        时间复杂度 O(num_iters * n * d)；空间复杂度 O(d)。
    """
    v = np.zeros_like(updates[0], dtype=np.float32)
    for _ in range(num_iters):
        denom = np.linalg.norm(updates - v, ord=2, axis=1)
        betas = alphas / np.maximum(denom, epsilon)
        v = np.dot(betas, updates) / betas.sum()
    return v


# 费曼学习法解释（RFA.__init__）
# (A) 功能概述：指定 Weiszfeld 迭代次数与数值平滑项，为几何中位数计算做准备。
# (B) 类比说明：像在进行多轮投票前先确定迭代轮数和最低票差，确保流程稳定。
# (C) 步骤拆解：
#     1. 调用基类构造函数保存联邦配置。
#     2. 设置默认迭代次数与 epsilon，并允许外部覆盖。
#     3. 标记算法类型为 'FedAvg'。
# (D) 最小示例：
#     >>> class Args: defense_params=None
#     >>> rfa = RFA(Args())
# (E) 边界/测试建议：缺少 defense_params 会报错；建议测试自定义迭代次数是否生效。
# (F) 参考：《Robust Aggregation for Federated Learning》；几何中位数相关教材。


# 费曼学习法解释（RFA.aggregate）
# (A) 功能概述：利用平滑 Weiszfeld 算法对客户端更新求几何中位数近似。
# (B) 类比说明：像在多份意见中寻找一个离大家都不太远的折中方案。
# (C) 步骤拆解：
#     1. 为每个客户端设置均匀权重。
#     2. 调用 `smoothed_weiszfeld` 迭代近似几何中位数。
#     3. 返回几何中位数向量作为鲁棒聚合结果。
# (D) 最小示例：
#     >>> agg = rfa.aggregate(np.array([[0, 1], [1, 0], [10, 10]]))
# (E) 边界/测试建议：当数据极端分散时需确保迭代次数足够；建议比较与算术均值的差别。
# (F) 参考：Weiszfeld 算法；几何中位数的统计性质。


# 费曼学习法解释（smoothed_weiszfeld）
# (A) 功能概述：通过迭代加权平均找到离所有向量最近的几何中位数近似。
# (B) 类比说明：像不停调整站位，试图找到一个位置，使自己到各同伴的距离之和最小。
# (C) 步骤拆解：
#     1. 初始化几何中位数估计 v。
#     2. 计算当前估计与每个向量之间的距离。
#     3. 以距离倒数为权重更新 v（距离越近权重越大），并加入 epsilon 平滑。
#     4. 重复迭代直至达到指定轮数。
# (D) 最小示例：
#     >>> v = smoothed_weiszfeld(np.array([[0,0],[1,1],[2,2]]), np.ones(3)/3, 1e-6, 3)
# (E) 边界/测试建议：若所有向量相同，距离为零需依赖 epsilon；建议测试数值稳定性。
# (F) 参考：Weiszfeld 算法原始论文；几何中位数优化方法。


__AI_ANNOTATION_SUMMARY__ = """
RFA.__init__: 初始化 Weiszfeld 迭代参数以支撑几何中位数聚合。
RFA.aggregate: 调用平滑 Weiszfeld 算法以几何中位数近似聚合客户端更新。
smoothed_weiszfeld: 迭代加权平均求解几何中位数的平滑算法实现。
"""
