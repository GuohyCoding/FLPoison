"""
坐标中位数聚合器：对客户端更新逐坐标取中位数，抵御极端值干扰。

源自 ICML 2018《Byzantine-robust distributed learning: Towards optimal statistical rates》。
坐标中位数在每个参数维度上分别取客户端更新的中位数，与简单均值相比对异常值更鲁棒。
"""
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from aggregators import aggregator_registry


@aggregator_registry
class Median(AggregatorBase):
    """
    坐标中位数聚合器：挨个坐标取中位数以减少恶意噪声影响。
    """

    def __init__(self, args, **kwargs):
        """
        初始化坐标中位数聚合器。

        参数:
            args (argparse.Namespace | Any): 运行配置对象。
            **kwargs: 预留关键字参数，当前未使用。

        返回:
            None
        """
        super().__init__(args)

    def aggregate(self, updates, **kwargs):
        """
        对客户端更新逐坐标取中位数，返回聚合结果。

        参数:
            updates (numpy.ndarray | list[numpy.ndarray]): 客户端更新集合，形状 (n, d)。
            **kwargs: 预留参数，当前未使用。

        返回:
            numpy.ndarray: 长度为 d 的坐标中位数向量。

        异常:
            ValueError: 当 updates 为空或维度不一致时，numpy 可能抛出异常。

        复杂度:
            时间复杂度 O(n * d); 空间复杂度 O(1)（numpy 原地操作除外）。
        """
        # 利用 numpy 对每个坐标求中位数，相比均值更能抑制异常值。
        return np.median(updates, axis=0)


# 费曼学习法解释 (Median.aggregate)
# (A) 功能概述：对每个参数维度单独取中位数，得到鲁棒的聚合向量。
# (B) 类比说明：像统计多位同学在每道题的得分，并采用中位数避免极端离群值影响班级成绩。
# (C) 步骤：
#     1. 收集所有客户端更新的二维数组。
#     2. 沿客户端维度调用 `np.median`，对每个坐标求中位数。
#     3. 返回得到的中位数向量作为服务器更新。
# (D) 最小示例：
#     >>> Median(args).aggregate(np.array([[0, 10], [1, 11], [100, -5]]))
#     array([1., 10.])
# (E) 边界条件与测试建议：
#     - 输入为空时将引发错误；建议在测试中覆盖空列表或形状不一致情况。
#     - 测试建议：验证中位数对极端值的鲁棒性，例如加入非常大的异常梯度。
# (F) 参考：Yin et al. (ICML 2018) 关于坐标中位数的理论分析，《Robust Statistics》。


__AI_ANNOTATION_SUMMARY__ = """
Median.aggregate: 对客户端更新逐坐标取中位数，增强异常值鲁棒性。
"""
