"""
Bucketing 聚合器：通过随机分桶与分层聚合抑制拜占庭客户端影响。

该实现基于 ICLR 2022《Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing》。
核心流程为先随机打乱客户端更新，再按固定桶大小分组，每个桶内求平均后交由选定的
基础聚合器（如 Krum、Mean）处理，从而提升鲁棒性。
"""
import math
import random
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from aggregators import aggregator_registry


@aggregator_registry
class Bucketing(AggregatorBase):
    """
    Bucketing 聚合器：先分桶再调用基础聚合器进行鲁棒融合。

    初始随机打乱客户端更新，按 `bucket_size` 切分为多个桶，
    对每个桶求平均后再交给选定聚合器进行最终整合。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 Bucketing 聚合器并实例化内部基础聚合器。

        参数:
            args (argparse.Namespace | Any): 运行配置对象，需包含 `defense_params`、`algorithm` 等字段。
            **kwargs: 预留的额外参数，当前未使用。

        返回:
            None

        异常:
            KeyError: 若 `selected_aggregator` 未注册对应实现。
            AttributeError: 当 `args` 缺失必要字段时触发。

        复杂度:
            时间复杂度 O(1)；空间复杂度 O(1)。
        """
        super().__init__(args)
        """
        bucket_size (int): 单个桶内包含的客户端数，常见取值 2/5/10。
        selected_aggregator (str): 桶级结果交由哪种聚合器处理（如 "Krum"、"Mean"）。
        """
        # 设置默认超参数，并支持通过 args.defense_params 覆写。
        self.default_defense_params = {
            "bucket_size": 2, "selected_aggregator": "Krum"}
        self.update_and_set_attr()
        # 根据配置从注册表获取基础聚合器类并实例化，用于桶级结果聚合。
        self.a_aggregator = aggregator_registry[self.selected_aggregator](
            args)
        # Bucketing 通常配合梯度上传算法使用，这里标记算法语义供参考。
        self.algorithm = "FedSGD"

    def aggregate(self, updates, **kwargs):
        """
        将客户端更新随机分桶，桶内求平均后调用基础聚合器输出最终结果。

        参数:
            updates (list | numpy.ndarray): 客户端的梯度/模型更新集合，长度等于客户端数量。
            **kwargs: 预留参数，当前未使用。

        返回:
            numpy.ndarray: 聚合后的更新向量，形状与基础聚合器输出一致。

        异常:
            TypeError: 当 `updates` 不支持 `random.shuffle` 或 `numpy.mean` 时可能抛出。

        复杂度:
            时间复杂度 O(n * d)，n 为客户端数量、d 为参数维度；空间复杂度 O(n * d)。
        """
        # 在原地打乱客户端更新顺序，降低恶意客户端集中于同一桶的概率。
        random.shuffle(updates)
        # 根据桶大小计算需要的桶数量，向上取整以覆盖所有客户端。
        num_buckets = math.ceil(
            len(updates) / self.bucket_size)
        # 按照 bucket_size 切片构造桶列表，最后一个桶可能人数不足。
        buckets = [updates[i:i + self.bucket_size]
                   for i in range(0, len(updates), self.bucket_size)]
        # 对每个桶求平均，得到桶级代表向量（也可替换为更鲁棒的统计方式）。
        bucket_avg_updates = np.array(
            [np.mean(buckets[bucket_id], axis=0) for bucket_id in range(num_buckets)])

        # 调用基础聚合器对桶级结果再次聚合，输出最终全局更新。
        return self.a_aggregator.aggregate(bucket_avg_updates)


# 费曼学习法解释（Bucketing.__init__）
# (A) 功能概述：设定 Bucketing 的默认参数并实例化桶内基础聚合器。
# (B) 类比说明：像先规划小组规模，再请一位资深导师负责整合各组意见。
# (C) 逐步拆解：
#     1. 调用父类构造函数，保存全局配置。
#     2. 设置默认桶大小与基础聚合器名称。
#     3. 调用 update_and_set_attr，让用户自定义参数覆盖默认值。
#     4. 从注册表取出指定基础聚合器并实例化。
#     5. 记录算法语义（FedSGD）以指示后续处理惯例。
# (D) 最小示例：
#     >>> class Args: defense_params = {"bucket_size": 3, "selected_aggregator": "Mean"}; algorithm="FedSGD"
#     >>> bucketing = Bucketing(Args())
#     >>> bucketing.bucket_size, bucketing.selected_aggregator
#     (3, "Mean")
# (E) 边界条件与测试建议：
#     - 若 `selected_aggregator` 未注册，将抛出 KeyError。
#     - 建议测试：1) 默认参数是否生效；2) 自定义参数是否正确覆盖。
# (F) 背景参考：
#     - 背景：Bucketing 专注抵御拜占庭客户端，依赖分桶思想。
#     - 推荐阅读：《Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing》《Distributed Algorithms》。


# 费曼学习法解释（Bucketing.aggregate）
# (A) 功能概述：随机分桶后对桶内更新求均值并交由基础聚合器整合。
# (B) 类比说明：像把多人意见随机分组，小组先商定意见，再由主持人汇总所有小组结论。
# (C) 逐步拆解：
#     1. 随机打乱更新，破坏恶意客户端可能的集中排列。
#     2. 按桶大小切分更新，形成多个小组。
#     3. 计算每个桶的平均更新，得到小组代表。
#     4. 将各小组代表输入基础聚合器，得到最终更新。
# (D) 最小示例：
#     >>> import numpy as np, random
#     >>> random.seed(0)
#     >>> updates = [np.array([i, i+1], dtype=float) for i in range(6)]
#     >>> result = bucketing.aggregate(updates.copy())
#     >>> result.shape
#     (2,)
# (E) 边界条件与测试建议：
#     - 若 `updates` 是不可原地打乱的结构（如元组），需先转换为列表。
#     - 建议测试：1) 不同随机种子结果是否落在合理范围；2) `bucket_size` ≥ 客户端数时是否退化为基础聚合器行为。
# (F) 背景参考：
#     - 背景：分桶结合鲁棒聚合是降低拜占庭攻击成功率的有效手段。
#     - 推荐阅读：《Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing》《Pattern Recognition and Machine Learning》。


__AI_ANNOTATION_SUMMARY__ = """
Bucketing.__init__: 初始化分桶策略参数并实例化桶级基础聚合器。
Bucketing.aggregate: 随机分桶求均值后调用基础聚合器完成最终客户端更新融合。
"""
