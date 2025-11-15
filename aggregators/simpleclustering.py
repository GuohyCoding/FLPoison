"""
简单聚类聚合器：通过聚类选择客户端多数簇的更新进行聚合。

该模块实现一种轻量级鲁棒策略：在梯度空间中对客户端更新进行无监督聚类，
选取成员数量最多的簇视为良性客户端集合，避免被少量异常更新主导。
"""
import numpy as np
from aggregators.aggregator_utils import prepare_grad_updates, wrapup_aggregated_grads
from aggregators.aggregatorbase import AggregatorBase
from aggregators import aggregator_registry
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth


@aggregator_registry
class SimpleClustering(AggregatorBase):
    """
    简单聚类聚合器：通过聚类选取良性客户端再执行平均聚合。
    """

    def __init__(self, args, **kwargs):
        """
        初始化聚类参数并继承基础聚合器配置。

        参数:
            args (argparse.Namespace | Any): 运行配置对象。
            **kwargs: 预留关键字参数，当前未使用。

        返回:
            None

        复杂度:
            时间复杂度 O(1)；空间复杂度 O(1)。
        """
        super().__init__(args)
        self.default_defense_params = {"clustering": "DBSCAN"}
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

    def aggregate(self, updates, **kwargs):
        """
        对客户端更新执行聚类，筛选成员最多的簇后进行聚合。

        参数:
            updates (numpy.ndarray | list[numpy.ndarray]): 客户端上传的模型更新或梯度。
            **kwargs: 需要包含
                - last_global_model (torch.nn.Module): 上一轮全局模型。

        返回:
            numpy.ndarray: 聚合后的向量，与当前算法语义一致。

        异常:
            KeyError: 缺少 'last_global_model' 时抛出。
            ValueError: 当配置的聚类算法不受支持时抛出。

        复杂度:
            时间复杂度受聚类算法影响，DBSCAN/MeanShift 典型为 O(n log n) 至 O(n^2)；
            空间复杂度 O(n * d)。
        """
        self.global_model = kwargs["last_global_model"]
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model
        )

        # 根据配置初始化聚类器，对客户端更新向量执行聚类。
        if self.clustering == "MeanShift":
            bandwidth = estimate_bandwidth(updates, quantile=0.5, n_samples=50)
            grad_cluster = MeanShift(
                bandwidth=bandwidth, bin_seeding=True, cluster_all=False
            )
        elif self.clustering == "DBSCAN":
            grad_cluster = DBSCAN(eps=0.05, min_samples=3)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.clustering}")

        grad_cluster.fit(updates)
        labels = grad_cluster.labels_
        n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
        if n_cluster <= 0:
            # 若聚类失败（全部标记为噪声），退化为普通平均。
            benign_idx = list(range(len(updates)))
        else:
            # 选择成员数量最多的簇作为良性客户端集合。
            benign_label = np.argmax([np.sum(labels == i) for i in range(n_cluster)])
            benign_idx = [int(idx) for idx in np.argwhere(labels == benign_label)]

        return wrapup_aggregated_grads(
            gradient_updates[benign_idx], self.args.algorithm, self.global_model
        )


# 费曼学习法解释（SimpleClustering.__init__）
# (A) 功能概述：设定默认聚类算法并继承基础配置。
# (B) 类比说明：像在开会前决定使用哪种分组方法，以便区分人员。
# (C) 步骤拆解：
#     1. 调用基类构造函数保存联邦配置。
#     2. 设置默认聚类算法为 DBSCAN，并允许外部覆盖。
#     3. 标记当前算法语义为 FedSGD。
# (D) 最小示例：
#     >>> class Args: defense_params=None
#     >>> sc = SimpleClustering(Args())
# (E) 边界与测试建议：缺少 defense_params 会报错；建议测试切换到 MeanShift 是否生效。
# (F) 参考：聚类算法综述、鲁棒聚合文献。


# 费曼学习法解释（SimpleClustering.aggregate）
# (A) 功能概述：对客户端更新执行聚类，选主流簇聚合以抑制异常值。
# (B) 类比说明：像把意见按相似度分组，取人数最多的一组作为集体意见。
# (C) 步骤拆解：
#     1. 将客户端更新转换为梯度向量。
#     2. 按配置选择聚类算法并拟合客户端更新。
#     3. 若聚类成功，选取成员最多的簇；否则退化为平均。
#     4. 对选定客户端的梯度执行聚合包装。
# (D) 最小示例：
#     >>> agg = sc.aggregate(updates, last_global_model=global_model)
# (E) 边界与测试建议：聚类参数需合理，避免所有点被视为噪声；建议测试不同数据分布下的鲁棒性。
# (F) 参考：DBSCAN、MeanShift 聚类算法介绍；鲁棒联邦学习研究。


__AI_ANNOTATION_SUMMARY__ = """
SimpleClustering.__init__: 设定默认聚类算法 DBSCAN 并继承基础配置。
SimpleClustering.aggregate: 对客户端更新执行聚类并选取最大簇进行聚合。
"""
