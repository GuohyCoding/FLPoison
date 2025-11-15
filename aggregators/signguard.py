"""
SignGuard 聚合器：结合范数门限与符号统计聚类过滤恶意客户端。

参考 ICDCS 2022《Byzantine-robust Federated Learning through Collaborative Malicious Gradient Filtering》，
核心流程包括：
1. 基于客户端梯度范数的中位数设定上下阈值，初步筛除异常客户端；
2. 在随机抽取的坐标子集上统计梯度符号分布，通过聚类找出主体簇；
3. 对被判定为良性的梯度按照中位数范数进行裁剪并聚合。
"""
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from aggregators import aggregator_registry
from aggregators.aggregator_utils import prepare_grad_updates, wrapup_aggregated_grads
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
import random


@aggregator_registry
class SignGuard(AggregatorBase):
    """
    SignGuard 聚合器：利用范数过滤与符号聚类联合筛除恶意客户端。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 SignGuard 聚合器并设置过滤与聚类相关超参数。

        参数:
            args (argparse.Namespace | Any): 联邦运行配置对象，应包含
                - num_clients (int): 客户端数量。
                - defense_params (dict, optional): 用户自定义防御参数。
            **kwargs: 预留关键字参数，当前实现未使用。

        返回:
            None

        异常:
            AttributeError: 当 args 缺少 defense_params 属性时可能抛出。

        复杂度:
            时间复杂度 O(1)；空间复杂度 O(1)。
        """
        super().__init__(args)
        """
        lower_bound (float): 范数下限相对系数，低于该值视为异常。
        upper_bound (float): 范数上限相对系数，高于该值视为异常。
        selection_fraction (float): 随机抽取的坐标比例，用于符号统计。
        clustering (str): 使用的聚类算法（MeanShift、DBSCAN、KMeans）。
        random_seed (int): 聚类随机种子。
        """
        self.default_defense_params = {
            "lower_bound": 0.1,
            "upper_bound": 3.0,
            "selection_fraction": 0.1,
            "clustering": "DBSCAN",
            "random_seed": 0,
        }
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

    def aggregate(self, updates, **kwargs):
        """
        执行 SignGuard 聚合：范数过滤 + 符号聚类 + 裁剪聚合。

        参数:
            updates (numpy.ndarray | list[numpy.ndarray]): 客户端上传的更新集合。
            **kwargs: 需要包含
                - last_global_model (torch.nn.Module): 上一轮全局模型。

        返回:
            numpy.ndarray: 聚合后的梯度向量。

        异常:
            KeyError: 缺少 'last_global_model' 时抛出。

        复杂度:
            时间复杂度近似 O(n * d)，若使用聚类算法可能增加额外开销；
            空间复杂度 O(n * d)。
        """
        self.global_model = kwargs["last_global_model"]
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model
        )

        # 1. 基于范数的上下阈值过滤客户端。
        S1_benign_idx, median_norm, client_norms = self.norm_filtering(
            gradient_updates
        )

        # 2. 基于符号统计的聚类过滤客户端。
        S2_benign_idx = self.sign_clustering(gradient_updates)

        # 取两种过滤手段的交集；若交集为空，可视为没有合格客户端。
        benign_idx = list(set(S1_benign_idx).intersection(S2_benign_idx))

        if len(benign_idx) == 0:
            # 若无交集，为避免返回空集合，这里可退回范数过滤结果。
            benign_idx = S1_benign_idx

        # 3. 对最终良性集合进行范数裁剪并聚合。
        grads_clipped_norm = np.clip(
            client_norms[benign_idx], a_min=0, a_max=median_norm
        )
        benign_clipped = (
            gradient_updates[benign_idx]
            / client_norms[benign_idx].reshape(-1, 1)
        ) * grads_clipped_norm.reshape(-1, 1)

        return wrapup_aggregated_grads(
            benign_clipped, self.args.algorithm, self.global_model
        )

    def norm_filtering(self, gradient_updates):
        """
        基于梯度范数进行初步过滤。

        参数:
            gradient_updates (numpy.ndarray): 客户端梯度矩阵，形状 (n, d)。

        返回:
            tuple[list[int], float, numpy.ndarray]:
                - 通过过滤的客户端索引列表；
                - 梯度范数的中位数；
                - 所有客户端的梯度范数。

        复杂度:
            时间复杂度 O(n * d)；空间复杂度 O(n)。
        """
        client_norms = np.linalg.norm(gradient_updates, axis=1)
        median_norm = np.median(client_norms)
        benign_mask = (client_norms > self.lower_bound * median_norm) & (
            client_norms < self.upper_bound * median_norm
        )
        benign_idx = np.argwhere(benign_mask)
        return benign_idx.reshape(-1).tolist(), median_norm, client_norms

    def sign_clustering(self, gradient_updates):
        """
        在随机坐标子集上统计符号特征并执行聚类过滤。

        参数:
            gradient_updates (numpy.ndarray): 客户端梯度矩阵，形状 (n, d)。

        返回:
            list[int]: 被视为良性客户端的索引列表。

        复杂度:
            时间复杂度取决于聚类算法，通常在 O(n * d_selected) 至 O(n^2)。
        """
        num_para = gradient_updates.shape[1]
        num_selected = max(1, int(self.selection_fraction * num_para))
        # 采用滑窗方式随机选取连续坐标区间，提高实现简单性。
        start_idx = random.randint(0, max(0, num_para - num_selected))
        randomized_weights = gradient_updates[:, start_idx : start_idx + num_selected]

        # 统计正、零、负符号比例，用于刻画方向信息。
        sign_grads = np.sign(randomized_weights)
        sign_features = np.empty((self.args.num_clients, 3), dtype=np.float32)

        def sign_feat(target):
            sign_count = (sign_grads == target).sum(axis=1, dtype=np.float32)
            sign_ratio = sign_count / num_selected
            return sign_ratio / (sign_ratio.max() + 1e-8)

        sign_features[:, 0] = sign_feat(1)   # 正号比例
        sign_features[:, 1] = sign_feat(0)   # 零值比例
        sign_features[:, 2] = sign_feat(-1)  # 负号比例

        # 根据配置选择聚类算法。
        if self.clustering == "MeanShift":
            bandwidth = estimate_bandwidth(sign_features, quantile=0.5, n_samples=50)
            sign_cluster = MeanShift(
                bandwidth=bandwidth, bin_seeding=True, cluster_all=False
            )
        elif self.clustering == "DBSCAN":
            sign_cluster = DBSCAN(eps=0.05, min_samples=3)
        elif self.clustering == "KMeans":
            sign_cluster = KMeans(n_clusters=2, random_state=self.random_seed)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.clustering}")

        sign_cluster.fit(sign_features)
        labels = sign_cluster.labels_
        n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
        if n_cluster <= 0:
            # 若聚类失败（全部为 -1），退化为保留所有客户端。
            return list(range(self.args.num_clients))

        # 选择含成员最多的簇作为良性客户端集合。
        benign_label = np.argmax([np.sum(labels == i) for i in range(n_cluster)])
        benign_idx = [int(idx) for idx in np.argwhere(labels == benign_label)]
        return benign_idx


# 费曼学习法解释（SignGuard.__init__）
# (A) 功能概述：设置范数门限、符号抽样比例与聚类方式，准备执行过滤策略。
# (B) 类比说明：像在比赛前先设定警戒值和抽查比例，并指定评委如何分组评估。
# (C) 步骤拆解：
#     1. 保存全局配置，继承基类属性。
#     2. 定义默认的 lower_bound、upper_bound、selection_fraction 等参数。
#     3. 允许外部通过 defense_params 覆盖默认值。
# (D) 最小示例：
#     >>> class Args: num_clients=10; defense_params=None
#     >>> sg = SignGuard(Args())
# (E) 边界条件与测试建议：若 defense_params 缺失会报错；建议测试不同聚类算法设置是否生效。
# (F) 参考：SignGuard 原论文；鲁棒聚合综述。


# 费曼学习法解释（SignGuard.aggregate）
# (A) 功能概述：结合范数过滤与符号聚类筛除异常客户端并裁剪聚合。
# (B) 类比说明：像先按音量筛掉过大或过小的演讲者，再根据语调相似度分组，最后选出主流声音求平均。
# (C) 步骤拆解：
#     1. 将客户端更新统一转换为梯度表示。
#     2. 使用 `norm_filtering` 基于范数上下阈值筛选客户端。
#     3. 使用 `sign_clustering` 基于符号特征聚类过滤。
#     4. 取交集后对保留梯度进行范数裁剪并求平均。
#     5. 返回聚合结果。
# (D) 最小示例：
#     >>> agg = sg.aggregate(updates, last_global_model=global_model)
# (E) 边界条件与测试建议：交集为空时需决定 fallback 策略；建议测试不同聚类算法与随机抽样的鲁棒性。
# (F) 参考：SignGuard 原论文；聚类算法（MeanShift、DBSCAN、KMeans）。


# 费曼学习法解释（SignGuard.norm_filtering）
# (A) 功能概述：根据客户端梯度范数的中位数设定上下阈值筛选候选客户端。
# (B) 类比说明：像以班级平均身高为基准，剔除明显高于或低于平均的异常值。
# (C) 步骤拆解：
#     1. 计算所有客户端梯度范数。
#     2. 求取范数中位数作为尺度参考。
#     3. 保留范数在 [lower_bound, upper_bound] 区间内的客户端。
# (D) 最小示例：
#     >>> idx, median_norm, norms = sg.norm_filtering(gradient_updates)
# (E) 边界条件与测试建议：lower_bound 或 upper_bound 设置不当会过滤过多或过少；建议调参与测试。
# (F) 参考：鲁棒统计与门限过滤方法。


# 费曼学习法解释（SignGuard.sign_clustering）
# (A) 功能概述：在随机坐标子集上统计梯度符号分布，通过聚类识别主群体。
# (B) 类比说明：像抽取部分试题，统计每位学生做对、做错、未作答的比例，再用聚类找出水平相近的学生。
# (C) 步骤拆解：
#     1. 随机选取部分坐标，形成子空间。
#     2. 统计各客户端在子空间上正、零、负符号的比例并归一化。
#     3. 按指定聚类算法对符号特征聚类。
#     4. 选择成员最多的簇作为良性客户端集合。
# (D) 最小示例：
#     >>> benign_idx = sg.sign_clustering(gradient_updates)
# (E) 边界条件与测试建议：聚类参数需合理设置；聚类失败时应有退化策略；建议针对不同算法分别测试。
# (F) 参考：聚类算法教材，符号统计在鲁棒检测中的应用。


__AI_ANNOTATION_SUMMARY__ = """
SignGuard.__init__: 配置范数门限、符号抽样比例与聚类算法。
SignGuard.aggregate: 结合范数过滤与符号聚类筛除异常客户端并裁剪聚合。
SignGuard.norm_filtering: 根据范数中位数设定上下阈值筛选候选客户端。
SignGuard.sign_clustering: 在随机坐标子集上统计符号特征并聚类筛选良性客户端。
"""
