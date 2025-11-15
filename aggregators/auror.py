"""
Auror 聚合器实现：用于抵御协同深度学习中的投毒客户端。

该模块以双阶段聚类为核心，通过识别输出层指示性特征，筛选出被认为的良性客户端，
从而提高联邦学习在恶意客户端存在时的鲁棒性。
"""
from copy import deepcopy
import numpy as np
from sklearn.cluster import KMeans
from aggregators.aggregator_utils import prepare_grad_updates, prepare_updates, wrapup_aggregated_grads
from aggregators.aggregatorbase import AggregatorBase
from aggregators import aggregator_registry
import warnings
from fl.models.model_utils import ol_from_vector
with warnings.catch_warnings():
    warnings.simplefilter("ignore")


@aggregator_registry
class Auror(AggregatorBase):
    """
    Auror 聚合器（ACSAC 2016）通过两阶段聚类识别并排除恶意客户端。

    第一步在输出层参数维度上执行两类聚类，依据聚类中心距离选择“指示性特征”；
    第二步在指示性特征维度上再度聚类，以多数群体判定良性客户端用于聚合。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 Auror 聚合器并注入默认防御参数。

        参数:
            args (argparse.Namespace | Any): 运行配置对象，需包含 algorithm、num_clients、defense_params 等字段。
            **kwargs: 预留的附加关键字参数，当前未使用。

        返回:
            None

        异常:
            AttributeError: 若 args 缺少必要字段（如 defense_params、num_clients）时可能抛出。

        复杂度:
            时间复杂度 O(1)；空间复杂度 O(1)。
        """
        super().__init__(args)
        """
        indicative_threshold (float): Threshold for selecting indicative features based on cluster distance. A smaller value selects more features, increasing false positives. Suggested thresholds: 
        MNIST LeNet5 FedSGD lr=0.01: 1e-4; CIFAR10 ResNet18 FedSGD lr=0.01: 7e-4

        indicative_find_epoch (int): The first n epoch to find and determinate the indicative features, after that, the indicative features will be fixed
        """
        # 设置默认防御参数，并允许通过 args.defense_params 覆写。
        self.default_defense_params = {
            "indicative_threshold": 0.002, "indicative_find_epoch": 10}
        self.update_and_set_attr()
        # 记录当前全局训练轮次，用于限定指示性特征的搜索阶段。
        self.epoch_cnt = 0
        # store the indices of indicative features of self.indicative_find_epoch
        self.indicative_idx = []
        # 原论文默认配合梯度上传的 FedSGD 算法，此处标记算法语义。
        self.algorithm = "FedSGD"

    def aggregate(self, updates, **kwargs):
        """
        执行 Auror 聚合流程，基于两段聚类筛除可疑客户端后返回聚合结果。

        参数:
            updates (numpy.ndarray): 客户端上传的模型更新或梯度，形状约为 [客户端数, 参数维度]。
            **kwargs: 需要包含 'last_global_model'（torch.nn.Module）和 'global_epoch'（可选日志用）。

        返回:
            numpy.ndarray: 聚合后的梯度向量（FedSGD）或模型向量（FedAvg 等）。

        异常:
            KeyError: 若 kwargs 缺少 'last_global_model' 键。
            ValueError: 当指示性特征索引为空导致聚类输入不合法时。

        复杂度:
            时间复杂度约 O(E * n * d)，其中 E 为指示性特征搜索轮数、n 为客户端数、d 为输出层维度；
            空间复杂度 O(n * d)。
        """
        # 1. find the indicative features (indices in feature vector) via 2-clustering with center distance threshold
        # 记录当前全局模型，后续用于提取输出层参数与重建聚合结果。
        self.global_model = kwargs['last_global_model']
        # get model parameters updates and gradient updates according to the algorithm type
        # 将客户端更新统一转换为梯度视角，便于后续处理。
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model)

        # for the first 10 epoch, initialize and find the indicative_idx
        if self.epoch_cnt < self.indicative_find_epoch:
            self.indicative_idx = []
            # prepare the the (gradient) updates of output layers' parameter for each client
            # 仅聚焦输出层参数，因为这些特征对分类决策最敏感，便于区分恶意客户端。
            self.ol_updates = np.array([
                ol_from_vector(
                    gradient_updates[cid], self.global_model, flatten=True, return_type='vector')
                for cid in range(self.args.num_clients)
            ])
            feature_dim = len(self.ol_updates[0])
            for feature_idx in range(feature_dim):
                # 针对每个输出层维度独立执行二聚类，捕捉异常漂移。
                feature_arr = self.ol_updates[:, feature_idx]
                kmeans = KMeans(n_clusters=2, random_state=0).fit(
                    feature_arr.reshape(-1, 1))
                centers = kmeans.cluster_centers_
                # self.args.logger.info(
                #     f"Global epoch {kwargs['global_epoch']}, abs(centers[0] - centers[1]):{abs(centers[0] - centers[1])}")
                # 若两簇中心差距足够大，则认为该维度能指示恶意行为。
                if abs(centers[0] - centers[1]) >= self.indicative_threshold:
                    self.indicative_idx.append(feature_idx)
            # convert the indicative_idx of output layer to the whole model's indices
            # 将输出层局部索引映射回全模型向量的全局索引位置。
            self.indicative_idx = np.array(
                self.indicative_idx, dtype=np.int64) + len(gradient_updates[0]) - len(self.ol_updates[0])

        # 2. cluster the indicative features for anomaly detection
        # 在指示性特征子空间执行第二次聚类，识别主流客户端集合。
        indicative_updates = gradient_updates[:, self.indicative_idx]
        kmeans = KMeans(n_clusters=2, random_state=0).fit(indicative_updates)
        labels = kmeans.labels_
        # KMeans 不会产生 -1 标签，此处保持与其他聚类接口一致的防御式写法。
        labels = labels[labels != -1]
        # 选择标签数量较多的一簇，作为良性客户端代表。
        benign_label = 1 if np.sum(labels) > len(labels) / 2 else 0
        self.epoch_cnt += 1

        # 筛选良性客户端的梯度并送入统一封装函数，得到最终聚合结果。
        benign_grad_updates = gradient_updates[np.where(
            labels == benign_label)]
        return wrapup_aggregated_grads(benign_grad_updates, self.args.algorithm, self.global_model)


# 费曼学习法解释（Auror.__init__）
# (A) 功能概述：初始化 Auror 聚合器的默认参数，并做好运行状态的起始设置。
# (B) 类比说明：像给新装的雷达系统设定默认灵敏度，并把初始计数器归零，准备开始巡航。
# (C) 逐步拆解：
#     1. 调用基类构造函数保存运行配置——让父类完成通用初始化。
#     2. 定义指示性阈值和搜索轮数的默认值——保证即使用户未配置也能运行。
#     3. 调用 update_and_set_attr 将默认值与外部配置融合——确保自定义设置生效。
#     4. 将轮数计数器设为 0、指示性特征索引设为空列表——为后续逐轮搜索做准备。
#     5. 标记算法类型为 "FedSGD"——与原始设计保持一致，提醒后续步骤按梯度语义处理。
# (D) 最小示例：
#     >>> class Args:
#     ...     algorithm = "FedSGD"
#     ...     num_clients = 5
#     ...     defense_params = {"indicative_threshold": 0.001}
#     >>> auror = Auror(Args())
#     >>> auror.indicative_threshold
#     0.001
# (E) 边界条件与测试建议：
#     - Args 若缺少 defense_params 或 num_clients 将导致 AttributeError。
#     - 建议测试：1) 默认配置是否正确注入；2) 自定义阈值是否覆盖默认值。
# (F) 背景参考：
#     - 背景：Auror 源于联邦学习鲁棒聚合研究，强调输出层的重要性。
#     - 推荐阅读：《Auror: Defending against Poisoning Attacks in Collaborative Deep Learning Systems》《Federated Learning》。


# 费曼学习法解释（Auror.aggregate）
# (A) 功能概述：通过两阶段聚类识别良性客户端并聚合其梯度。
# (B) 类比说明：像先挑出最敏感传感器的异常信号，再把表现正常的传感器数据平均后汇报。
# (C) 逐步拆解：
#     1. 保存最新全局模型——后续需要提取输出层权重并返回聚合结果。
#     2. 将客户端更新统一转换为梯度矩阵——方便不同算法共用同一处理流程。
#     3. 若仍处于指示性特征发现阶段，针对输出层每个维度执行两类聚类——辨别哪些维度能区分恶意行为。
#     4. 将这些指示性维度映射回全模型向量索引——确保后续聚类关注正确的位置。
#     5. 在指示性特征子空间对客户端再执行一次 KMeans——把多数群体视为良性集合。
#     6. 筛选出良性客户端梯度，交给 wrapup_aggregated_grads 封装——得到符合算法语义的最终更新。
# (D) 最小示例（伪代码）：
#     >>> from aggregators.auror import Auror
#     >>> auror = Auror(Args())  # Args 同上，需包含 last_global_model 等
#     >>> updates = np.random.randn(Args.num_clients, param_dim)
#     >>> result = auror.aggregate(updates, last_global_model=global_model, global_epoch=0)
# (E) 边界条件与测试建议：
#     - 指示性特征若为空会导致后续聚类失败，应检查阈值设置；KMeans 对高维小样本可能不稳定。
#     - 建议测试：1) 在纯良性数据上是否返回均值；2) 某客户端明显异常时是否被排除。
# (F) 背景参考：
#     - 背景：Auror 利用输出层特征聚类检测恶意客户端，属于聚类型防御。
#     - 推荐阅读：《Auror: Defending against Poisoning Attacks in Collaborative Deep Learning Systems》《Pattern Recognition and Machine Learning》。


__AI_ANNOTATION_SUMMARY__ = """
Auror.__init__: 初始化 Auror 聚合器默认参数并建立状态变量以支撑指示性特征搜索。
Auror.aggregate: 对客户端梯度执行两阶段聚类筛选良性集合并返回聚合结果。
"""
