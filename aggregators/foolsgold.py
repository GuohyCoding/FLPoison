"""
FoolsGold 聚合器：通过历史更新相似性识别协同的恶意客户端。

本模块实现 RAID 2020《The Limitations of Federated Learning in Sybil Settings》中提出的 FoolsGold 算法。
核心思想是维护客户端梯度的历史累积向量，计算两两余弦相似度，
对相似度过高的客户端进行“赎罪”调整（pardoning），最终根据调整后的权重重新加权聚合。
"""
import copy
import numpy as np
import sklearn.metrics.pairwise as smp
from aggregators.aggregator_utils import prepare_grad_updates, wrapup_aggregated_grads
from aggregators.aggregatorbase import AggregatorBase
from aggregators import aggregator_registry
from fl.models.model_utils import ol_from_model


@aggregator_registry
class FoolsGold(AggregatorBase):
    """
    FoolsGold 聚合器：利用历史相似度抑制协同模型投毒。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 FoolsGold 聚合器并设置用于赎罪与特征筛选的超参数。

        参数:
            args (argparse.Namespace | Any): 运行配置对象，应包含
                - num_clients (int): 客户端数量。
                - num_classes (int): 数据分类数，用于输出层特征筛选。
                - defense_params (dict, optional): 用于覆盖默认参数。
            **kwargs: 预留关键字参数，当前未使用。

        返回:
            None

        异常:
            AttributeError: 当 args 缺失所需字段时抛出。

        复杂度:
            时间复杂度 O(1); 空间复杂度 O(1)。
        """
        super().__init__(args)
        """
        epsilon (float): 赎罪与软化权重时使用的数值稳定项。
        topk_ratio (float): 输出层特征选择比例，用于构建指示性掩码。
        """
        self.default_defense_params = {
            "epsilon": 1.0e-6, "topk_ratio": 0.1}
        self.update_and_set_attr()

        self.algorithm = "FedSGD"
        # 记录历史梯度（每一轮归一化后的客户端更新）。
        self.checkpoints = []

    def aggregate(self, updates, **kwargs):
        """
        基于 FoolsGold 策略重新加权客户端梯度并聚合。

        参数:
            updates (numpy.ndarray | list[numpy.ndarray]): 客户端上传的梯度或参数更新。
            **kwargs: 必须包含
                - last_global_model (torch.nn.Module): 上一轮全局模型，用于梯度向量化。

        返回:
            numpy.ndarray: 加权后的聚合梯度向量。

        异常:
            KeyError: 缺少 'last_global_model' 时抛出。

        复杂度:
            时间复杂度 O(n * d)，空间复杂度 O(n * d)，n 为客户端数、d 为参数维度。
        """
        self.global_model = kwargs["last_global_model"]
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model)

        feature_dim = len(gradient_updates[0])
        wv = np.zeros((self.args.num_clients, 1), dtype=np.float32)

        # 1. 归一化每个客户端梯度，避免因幅值差异导致历史累积偏差。
        for cid in range(self.args.num_clients):
            cid_norm = np.linalg.norm(gradient_updates[cid])
            if cid_norm > 1:
                gradient_updates[cid] /= cid_norm

        # 保存当前轮梯度（深拷贝）并累计成历史向量。
        self.checkpoints.append(copy.deepcopy(gradient_updates))
        sumed_updates = np.sum(self.checkpoints, axis=0)

        # 2. 基于上一轮全局模型的输出层权重提取指示性特征掩码。
        ol_last_global_model = ol_from_model(
            self.global_model, flatten=False, return_type='vector')
        indicative_mask = self.get_indicative_mask(
            ol_last_global_model, feature_dim)

        # 3. 根据指示性特征子空间，计算客户端历史累积向量的余弦相似度矩阵。
        cos_dist = smp.cosine_similarity(
            sumed_updates[:, indicative_mask == 1]) - np.eye(self.args.num_clients, dtype=np.float32)

        # 4. 执行赎罪机制，调整权重并聚合梯度。
        wv = self.pardoning(cos_dist)
        agg_grad_updates = np.dot(gradient_updates.T, wv)
        return wrapup_aggregated_grads(
            agg_grad_updates, self.args.algorithm, self.global_model, aggregated=True)

    def pardoning(self, cos_dist):
        """
        赎罪机制：缩放相似度矩阵并映射为 [0, 1] 权重。

        参数:
            cos_dist (numpy.ndarray): 客户端两两余弦相似度矩阵（自对角线置零）。

        返回:
            numpy.ndarray: 长度为 num_clients 的权重向量。

        复杂度:
            时间复杂度 O(n^2)，空间复杂度 O(n^2)。
        """
        max_cs = np.max(cos_dist, axis=1) + self.epsilon

        # 对每对客户端比较最大相似度，根据相对大小缩放相似度矩阵，惩罚高度相似的客户端。
        for i in range(self.args.num_clients):
            for j in range(self.args.num_clients):
                if i == j:
                    continue
                if max_cs[i] < max_cs[j]:
                    cos_dist[i][j] *= max_cs[i] / max_cs[j]

        # 将最大相似度转换为权重，越相似者权重越低。
        wv = 1 - np.max(cos_dist, axis=1)
        wv = np.clip(wv, 0, 1)
        wv /= np.max(wv) + self.epsilon
        wv[wv == 1] = 0.99

        # Logit 变换平滑权重，防止极端值。
        wv = np.log(wv / (1 - wv) + self.epsilon) + 0.5
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[wv < 0] = 0
        return wv

    def get_indicative_mask(self, ol_vec, feature_dim):
        """
        根据输出层参数的 Top-K 绝对值构建指示性特征掩码。

        参数:
            ol_vec (numpy.ndarray): 输出层权重矩阵，形状 (num_classes, num_features)。
            feature_dim (int): 全局梯度向量长度。

        返回:
            numpy.ndarray: 二值向量，1 表示被选为指示性特征。

        复杂度:
            时间复杂度 O(num_classes * num_features)，空间复杂度 O(feature_dim)。
        """
        class_dim, ol_feature_dim = ol_vec.shape[0], ol_vec.shape[1]
        ol_indicative_idx = np.zeros(
            (class_dim, ol_feature_dim), dtype=np.int64)
        topk = int(class_dim * self.topk_ratio)
        for i in range(class_dim):
            sig_features_idx = np.argpartition(ol_vec[i], -topk)[-topk:]
            ol_indicative_idx[i][sig_features_idx] = 1

        ol_indicative_idx = ol_indicative_idx.flatten()
        indicative_mask = np.pad(
            ol_indicative_idx,
            (feature_dim - len(ol_indicative_idx), 0),
            'constant'
        )
        return indicative_mask


# 费曼学习法解释 (FoolsGold.__init__)
# (A) 功能概述：设定赎罪用的数值稳定项与特征筛选比例，并准备历史梯度缓存。
# (B) 类比说明：像在开设银行账户前先设定风控阈值，并准备客户历史记录的档案夹。
# (C) 步骤拆解：
#     1. 保存联邦配置并继承基类属性。
#     2. 指定默认的 epsilon 与 topk_ratio，并允许覆盖。
#     3. 初始化历史梯度列表 `checkpoints`。
# (D) 最小示例：
#     >>> class Args: num_clients=5; num_classes=10; defense_params=None
#     >>> fg = FoolsGold(Args())
# (E) 边界条件与测试建议：
#     - 确保 defense_params 可覆盖默认值；若缺字段会抛 AttributeError。
# (F) 背景参考：FoolsGold 论文、鲁棒聚合与相似度度量教材。


# 费曼学习法解释 (FoolsGold.aggregate)
# (A) 功能概述：累积客户端梯度并按相似度对其加权，抑制协同攻击。
# (B) 类比说明：像记录每位学生历次答案的方向，凡是答案过于趋同的学生，其期末成绩会被降低权重。
# (C) 步骤拆解：
#     1. 将更新转换为梯度并控制每个梯度的范数。
#     2. 累积历史梯度形成长期行为特征。
#     3. 通过输出层 Top-K 特征构建掩码，专注于区分度高的坐标。
#     4. 计算历史累积的余弦相似度矩阵，去除对角线。
#     5. 调用赎罪函数将相似度转化为权重，再按权重对当前梯度加权平均。
# (D) 最小示例：
#     >>> agg = fg.aggregate(updates, last_global_model=global_model)
# (E) 边界条件与测试建议：
#     - 历史列表会持续增长，需关注内存开销；可针对长训练测试分页存储。
#     - 测试建议：对比纯良性与多 Sybil 场景下的权重分布变化。
# (F) 背景参考：协同攻击防御、余弦相似度的应用。


# 费曼学习法解释 (FoolsGold.pardoning)
# (A) 功能概述：根据最大相似度缩放余弦矩阵，将其映射成最终聚合权重。
# (B) 类比说明：像在评分时“赎罪”——若某学生的答案与他人过于相似，就削减他的得分。
# (C) 步骤拆解：
#     1. 计算每个客户端的最大相似度并加入 epsilon。
#     2. 两两比较，若某人相似度较低，其与相似度更高者的联系会被弱化。
#     3. 将最大相似度映射到 [0, 1] 权重，再经过 logit 平滑。
# (D) 最小示例：
#     >>> weights = fg.pardoning(cos_dist)
# (E) 边界条件与测试建议：
#     - 若 `wv` 全为 0 将导致后续除零，已通过 epsilon 缓解；仍建议在测试中覆盖极端场景。
# (F) 背景参考：FoolsGold 原论文对 pardoning 的解释。


# 费曼学习法解释 (FoolsGold.get_indicative_mask)
# (A) 功能概述：挑选输出层最具区分性的特征作为相似度计算的关注子空间。
# (B) 类比说明：像考试时只挑选最能区分学生的题目来比较成绩。
# (C) 步骤拆解：
#     1. 针对每个类别选取绝对值最大的 Top-K 特征。
#     2. 将二值化选择结果展平，并通过零填充与全局梯度维度对齐。
# (D) 最小示例：
#     >>> mask = fg.get_indicative_mask(ol_vec, feature_dim)
# (E) 边界条件与测试建议：
#     - 若 topk_ratio*class_dim < 1 可能导致选取为空，应在测试中验证并设置下限。
# (F) 背景参考：特征选择与输出层敏感特征分析。


__AI_ANNOTATION_SUMMARY__ = """
FoolsGold.__init__: 初始化赎罪参数与历史梯度缓存以支撑相似度防御。
FoolsGold.aggregate: 基于历史余弦相似度重新加权客户端梯度后聚合。
FoolsGold.pardoning: 调整余弦矩阵并映射为聚合权重抑制协同攻击。
FoolsGold.get_indicative_mask: 依输出层 Top-K 特征构建指示性掩码。
"""
