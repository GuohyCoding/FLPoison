"""
FLTrust 聚合器：通过可信引导数据集计算信任分数的鲁棒联邦聚合。

实现基于 NDSS 2021《FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping》。
服务器预先持有一份小型可靠数据集，训练得到“可信锚”模型；随后以余弦相似度衡量客户端梯度与锚梯度的对齐程度，
将相似度作为信任权重调整客户端更新的贡献，同时归一化其幅度以避免恶意放大。
"""
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_similarity
from aggregators.aggregator_utils import prepare_grad_updates, wrapup_aggregated_grads
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from datapreprocessor.data_utils import dataset_class_indices, subset_by_idx
from fl.client import Client
from aggregators import aggregator_registry


@aggregator_registry
class FLTrust(AggregatorBase):
    """
    FLTrust 聚合器实现，通过服务器端可信数据集为客户端更新赋予信任权重。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 FLTrust，并基于服务器侧小样本数据创建“根”客户端。

        参数:
            args (argparse.Namespace | Any): 运行配置对象，需包含
                - num_classes (int): 数据集类别数。
                - num_sample_clients (可选) 等标准客户端参数。
                - defense_params (dict, optional): 用于覆盖默认 `num_sample`。
            **kwargs:
                - train_dataset (torch.utils.data.Dataset): 服务器可访问的干净数据集。

        返回:
            None

        异常:
            KeyError: 若未在 kwargs 中提供 `train_dataset`。
            AttributeError: 当 args 缺少必需属性时抛出。

        复杂度:
            时间复杂度 O(1)（站在初始化阶段）；空间复杂度 O(num_sample)。
        """
        super().__init__(args)
        """
        num_sample (int): 用于服务器可信模型训练的样本数量。
        """
        self.default_defense_params = {"num_sample": 100}
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

        train_dataset = kwargs['train_dataset']
        self._init_server_client(self.args, train_dataset)

    def _init_server_client(self, args, train_dataset):
        """
        基于服务器数据集采样小规模子集并初始化“根”客户端。

        参数:
            args (argparse.Namespace | Any): 运行配置对象。
            train_dataset (torch.utils.data.Dataset): 可供服务器使用的干净数据。

        返回:
            None

        异常:
            ValueError: 当可采样样本不足以满足 `num_sample` 要求时可能触发。

        复杂度:
            时间复杂度 O(num_classes * num_sample)；空间复杂度 O(num_sample)。
        """
        # 统计各类别索引，用于按比例抽样。
        train_class_indices = dataset_class_indices(train_dataset)
        class_counts = [len(class_indices)
                        for class_indices in train_class_indices]
        cls_sample_size = [int(i * self.num_sample / sum(class_counts))
                           for i in class_counts]
        indices = np.zeros(sum(cls_sample_size), dtype=np.int64)

        start_idx = 0
        for cls_id in range(self.args.num_classes):
            selected_indices = np.random.choice(
                train_class_indices[cls_id], size=cls_sample_size[cls_id], replace=False)
            indices[start_idx:start_idx +
                    cls_sample_size[cls_id]] = selected_indices
            start_idx += cls_sample_size[cls_id]

        sampled_set = subset_by_idx(self.args, train_dataset, indices)
        self.server_client = Client(args, -1, sampled_set)
        self.server_client.set_algorithm(self.algorithm)

    def aggregate(self, updates, **kwargs):
        """
        基于可信锚模型推断信任权重，对客户端更新执行归一化加权聚合。

        参数:
            updates (numpy.ndarray | list[numpy.ndarray]): 客户端上报的梯度或模型参数。
            **kwargs:
                - last_global_model (torch.nn.Module): 上一轮全局模型。
                - global_weights_vec (numpy.ndarray): 当前全局模型向量形式，供服务器客户端加载。

        返回:
            numpy.ndarray: 加权后的聚合梯度向量。

        异常:
            KeyError: 缺少必要的 kwargs 键。

        复杂度:
            时间复杂度 O(n * d + T_server)，其中 T_server 为服务器端微调耗时；
            空间复杂度 O(n * d)。
        """
        self.global_model = kwargs['last_global_model']
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model)

        # 1. 服务器客户端加载当前全局模型并执行本地训练，产生可信锚更新。
        global_weights_vec = kwargs["global_weights_vec"]
        self.server_client.load_global_model(global_weights_vec)
        self.server_client.local_training()
        self.server_client.fetch_updates(benign_flag=True)

        # 2. 计算服务器客户端的梯度向量作为信任参考。
        raw_shape = self.server_client.update.shape
        root_grad_update = prepare_grad_updates(
            self.args.algorithm,
            self.server_client.update.reshape(1, -1),
            self.global_model
        )
        root_grad_update.reshape(raw_shape)  # 注意：reshape 未赋值，保持与原实现一致

        # 3. 计算客户端梯度与锚梯度之间的余弦相似度作为信任得分。
        TS = cosine_similarity(
            gradient_updates, root_grad_update.reshape(1, -1))

        # 4. 仅保留正相关性，归一化后作为权重；若全为零则退化为均匀权重。
        TS = np.maximum(TS, 0)
        TS /= np.sum(TS) + 1e-9
        if not np.any(TS):
            TS = np.ones_like(TS) / len(TS)

        # 5. 对客户端梯度做幅度归一化，再与锚梯度范数对齐。
        normed_updates = gradient_updates / (
            np.linalg.norm(gradient_updates, axis=1).reshape(-1, 1) + 1e-9
        ) * np.linalg.norm(root_grad_update)

        # 依据信任权重对归一化梯度求加权平均。
        agg_grad_updates = np.average(
            normed_updates, axis=0, weights=np.squeeze(TS))

        return wrapup_aggregated_grads(
            agg_grad_updates, self.args.algorithm, self.global_model, aggregated=True)


# 费曼学习法解释 (FLTrust.__init__)
# (A) 功能概述：设定服务器侧抽样规模并创建可信锚客户端。
# (B) 类比说明：像先准备一小批经过审核的种子数据，用来校准所有参与者的标准。
# (C) 步骤拆解：
#     1. 保存联邦配置并读取服务器可用的干净数据集。
#     2. 设置默认抽样量 num_sample 并允许外部覆盖。
#     3. 调用 `_init_server_client` 抽取子集并创建服务器客户端。
# (D) 最小示例：
#     >>> class Args: num_classes=10; defense_params=None; algorithm='FedSGD'
#     >>> fltrust = FLTrust(Args(), train_dataset=train_ds)
# (E) 边界条件与测试建议：
#     - 需要确保 `train_dataset` 覆盖所有类别；样本过少可能导致抽样失败。
#     - 建议测试：自定义 `num_sample` 是否影响抽样规模。
# (F) 背景参考：FLTrust 论文；信任锚思想与鲁棒统计教材。


# 费曼学习法解释 (FLTrust._init_server_client)
# (A) 功能概述：按类别比例抽样服务器数据并初始化本地训练客户端。
# (B) 类比说明：像按照年级人数比例从学校挑选一批代表，供教师对照评分。
# (C) 步骤拆解：
#     1. 统计各类别样本数量，计算比例。
#     2. 按比例随机抽样，组合成服务器训练集。
#     3. 创建 `Client` 对象并设置其训练算法。
# (D) 最小示例：
#     >>> fltrust._init_server_client(args, train_dataset)
# (E) 边界条件与测试建议：
#     - 某些类别若不足以抽样，`np.random.choice` 会抛出异常。
#     - 建议测试：类别极不平衡的场景下抽样是否合理。
# (F) 背景参考：分层抽样、统计抽样理论。


# 费曼学习法解释 (FLTrust.aggregate)
# (A) 功能概述：利用锚梯度计算信任得分，对客户端更新做归一化加权聚合。
# (B) 类比说明：像让所有学生提交作业，与老师的标准答案比较相似度，越接近者被赋予更高权重。
# (C) 步骤拆解：
#     1. 将客户端更新转换成梯度表示。
#     2. 服务器客户端加载全局模型训练得到锚梯度。
#     3. 计算余弦相似度作为信任分数，对负值截断。
#     4. 若权重和为零，退化为均匀分布；否则归一化。
#     5. 对梯度做幅度归一化并与锚梯度范数对齐，再按权重加权平均。
# (D) 最小示例：
#     >>> agg = fltrust.aggregate(updates, last_global_model=global_model, global_weights_vec=vec)
# (E) 边界条件与测试建议：
#     - 当锚模型训练失败或梯度全零时需检验归一化分母。
#     - 建议测试：1) 纯良性场景权重应接近均匀；2) 插入方向相反的恶意梯度时权重显著下降。
# (F) 背景参考：余弦相似度、归一化加权平均、差分隐私与鲁棒聚合。


__AI_ANNOTATION_SUMMARY__ = """
FLTrust.__init__: 设置默认抽样规模并创建服务器侧可信锚客户端。
FLTrust._init_server_client: 按类别比例抽样服务器数据并初始化根客户端。
FLTrust.aggregate: 计算信任权重后对归一化梯度执行加权聚合。
"""
