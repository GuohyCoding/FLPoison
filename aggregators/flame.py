"""
FLAME 聚合器: 结合余弦聚类、裁剪与噪声注入的联邦鲁棒防御。

源自 USENIX Security 2022 "FLAME: Taming Backdoors in Federated Learning"。
流程概览:
1) 基于余弦距离的 HDBSCAN 聚类识别潜在良性客户端;
2) 按梯度范数中位数裁剪被接受的客户端更新;
3) 向聚合模型添加高斯噪声以满足差分隐私与鲁棒性需求。
"""
from copy import deepcopy
import torch
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
import hdbscan
from aggregators import aggregator_registry
from aggregators.aggregator_utils import normclipping, prepare_updates
from fl.models.model_utils import add_vec2model, model2vec


@aggregator_registry
class FLAME(AggregatorBase):
    """
    FLAME 聚合器实现, 先聚类再裁剪并加噪以抑制后门客户端。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 FLAME 聚合器并设置默认噪声缩放系数 gamma。

        参数:
            args (argparse.Namespace 或兼容对象): 联邦运行配置, 应包含
                algorithm (str): 联邦算法名称, 默认针对 FedAvg。
                defense_params (dict, 可选): 用于覆盖默认 gamma。
            **kwargs: 预留扩展参数, 当前未使用。

        返回:
            None

        异常:
            AttributeError: 当 args 缺少 defense_params 属性时可能抛出。

        复杂度:
            时间复杂度 O(1); 空间复杂度 O(1)。
        """
        super().__init__(args)
        self.algorithm = "FedAvg"
        self.default_defense_params = {"gamma": 1.2e-5}
        self.update_and_set_attr()

    def aggregate(self, updates, **kwargs):
        """
        执行 FLAME 聚合, 依次完成聚类筛选、裁剪、噪声注入并返回结果。

        参数:
            updates (numpy.ndarray 或 list[numpy.ndarray]): 客户端更新集合。
            **kwargs: 需要包含 last_global_model (torch.nn.Module)。

        返回:
            numpy.ndarray: 当算法为 FedAvg 时返回新模型向量, 否则返回增量梯度向量。

        异常:
            KeyError: 未提供 last_global_model 时抛出。
            RuntimeError: 聚类未识别出良性客户端或噪声添加失败时应由调用方捕获。

        复杂度:
            时间复杂度约 O(n * d + n log n); 空间复杂度 O(n * d)。
        """
        self.global_model = kwargs['last_global_model']
        model_updates, gradient_updates = prepare_updates(
            self.args.algorithm, updates, self.global_model)

        benign_idx = self.cosine_clustering(model_updates)
        if len(benign_idx) == 0:
            raise RuntimeError("FLAME 未找到良性客户端, 建议调整聚类参数或检查输入。")

        aggregated_model, median_norm = self.adpative_clipping(
            self.global_model, gradient_updates, benign_idx)
        self.add_noise2model(self.gamma * median_norm, aggregated_model)

        if self.args.algorithm == 'FedAvg':
            return model2vec(aggregated_model)
        else:
            return model2vec(aggregated_model) - model2vec(self.global_model)

    def cosine_clustering(self, model_updates):
        """
        基于余弦距离执行 HDBSCAN 聚类, 返回标签为 0 的客户端索引。

        参数:
            model_updates (numpy.ndarray): 客户端模型向量, 形状 (n, d)。

        返回:
            list[int]: 判定为良性的客户端索引。

        异常:
            ValueError: HDBSCAN 在样本不足或参数不当时可能抛出。

        复杂度:
            时间复杂度约 O(n log n) 到 O(n^2); 空间复杂度 O(n^2)。
        """
        cluster = hdbscan.HDBSCAN(
            metric="cosine",
            algorithm="generic",
            min_cluster_size=self.args.num_clients // 2 + 1,
            min_samples=1,
            allow_single_cluster=True,
        )
        if isinstance(model_updates, torch.Tensor):
            # HDBSCAN expects a NumPy array on CPU.
            model_updates = model_updates.detach().cpu().numpy()
        cluster.fit(model_updates.astype(np.float64, copy=False))
        return [idx for idx, label in enumerate(cluster.labels_) if label == 0]

    def adpative_clipping(self, last_global_model, gradient_updates, benign_idx):
        """
        使用梯度 L2 范数的中位数作为阈值裁剪良性客户端, 并返回聚合模型。

        参数:
            last_global_model (torch.nn.Module): 上一轮全局模型。
            gradient_updates (numpy.ndarray): 客户端梯度矩阵。
            benign_idx (Sequence[int]): 良性客户端索引。

        返回:
            tuple[torch.nn.Module, float]: (聚合模型, 梯度范数中位数)。

        复杂度:
            时间复杂度 O(n * d); 空间复杂度 O(d)。
        """
        if torch.is_tensor(gradient_updates):
            gradient_updates = gradient_updates.detach().cpu().numpy()
        median_norm = np.median(np.linalg.norm(gradient_updates, axis=1))
        clipped_gradient_updates = normclipping(
            gradient_updates[benign_idx], median_norm)
        aggregated_gradient = np.mean(clipped_gradient_updates, axis=0)
        aggregated_model = add_vec2model(aggregated_gradient, last_global_model)
        return aggregated_model, median_norm

    def add_noise2model(self, noise_scale, model, only_weights=True):
        """
        向模型参数注入高斯噪声, 可选择跳过偏置与 BN 统计量。

        参数:
            noise_scale (float): 噪声尺度, 通常与 median_norm 成正比。
            model (torch.nn.Module): 待加噪模型。
            only_weights (bool): True 时跳过 running_mean 等统计量。

        返回:
            None

        异常:
            RuntimeError: state_dict 无法正确加载时抛出。

        复杂度:
            时间复杂度 O(d); 空间复杂度 O(d)。
        """
        model_state_dict = deepcopy(model.state_dict())
        for key, param in model_state_dict.items():
            if only_weights and any(sub in key for sub in ['running_mean', 'running_var', 'num_batches_tracked']):
                continue
            std = noise_scale * (param.data.std() if param.data.numel() > 0 else 0.0)
            if std == 0:
                continue
            noise = torch.normal(mean=0, std=std, size=param.size()).to(param.device)
            param.data += noise
        model.load_state_dict(model_state_dict)


# 费曼学习法解释 (FLAME.__init__)
# (A) 做什么: 设置默认的噪声系数并继承基础聚合器配置。
# (B) 类比: 像开机前调整机器的默认灵敏度。
# (C) 步骤:
#     1. 记录全局配置, 继承基类属性。
#     2. 设定算法和默认防御参数。
#     3. 调用 update_and_set_attr 让参数生效。
# (D) 示例:
#     >>> class Args: algorithm='FedAvg'; defense_params=None
#     >>> flame = FLAME(Args())
#     >>> flame.gamma
#     1.2e-05
# (E) 边界与测试:
#     - 缺少 defense_params 属性会报错。
#     - 测试建议: 自定义 defense_params 能否覆盖 gamma。
# (F) 参考: FLAME 论文; Differential Privacy 基础。


# 费曼学习法解释 (FLAME.aggregate)
# (A) 做什么: 通过聚类、裁剪、加噪生成鲁棒聚合结果。
# (B) 类比: 先分组找出主流意见, 再给每人设音量上限并加入背景噪声。
# (C) 步骤:
#     1. 获取上一轮模型并整理客户端更新。
#     2. 用余弦聚类挑出疑似正常的客户端。
#     3. 以中位数范数裁剪良性梯度。
#     4. 按比例向模型添加噪声。
#     5. 根据算法语义返回模型向量或梯度。
# (D) 示例:
#     >>> result = flame.aggregate(updates, last_global_model=global_model)
# (E) 边界与测试:
#     - 聚类全被判为异常需调整参数。
#     - 测试建议: 纯良性 vs. 含异常客户端的行为对比。
# (F) 参考: FLAME 论文; HDBSCAN 文献。


# 费曼学习法解释 (FLAME.cosine_clustering)
# (A) 做什么: 用 HDBSCAN 对客户端更新做余弦聚类并挑出主簇。
# (B) 类比: 将方向相近的向量归为同一队伍。
# (C) 步骤:
#     1. 按设定初始化聚类器。
#     2. 拟合客户端模型更新。
#     3. 选取标签为 0 的客户端作为主簇。
# (D) 示例:
#     >>> benign = flame.cosine_clustering(model_updates)
# (E) 边界与测试:
#     - 客户端过少或更新一致需验证聚类结果。
#     - 测试建议: 构造离群更新是否被排除。
# (F) 参考: HDBSCAN 算法资料。


# 费曼学习法解释 (FLAME.adpative_clipping)
# (A) 做什么: 用梯度范数中位数裁剪良性梯度并生成聚合模型。
# (B) 类比: 给参与者设统一音量上线, 再求平均音量。
# (C) 步骤:
#     1. 计算所有梯度的 L2 范数中位数。
#     2. 裁剪良性梯度到该阈值以内。
#     3. 将裁剪后的梯度均值加到全局模型上。
# (D) 示例:
#     >>> agg_model, median = flame.adpative_clipping(global_model, grads, benign)
# (E) 边界与测试:
#     - median 为 0 时结果恒为零梯度。
#     - 测试建议: 单个异常梯度是否被抑制。
# (F) 参考: 范数裁剪与差分隐私文献。


# 费曼学习法解释 (FLAME.add_noise2model)
# (A) 做什么: 向模型参数依比例添加高斯噪声。
# (B) 类比: 给设备施加微小随机抖动以隐藏真实信号。
# (C) 步骤:
#     1. 深拷贝 state_dict 避免直接修改原参数。
#     2. 可选跳过 BN 统计量。
#     3. 依据参数标准差生成噪声并加到参数。
#     4. 重新加载更新后的 state_dict。
# (D) 示例:
#     >>> flame.add_noise2model(1e-4, model)
# (E) 边界与测试:
#     - 标准差为 0 的参数不会被加噪。
#     - 测试建议: 噪声尺度为 0 时模型不变; 噪声增大时方差增加。
# (F) 参考: 差分隐私与噪声注入资料。


__AI_ANNOTATION_SUMMARY__ = """
FLAME.__init__: 初始化 FLAME 并配置默认噪声系数 gamma。
FLAME.aggregate: 结合余弦聚类、自适应裁剪和噪声注入完成鲁棒聚合。
FLAME.cosine_clustering: 使用 HDBSCAN 对客户端模型更新做余弦聚类并返回主簇索引。
FLAME.adpative_clipping: 基于梯度范数中位数裁剪良性梯度并生成新模型。
FLAME.add_noise2model: 向模型参数注入高斯噪声可选跳过 BN 统计量。
"""
