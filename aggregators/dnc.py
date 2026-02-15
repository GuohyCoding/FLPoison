"""
DnC 聚合器：通过子采样与奇异值投影识别并过滤潜在恶意客户端。

本实现源自 NDSS 2021《Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning》，
核心策略是随机子采样参数维度，将中心化梯度投影到主奇异向量上获取异常得分，多轮重复后选取得分较小的客户端进行聚合。
"""
from aggregators.aggregator_utils import prepare_grad_updates, wrapup_aggregated_grads
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
import torch
from aggregators import aggregator_registry


@aggregator_registry
class DnC(AggregatorBase):
    """
    Divide-and-Conquer (DnC) 聚合器，用于在存在模型投毒时识别相对可靠的客户端。

    通过随机子采样参数、对中心化梯度执行奇异值分解并利用主奇异向量计算异常得分，
    重复多次后保留得分较低的客户端集合，再对其梯度进行聚合。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 DnC 聚合器并设定防御相关超参数。

        参数:
            args (argparse.Namespace | Any): 运行配置对象，应包含
                - num_clients (int): 客户端数量。
                - num_adv (int): 恶意客户端估计数量。
                - defense_params (dict, optional): 可覆盖默认防御参数。
            **kwargs: 预留关键字参数，当前未使用。

        返回:
            None

        异常:
            AttributeError: 当 args 缺少上述字段或 defense_params 属性时可能抛出。

        复杂度:
            时间复杂度 O(1)；空间复杂度 O(1)。
        """
        super().__init__(args)
        """
        subsample_frac (float): 子采样参数的比例，控制维度压缩程度。
        num_iters (int): 重复子采样与投影的迭代次数。
        fliter_frac (float): 过滤恶意客户端的比例系数，k = n - fliter_frac * f。
        """
        self.default_defense_params = {
            "subsample_frac": 0.2, "num_iters": 5, "fliter_frac": 1.0}
        self.update_and_set_attr()
        # DnC 原论文假设客户端上传梯度，默认针对 FedSGD 场景。
        self.algorithm = "FedSGD"

    def aggregate(self, updates, **kwargs):
        """
        通过多轮子采样与奇异向量投影筛选客户端并聚合其梯度。

        参数:
            updates (numpy.ndarray | list[numpy.ndarray]): 客户端上传的参数更新或梯度集合。
            **kwargs: 需包含
                - last_global_model (torch.nn.Module): 上一轮全局模型，用于梯度向量化。

        返回:
            numpy.ndarray: 聚合后的更新向量（FedSGD 梯度形式）。

        异常:
            KeyError: 当 kwargs 缺少 'last_global_model'。
            ValueError: 当 subsample_frac 过小导致子采样维度为 0 时需提前校验。

        复杂度:
            时间复杂度近似 O(num_iters * n * d_sub)，其中 d_sub 为子采样维度；
            空间复杂度 O(n * d_sub)。
        """
        # 保存上一轮全局模型，供向量转换与结果封装使用。
        self.global_model = kwargs['last_global_model']
        # 将客户端更新统一转换为梯度表示，支持不同算法共用。
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model)

        # 尽量在 GPU 上完成计算（若可用）
        if torch.is_tensor(gradient_updates):
            try:
                device = next(self.global_model.parameters()).device
            except StopIteration:
                device = gradient_updates.device
            gradient_updates = gradient_updates.to(device)
        num_param = gradient_updates.shape[1]
        # 初始化为所有客户端，随着迭代逐步交集缩小。
        benign_idx = set(range(self.args.num_clients))

        for _ in range(self.num_iters):
            # 1. 随机子采样参数维度以降低计算复杂度。
            sample_size = int(self.subsample_frac * num_param)
            # 防御性判断：确保 sample_size 至少为 1，避免后续错误。
            if sample_size <= 0:
                raise ValueError("subsample_frac 太小，导致子采样维度为 0。")
            param_idx = torch.randperm(num_param, device=gradient_updates.device)[:sample_size]

            # 根据子采样索引提取梯度子向量。
            sampled_grads = gradient_updates.index_select(1, param_idx)

            # 2. 计算均值并进行中心化，消除整体偏移影响。
            mu = torch.mean(sampled_grads, dim=0)
            centered_grads = sampled_grads - mu

            # 3. 通过奇异值分解获取主奇异向量，用于度量异常方向的投影强度。
            _, _, Vh = torch.linalg.svd(centered_grads, full_matrices=False)
            v = Vh[0, :]
            # 根据投影长度平方作为异常得分，越大越可能来自恶意客户端。
            score = torch.matmul(centered_grads, v)**2

            # 计算需保留的客户端数量，默认剔除 fliter_frac * num_adv 个客户端。
            k = int(self.args.num_clients - self.fliter_frac * self.args.num_adv)
            if k <= 0:
                raise ValueError("过滤数量过大，导致没有客户端被保留。")

            # 取得分最小的 k 个索引作为当前迭代的候选良性客户端。
            if k != len(score):
                dnc_idx = torch.topk(score, k, largest=False).indices
            else:
                dnc_idx = torch.arange(len(score), device=score.device)
            # 与历史良性集合取交集，保证多轮一致认为可靠。
            benign_idx = benign_idx.intersection(set(dnc_idx.detach().cpu().tolist()))

        # 使用最终保留的客户端梯度执行聚合并封装返回。
        return wrapup_aggregated_grads(
            gradient_updates[list(benign_idx)], self.args.algorithm, self.global_model)


# 费曼学习法解释（DnC.__init__）
# (A) 功能概述：设定 DnC 所需的子采样比例、迭代次数与过滤比例。
# (B) 类比说明：像配置一套筛查机器，需要先设定筛网密度、重复筛查次数，以及要剔除多少嫌疑样本。
# (C) 逐步拆解：
#     1. 调用父类构造函数保存联邦配置。
#     2. 设置默认的 `subsample_frac`、`num_iters`、`fliter_frac`。
#     3. 调用 `update_and_set_attr`，将用户自定义参数覆盖默认值。
#     4. 记录算法语义为 `FedSGD` 以提示后续处理逻辑。
# (D) 最小示例：
#     >>> class Args: num_clients=10; num_adv=2; defense_params=None
#     >>> dnc = DnC(Args())
#     >>> dnc.subsample_frac, dnc.num_iters
#     (0.2, 5)
# (E) 边界条件与测试建议：
#     - 缺失 `num_clients` 或 `num_adv` 会触发 AttributeError。
#     - 建议测试：1) 自定义防御参数能否覆盖默认值；2) `fliter_frac` 设置为 0 时是否保留全部客户端。
# (F) 背景参考：
#     - 背景：DnC 属于鲁棒聚合策略，针对模型投毒攻击。
#     - 推荐阅读：《Manipulating the Byzantine》、鲁棒统计教材。


# 费曼学习法解释（DnC.aggregate）
# (A) 功能概述：多轮子采样并根据奇异向量投影得分筛选较可信的客户端，再聚合其梯度。
# (B) 类比说明：像反复抽查商品的不同部位，找出异常离谱的批次并剔除，最后汇总剩余商品的平均质量。
# (C) 逐步拆解：
#     1. 获取全局模型并将客户端更新统一转换为梯度向量。
#     2. 初始化良性集合为全部客户端。
#     3. 重复 `num_iters` 次：
#         - 随机选取部分参数维度，降低计算量。
#         - 中心化子采样梯度，消除整体偏移。
#         - 对中心化矩阵做 SVD，取第一右奇异向量表示主要异常方向。
#         - 将梯度投影到该向量上并平方得到异常得分。
#         - 根据得分保留前 k 个最小值的客户端，并与历史集合求交集。
#     4. 用最终保留下的客户端梯度调用 `wrapup_aggregated_grads` 完成聚合。
# (D) 最小示例：
#     >>> updates = np.random.randn(10, 100)
#     >>> result = dnc.aggregate(updates, last_global_model=global_model)
# (E) 边界条件与测试建议：
#     - 若 `subsample_frac` 太小、`k` 非正需提前检测。
#     - 建议测试：1) 纯良性场景下多数客户端应被保留；2) 构造明显异常梯度时应被逐渐剔除。
# (F) 背景参考：
#     - 背景：奇异值分解常用于异常检测与降维。
#     - 推荐阅读：《Manipulating the Byzantine》、 《Matrix Computations》。


__AI_ANNOTATION_SUMMARY__ = """
DnC.__init__: 设置 DnC 聚合器的子采样比例、迭代次数与过滤比例并初始化运行环境。
DnC.aggregate: 多轮子采样与奇异向量投影筛选低得分客户端后聚合其梯度结果。
"""
