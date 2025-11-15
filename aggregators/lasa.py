"""
LASA 聚合器：结合稀疏化与双重中位数检测的鲁棒联邦防御。

算法参考《LASA: Layer-Adaptive Sparsity for Byzantine-Robust Federated Learning》，
核心流程包括：对客户端更新施加 Top-k 稀疏化、按层分别进行基于中位数-Z 分数的幅度与方向过滤，
最后对筛选出的良性客户端执行裁剪后的平均聚合。
"""
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from aggregators import aggregator_registry
from fl.models.model_utils import state2vec, vec2state


@aggregator_registry
class LASA(AggregatorBase):
    """
    LASA 聚合器：层级自适应稀疏化与双重过滤结合的鲁棒聚合策略。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 LASA 聚合器并配置稀疏率与过滤阈值。

        参数:
            args (argparse.Namespace | Any): 运行配置对象，需包含
                - num_classes (int): 数据类别数，用于方向特征标准化。
                - defense_params (dict, optional): 可覆盖默认防御参数。
            **kwargs: 预留关键字参数，当前未使用。

        返回:
            None

        异常:
            AttributeError: 当 args 未包含 defense_params 时可能抛出。

        复杂度:
            时间复杂度 O(1); 空间复杂度 O(1)。
        """
        super().__init__(args)
        self.default_defense_params = {
            "norm_bound": 2, "sign_bound": 1, "sparsity": 0.3}
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

    def aggregate(self, updates, **kwargs):
        """
        执行 LASA 聚合流程：稀疏化、双重过滤并对良性梯度平均。

        参数:
            updates (numpy.ndarray): 客户端上传的梯度向量集合。
            **kwargs: 需要包含
                - last_global_model (torch.nn.Module): 上一轮全局模型。

        返回:
            numpy.ndarray: 聚合后的梯度向量。

        异常:
            KeyError: 缺少必要的 kwargs 键时抛出。

        复杂度:
            时间复杂度 O(n * d)，空间复杂度 O(n * d)；n 为客户端数，d 为参数维度。
        """
        num_clients = len(updates)
        self.global_model = kwargs['last_global_model']

        # 将每个客户端更新恢复为 state_dict 形式，以便按层处理。
        dict_form_updates = [
            vec2state(updates[i], self.global_model, numpy=True)
            for i in range(num_clients)
        ]

        # 1. 根据客户端范数中位数进行裁剪与缩放。
        client_norms = np.linalg.norm(updates, axis=1)
        median_norm = np.median(client_norms)
        grads_clipped_norm = np.clip(client_norms, a_min=0, a_max=median_norm)
        grad_clipped = (updates / client_norms.reshape(-1, 1)) * \
            grads_clipped_norm.reshape(-1, 1)
        dict_form_grad_clipped = [
            vec2state(grad_clipped[i], self.global_model, numpy=True)
            for i in range(num_clients)
        ]

        # 2. 对每个客户端独立实施 Top-k 稀疏化，仅保留显著参数。
        for i in range(len(dict_form_updates)):
            dict_form_updates[i] = self.sparse_update(dict_form_updates[i])

        key_mean_weight = {}
        for key in dict_form_updates[0].keys():
            if 'num_batches_tracked' in key:
                # BatchNorm 统计量不参与稀疏化/过滤。
                continue

            # 3. 提取当前层所有客户端的扁平化向量，准备进行统计过滤。
            key_flattened_updates = np.array([
                dict_form_updates[i][key].flatten()
                for i in range(num_clients)
            ])

            # 4. 第一道过滤：根据 L2 范数的中位数-Z 分数筛选异常幅度。
            grad_l2norm = np.linalg.norm(key_flattened_updates, axis=1)
            S1_benign_idx = self.mz_score(grad_l2norm, self.norm_bound)

            # 5. 第二道过滤：以符号一致性衡量更新方向，继续使用 MZ-score。
            layer_signs = np.empty(num_clients)
            for i in range(num_clients):
                sign_feat = np.sign(dict_form_updates[i][key])
                # 利用符号比例描述方向一致性，并结合稀疏率做归一化。
                layer_signs[i] = 0.5 * np.sum(sign_feat) / \
                    np.sum(np.abs(sign_feat)) * (1 - self.sparsity)
            S2_benign_idx = self.mz_score(layer_signs, self.sign_bound)

            # 两道过滤取交集，若交集为空则退化为全部客户端。
            benign_idx = list(set(S1_benign_idx).intersection(S2_benign_idx))
            benign_idx = benign_idx if len(
                benign_idx) != 0 else list(range(num_clients))

            # 6. 对通过过滤的客户端梯度求平均，获取当前层的聚合结果。
            key_mean_weight[key] = np.mean(
                [dict_form_grad_clipped[i][key] for i in benign_idx], axis=0)

        # 汇总层级结果并返回向量化表示。
        return state2vec(key_mean_weight, numpy_flg=True)

    def sparse_update(self, update):
        """
        采用 Top-k 策略对卷积层与全连接层参数执行稀疏化。

        参数:
            update (dict[str, numpy.ndarray]): 单个客户端的参数字典。

        返回:
            dict[str, numpy.ndarray]: 稀疏化后的参数字典。

        复杂度:
            时间复杂度 O(d log d)，空间复杂度 O(d)。
        """
        mask = {}
        for key in update.keys():
            if len(update[key].shape) == 4 or len(update[key].shape) == 2:
                mask[key] = np.ones_like(update[key], dtype=np.float32)
        if self.sparsity == 0.0:
            return mask

        # 抽取需要稀疏化层的绝对值，将所有权重拼接后寻找全局 Top-k。
        weight_abs = [np.abs(update[key])
                      for key in update.keys() if key in mask]
        all_scores = np.concatenate([value.flatten() for value in weight_abs])
        num_topk = int(len(all_scores) * (1 - self.sparsity))
        kth_largest = np.partition(all_scores, -num_topk)[-num_topk]

        for key in mask.keys():
            # 保留绝对值大于阈值的元素，其余置零，实现稀疏化。
            mask[key] = np.where(
                np.abs(update[key]) <= kth_largest, 0, mask[key])
            update[key].data *= mask[key]

        return update

    def mz_score(self, values, bound):
        """
        计算中位数-Z 分数并返回低于阈值的索引，用于异常检测。

        参数:
            values (numpy.ndarray): 待评估的特征向量。
            bound (float): 阈值，越小意味着容忍度越低。

        返回:
            numpy.ndarray: 满足条件的索引数组。

        复杂度:
            时间复杂度 O(n)，空间复杂度 O(1)。
        """
        med, std = np.median(values), np.std(values)
        # 注意：std 若为 0 会导致除零；当前实现保持原逻辑。
        for i in range(len(values)):
            values[i] = np.abs((values[i] - med) / std)
        return np.argwhere(values < bound).squeeze(-1)


# 费曼学习法解释 (LASA.__init__)
# (A) 功能概述：设定稀疏率与过滤阈值，并继承基础配置。
# (B) 类比说明：像在开赛前确定裁判判罚标准与允许的犯规次数。
# (C) 步骤：
#     1. 调用基类构造函数保存运行参数。
#     2. 设置默认的 norm_bound、sign_bound 与 sparsity。
#     3. 调用 update_and_set_attr 让配置生效。
# (D) 示例：
#     >>> class Args: defense_params=None; num_classes=10
#     >>> lasa = LASA(Args())
# (E) 边界与测试：缺少 defense_params 字段时会报错；建议测试自定义稀疏率是否覆盖成功。
# (F) 参考：《LASA》论文、稀疏训练与鲁棒统计教材。


# 费曼学习法解释 (LASA.aggregate)
# (A) 功能概述：对客户端梯度执行裁剪、稀疏化、双重过滤，最后聚合。
# (B) 类比说明：像先将每位选手的成绩归一化，再剔除异常夸张的表现，最后仅用合格者计算平均。
# (C) 步骤：
#     1. 将梯度转为 state_dict 方便逐层处理。
#     2. 按中位数裁剪客户端范数，避免极大值。
#     3. 对每个客户端的关键层执行 Top-k 稀疏化。
#     4. 逐层计算 L2 范数 MZ-score 过滤异常幅度。
#     5. 计算符号一致性并再次通过 MZ-score 过滤方向。
#     6. 对通过过滤的客户端梯度求均值，再恢复为向量。
# (D) 示例：
#     >>> agg = lasa.aggregate(updates, last_global_model=global_model)
# (E) 边界与测试：当稀疏率较高或 std=0 时需关注数值稳定性；建议测试纯良性与含异常客户端两种场景。
# (F) 参考：《LASA》论文；鲁棒聚合与 MZ-score 方法。


# 费曼学习法解释 (LASA.sparse_update)
# (A) 功能概述：在卷积与全连接层上保留最显著的 Top-k 权重。
# (B) 类比说明：像把一篇文章中最重要的句子高亮，其他部分淡化。
# (C) 步骤：
#     1. 为需稀疏化的层创建掩码。
#     2. 计算所有候选权重的绝对值并选取全局 Top-k。
#     3. 小于阈值的元素置零，仅保留关键权重。
# (D) 示例：
#     >>> sparse_state = lasa.sparse_update(client_state)
# (E) 边界与测试：当 sparsity=0 时直接返回掩码；建议测试不同 sparsity 对有效特征的影响。
# (F) 参考：稀疏表示与剪枝技术文献。


# 费曼学习法解释 (LASA.mz_score)
# (A) 功能概述：根据中位数-Z 分数判断输入是否接近中位水平。
# (B) 类比说明：像计算每个人离中等成绩的距离，距离太远的被视为异常。
# (C) 步骤：
#     1. 计算数据的中位数与标准差。
#     2. 将每个值与中位数的差除以标准差，取绝对值即 MZ-score。
#     3. 返回得分低于阈值的索引。
# (D) 示例：
#     >>> idx = lasa.mz_score(np.array([0.1, 2.0, 0.2]), bound=1.5)
# (E) 边界与测试：std=0 时会除零；建议在测试中覆盖该情况并考虑加 epsilon。
# (F) 参考：鲁棒统计教材、MZ-score 应用案例。


__AI_ANNOTATION_SUMMARY__ = """
LASA.__init__: 初始化稀疏率与过滤阈值，用于后续层级鲁棒聚合。
LASA.aggregate: 对客户端梯度执行裁剪、稀疏化与双重过滤后聚合。
LASA.sparse_update: 按 Top-k 策略稀疏化卷积与全连接层参数。
LASA.mz_score: 计算中位数-Z 分数筛选接近中位数的样本。
"""
