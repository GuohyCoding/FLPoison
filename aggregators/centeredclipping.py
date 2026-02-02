"""
Centered Clipping 聚合器：面向带动量更新的拜占庭鲁棒聚合策略。

算法来源于 ICML 2021《Learning from History for Byzantine Robust Optimization》，
核心思想是维护服务器端的历史动量，将客户端上传的动量更新先相对于历史动量
进行裁剪，再在累计结果上执行范数裁剪，以抑制异常梯度的影响。
"""
from copy import deepcopy
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
import torch
from aggregators import aggregator_registry


@aggregator_registry
class CenteredClipping(AggregatorBase):
    """
    Centered Clipping 聚合器实现：利用历史动量与范数裁剪提升鲁棒性。

    通过维护服务器端动量 `self.momentum`，对每轮客户端更新执行多次中心裁剪迭代，
    最终输出裁剪后的动量作为全局更新。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 Centered Clipping 聚合器并设定默认裁剪参数。

        参数:
            args (argparse.Namespace | Any): 运行配置对象，应包含 `defense_params`、`algorithm` 等字段。
            **kwargs: 预留的扩展参数，当前未使用。

        返回:
            None

        异常:
            AttributeError: 当 `args` 缺少所需字段（如 `defense_params`）时可能抛出。

        复杂度:
            时间复杂度 O(1)；空间复杂度 O(1)。
        """
        super().__init__(args)
        self.algorithm = "FedSGD"
        """
        norm_threshold (float): 聚合后梯度/动量的 L2 范数上限。
        num_iters (int): 中心裁剪迭代次数，原论文建议通常为 1~3。
        """
        # 设置默认参数并允许用户覆盖，以适应不同数据规模。
        self.default_defense_params = {
            "norm_threshold": 100, "num_iters": 1}
        self.update_and_set_attr()
        # 服务器端动量初始化为 None，首轮聚合时将会创建与梯度同形状的零向量。
        self.momentum = None

    def aggregate(self, updates, **kwargs):
        """
        聚合客户端上传的动量（或梯度）更新，并返回裁剪后的服务器动量。

        参数:
            updates (numpy.ndarray | list[numpy.ndarray]): 客户端上传的更新集合，
                通常为动量或梯度向量列表。
            **kwargs: 预留参数，当前未使用。

        返回:
            numpy.ndarray: 裁剪后的服务器端动量向量，将作为全局更新广播。

        异常:
            ValueError: 当 `updates` 为空集合或长度不一致时，np.zeros_like/norm 可能报错。

        复杂度:
            时间复杂度 O(T * n * d)，其中 T 为 `num_iters`、n 为客户端数、d 为参数维度；
            空间复杂度 O(d)。
        """
        if self.momentum is None:
            # 当首次聚合时，用与更新相同形状的零向量初始化服务器动量。
            if torch.is_tensor(updates):
                self.momentum = torch.zeros_like(updates[0])
            else:
                self.momentum = np.zeros_like(updates[0], dtype=np.float32)

        for _ in range(self.num_iters):
            # 逐轮执行中心裁剪：先将各客户端更新与历史动量之差进行裁剪，再求平均。
            if torch.is_tensor(updates):
                clipped = torch.stack(
                    [self.clip(v - self.momentum) for v in updates], dim=0
                )
                self.momentum = clipped.mean(dim=0) + self.momentum
            else:
                self.momentum = (
                    sum(self.clip(v - self.momentum)
                        for v in updates) / len(updates)
                    + self.momentum
                )

        # 返回深拷贝，避免调用方意外修改内部状态。
        return deepcopy(self.momentum)

    def clip(self, v):
        """
        对向量执行 L2 范数裁剪，限制其大小不超过 `norm_threshold`。

        参数:
            v (numpy.ndarray): 待裁剪的向量。

        返回:
            numpy.ndarray: 裁剪后的向量，与输入形状一致。

        异常:
            ValueError: 当 `np.linalg.norm` 遇到包含 NaN 的向量时可能抛出。

        复杂度:
            时间复杂度 O(d)；空间复杂度 O(1)。
        """
        # 计算裁剪比例，当向量范数超过阈值时按比例缩放。
        if torch.is_tensor(v):
            norm = torch.norm(v, p=2)
            scale = torch.clamp(
                self.norm_threshold / (norm + 1e-12),
                max=1.0,
            )
            return v * scale
        scale = min(1, self.norm_threshold / np.linalg.norm(v, ord=2))
        return v * scale


# 费曼学习法解释（CenteredClipping.__init__）
# (A) 功能概述：设置裁剪阈值与迭代次数，并初始化服务器动量。
# (B) 类比说明：像给滤波器设定最大增益和运行次数，并把滤波器状态清零。
# (C) 逐步拆解：
#     1. 调用父类构造函数保存运行配置。
#     2. 指定默认的范数阈值和迭代次数，允许外界覆盖。
#     3. 调用 `update_and_set_attr`；将配置同步到实例属性。
#     4. 将动量状态设为 None，表示尚未初始化。
# (D) 最小示例：
#     >>> class Args: defense_params=None; algorithm="FedSGD"
#     >>> cc = CenteredClipping(Args())
#     >>> cc.norm_threshold, cc.num_iters
#     (100, 1)
# (E) 边界条件与测试建议：
#     - 若 `args` 未提供 `defense_params` 字段，会触发 AttributeError。
#     - 建议测试：1) 默认参数是否正确设置；2) 自定义参数能否覆盖默认值。
# (F) 背景参考：
#     - 背景：动量方法常见于联邦优化，聚合器需要兼容动量形式。
#     - 推荐阅读：《Learning from History for Byzantine Robust Optimization》《Deep Learning》。


# 费曼学习法解释（CenteredClipping.aggregate）
# (A) 功能概述：多次执行中心裁剪，融合客户端动量并更新服务器动量。
# (B) 类比说明：像让多个报告先减去旧结论后再压缩，合并后再加回旧结论形成新共识。
# (C) 逐步拆解：
#     1. 若服务器尚无动量，则创建一个与更新同形状的零向量。
#     2. 重复 `num_iters` 轮中心裁剪：每轮都先计算客户端更新与当前动量的差值。
#     3. 对差值调用 `clip`，限制每个客户端的影响力。
#     4. 将裁剪后的差值求平均并加回动量，得到新的动量估计。
#     5. 返回动量的深拷贝，确保外部修改不影响内部状态。
# (D) 最小示例：
#     >>> import numpy as np
#     >>> updates = [np.array([2.0, 0.0]), np.array([0.0, 2.0])]
#     >>> cc.momentum = np.zeros(2)
#     >>> cc.aggregate(updates)
#     array([1., 1.])
# (E) 边界条件与测试建议：
#     - `updates` 需非空且形状一致；若存在极端大向量，应调整 `norm_threshold`。
#     - 建议测试：1) 在不同 `num_iters` 下结果是否稳定；2) 当 `updates` 全相同是否输出相同向量。
# (F) 背景参考：
#     - 背景：中心裁剪结合动量有助于减弱拜占庭噪声。
#     - 推荐阅读：《Learning from History for Byzantine Robust Optimization》《Robust Statistics》。


# 费曼学习法解释（CenteredClipping.clip）
# (A) 功能概述：将向量的 L2 范数限制在预设阈值以内。
# (B) 类比说明：像给音量设置上限，超过阈值就按比例压低。
# (C) 逐步拆解：
#     1. 计算向量的 L2 范数——判断是否超出阈值。
#     2. 用阈值除以范数得到缩放比例，若范数较小则比例为 1。
#     3. 将向量乘以比例，输出裁剪后的向量。
# (D) 最小示例：
#     >>> cc.norm_threshold = 5.0
#     >>> cc.clip(np.array([3.0, 4.0]))
#     array([3., 4.])
#     >>> cc.norm_threshold = 2.5
#     >>> cc.clip(np.array([3.0, 4.0]))
#     array([1.5, 2. ])
# (E) 边界条件与测试建议：
#     - 当输入向量全零时范数为 0，当前实现会出现除零；建议测试并在未来修复。
#     - 建议测试：1) 范数小于阈值时保持原值；2) 范数大于阈值时按比例缩放。
# (F) 背景参考：
#     - 背景：范数裁剪是联邦学习差分隐私与鲁棒性常用的预处理技术。
#     - 推荐阅读：《Deep Learning with Differential Privacy》《Machine Learning》。


__AI_ANNOTATION_SUMMARY__ = """
CenteredClipping.__init__: 初始化裁剪阈值、迭代次数并准备服务器端动量状态。
CenteredClipping.aggregate: 多轮中心裁剪融合客户端动量，输出鲁棒的全局更新。
CenteredClipping.clip: 将向量的 L2 范数限制在阈值以内，控制单个客户端影响。
"""
