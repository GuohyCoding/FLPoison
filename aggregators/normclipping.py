"""
范数裁剪聚合器：裁剪客户端梯度的范数并可选加噪以提升联邦学习鲁棒性。

实现依据 NeurIPS 2020《Can You Really Backdoor Federated Learning》，主要步骤：
- 对每个客户端梯度执行 L2 范数裁剪，抑制异常更新；
- 视需要向裁剪后的梯度添加高斯噪声（弱差分隐私）；
- 返回符合当前算法语义的聚合结果。
"""
from aggregators.aggregatorbase import AggregatorBase
from aggregators import aggregator_registry
from aggregators.aggregator_utils import (
    addnoise,
    normclipping,
    prepare_grad_updates,
    wrapup_aggregated_grads,
)


@aggregator_registry
class NormClipping(AggregatorBase):
    """
    NormClipping 聚合器：对客户端梯度进行范数裁剪，并可按需添加噪声。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 NormClipping 聚合器并配置默认裁剪阈值与噪声参数。

        参数:
            args (argparse.Namespace | Any): 联邦运行配置对象，应包含
                - algorithm (str): 当前联邦算法名称，如 'FedOpt'。
                - defense_params (dict, optional): 用户自定义的防御参数。
            **kwargs: 预留关键字参数，当前未使用。

        返回:
            None

        异常:
            AttributeError: 当 args 缺少 defense_params 属性时可能抛出。

        复杂度:
            时间复杂度 O(1)；空间复杂度 O(1)。
        """
        super().__init__(args)
        self.default_defense_params = {
            "weakDP": False,
            "norm_threshold": 3,
            "noise_mean": 0,
            "noise_std": 0.002,
        }
        self.update_and_set_attr()
        self.algorithm = "FedOpt"

    def aggregate(self, updates, **kwargs):
        """
        对客户端更新执行范数裁剪，并在需要时加噪后返回聚合结果。

        参数:
            updates (numpy.ndarray | list[numpy.ndarray]): 客户端上传的梯度或参数更新。
            **kwargs: 需要包含
                - last_global_model (torch.nn.Module): 上一轮全局模型。

        返回:
            numpy.ndarray: 裁剪（及加噪）后的聚合向量，符合当前算法语义。

        异常:
            KeyError: 缺少 'last_global_model' 时抛出。

        复杂度:
            时间复杂度 O(n * d)；空间复杂度 O(n * d)，n 为客户端数量，d 为参数维度。
        """
        self.global_model = kwargs["last_global_model"]
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model
        )

        # 1. 逐客户端执行 L2 范数裁剪，抑制梯度幅度异常。
        normed_updates = normclipping(gradient_updates, self.norm_threshold)

        # 2. 若启用弱差分隐私，则向裁剪后的梯度添加高斯噪声。
        if self.weakDP:
            normed_updates = addnoise(
                normed_updates, self.noise_mean, self.noise_std
            )

        # 3. 返回聚合结果；FedOpt 场景下仍保持梯度语义。
        return wrapup_aggregated_grads(
            normed_updates, self.args.algorithm, self.global_model
        )


# 费曼学习法解释（NormClipping.__init__）
# (A) 功能概述：设定裁剪阈值与噪声参数，确保范数裁剪策略可执行。
# (B) 类比说明：像给仪器设置最大量程并调好背景噪声级别。
# (C) 步骤拆解：
#     1. 调用基类构造函数保存联邦配置。
#     2. 设置默认的 `norm_threshold`、`weakDP`、噪声均值与方差，并允许覆盖。
#     3. 标记当前算法类型为 'FedOpt'。
# (D) 最小示例：
#     >>> class Args: algorithm='FedOpt'; defense_params=None
#     >>> nc = NormClipping(Args())
# (E) 边界条件与测试建议：
#     - 未提供 defense_params 属性会触发 AttributeError。
#     - 建议测试自定义阈值是否覆盖默认值。
# (F) 参考：NeurIPS 2020《Can You Really Backdoor Federated Learning》；差分隐私教材。


# 费曼学习法解释（NormClipping.aggregate）
# (A) 功能概述：对客户端梯度执行范数裁剪，并在需要时添加噪声后返回结果。
# (B) 类比说明：像限制每位发言者的音量不超过阈值，再根据需要加入背景噪声保护隐私。
# (C) 步骤拆解：
#     1. 将客户端更新统一转换为梯度向量。
#     2. 调用 `normclipping` 对每个梯度执行 L2 裁剪。
#     3. 如果启用弱差分隐私，向裁剪后的梯度添加高斯噪声。
#     4. 使用 `wrapup_aggregated_grads` 生成与算法语义一致的聚合结果。
# (D) 最小示例：
#     >>> agg = nc.aggregate(updates, last_global_model=global_model)
# (E) 边界条件与测试建议：
#     - 阈值过小会导致过度裁剪，噪声过大影响收敛。
#     - 建议测试：关闭弱DP 时输出是否仅裁剪；开启弱DP 时噪声统计是否符合预期。
# (F) 参考：差分隐私与鲁棒聚合相关文献。


__AI_ANNOTATION_SUMMARY__ = """
NormClipping.__init__: 配置梯度裁剪阈值与噪声参数，初始化范数裁剪聚合器。
NormClipping.aggregate: 对客户端梯度执行范数裁剪并可选加噪后返回聚合结果。
"""
