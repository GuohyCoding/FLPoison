"""
CRFL 聚合器：针对后门攻击的可认证鲁棒联邦学习机制。

实现出自 ICML 2021《CRFL: Certifiably Robust Federated Learning against Backdoor Attacks》，
核心流程包括：均值聚合梯度、执行全局范数裁剪，再加入高斯噪声以提供鲁棒与隐私保障。
"""
from copy import deepcopy
from aggregators.aggregator_utils import addnoise, prepare_grad_updates, wrapup_aggregated_grads
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from aggregators import aggregator_registry


@aggregator_registry
class CRFL(AggregatorBase):
    """
    CRFL 聚合器：通过裁剪与加噪结合提升后门防御能力。

    先对客户端梯度求均值，再按阈值裁剪范数，随后注入高斯噪声，
    以限制恶意客户端影响并提供概率意义上的鲁棒保证。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 CRFL 聚合器，设定裁剪阈值与噪声参数。

        参数:
            args (argparse.Namespace | Any): 运行配置对象，应至少包含
                - algorithm (str): 当前联邦算法名称，CRFL 原设定为 'FedOpt'。
                - defense_params (dict, optional): 用户自定义的防御参数。
            **kwargs: 预留关键字参数，当前未使用。

        返回:
            None

        异常:
            AttributeError: 当 args 缺少上述字段或 `defense_params` 属性时可能抛出。

        复杂度:
            时间复杂度 O(1)；空间复杂度 O(1)。
        """
        super().__init__(args)
        self.algorithm = 'FedOpt'
        # 默认范数阈值与噪声配置，可视数据尺度与防御需求调整。
        self.default_defense_params = {
            "norm_threshold": 3, "noise_mean": 0, "noise_std": 0.001}
        self.update_and_set_attr()

    def aggregate(self, updates, **kwargs):
        """
        对客户端更新执行均值聚合、范数裁剪并加噪，输出鲁棒更新。

        参数:
            updates (numpy.ndarray | list[numpy.ndarray]): 客户端上传的更新集合。
            **kwargs: 需包含
                - last_global_model (torch.nn.Module): 上一轮全局模型，用于封装返回值。

        返回:
            numpy.ndarray: 经裁剪加噪后的聚合向量（FedOpt 梯度形式）。

        异常:
            KeyError: 若 kwargs 缺少 'last_global_model'。
            ValueError: 当 updates 为空导致均值或范数计算失败时。

        复杂度:
            时间复杂度 O(n * d)，n 为客户端数、d 为参数维度；空间复杂度 O(d)。
        """
        # 保存上一轮全局模型，以便 wrapup_aggregated_grads 根据算法语义返回结果。
        self.global_model = kwargs['last_global_model']
        # 将客户端更新统一转换为梯度形式，支持不同联邦优化算法共享流程。
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model)

        # 1. 对梯度求均值，得到基础聚合结果。
        agg_update = np.mean(gradient_updates, axis=0)
        # 2. 执行范数裁剪，防止异常大梯度主导全局更新。
        normed_agg_update = agg_update * \
            min(1, self.norm_threshold / (np.linalg.norm(agg_update)+1e-10))

        # 3. 加入高斯噪声以提升鲁棒性和可认证性，aggregated=True 表示无需再次求均值。
        return wrapup_aggregated_grads(
            addnoise(normed_agg_update, self.noise_mean, self.noise_std),
            self.args.algorithm, self.global_model, aggregated=True)


# 费曼学习法解释（CRFL.__init__）
# (A) 功能概述：设定 CRFL 所需的范数阈值与噪声参数，为聚合流程做准备。
# (B) 类比说明：像在搭建滤波器前先确定最大输出幅度和噪声注入强度，确保系统安全运行。
# (C) 逐步拆解：
#     1. 调用父类构造函数记录全局配置。
#     2. 声明当前使用的联邦算法类型（FedOpt）。
#     3. 设置默认的范数阈值与噪声参数。
#     4. 调用 `update_and_set_attr`，让用户自定义配置覆盖默认值并写入实例属性。
# (D) 最小示例：
#     >>> class Args: algorithm='FedOpt'; defense_params={'norm_threshold': 2.5}
#     >>> crfl = CRFL(Args())
#     >>> crfl.norm_threshold, crfl.noise_std
#     (2.5, 0.001)
# (E) 边界条件与测试建议：
#     - 若 `args` 缺少 `defense_params` 属性会抛出 AttributeError。
#     - 建议测试：1) 默认参数是否设置正确；2) 提供自定义参数时能否覆盖默认值。
# (F) 背景参考：
#     - 背景：CRFL 结合裁剪与噪声以构建可认证鲁棒性。
#     - 推荐阅读：《CRFL: Certifiably Robust Federated Learning against Backdoor Attacks》《Robust Statistics》。


# 费曼学习法解释（CRFL.aggregate）
# (A) 功能概述：对客户端梯度做均值、裁剪、加噪，生成鲁棒的联邦更新。
# (B) 类比说明：像把所有人的意见平均后，先限制“音量”过大的意见，再在结果中加入细微噪声保护隐私与稳健性。
# (C) 逐步拆解：
#     1. 读取上一轮全局模型，供结果封装使用。
#     2. 将原始更新转换为梯度向量，保证不同算法语义统一。
#     3. 计算梯度均值，得到初步全局更新。
#     4. 计算均值向量的范数，并按阈值裁剪，防止异常放大。
#     5. 对裁剪后的向量加入高斯噪声，进一步削弱恶意影响并提供概率保证。
#     6. 调用 `wrapup_aggregated_grads`，依据算法类型返回最终更新。
# (D) 最小示例：
#     >>> import numpy as np
#     >>> class Args: algorithm='FedOpt'; defense_params=None
#     >>> crfl = CRFL(Args())
#     >>> crfl.global_model = dummy_model  # 需提供与 wrapup 兼容的模型
#     >>> grads = np.array([[0.5, 0.0], [0.6, -0.1]])
#     >>> result = crfl.aggregate(grads, last_global_model=dummy_model)
# (E) 边界条件与测试建议：
#     - 输入更新必须非空且形状一致；阈值过小可能导致过度裁剪。
#     - 建议测试：1) 噪声标准差设为 0 时结果是否仅受裁剪影响；2) 极端大梯度是否被有效限制。
# (F) 背景参考：
#     - 背景：范数裁剪与加噪结合可提供差分隐私与鲁棒性双重保障。
#     - 推荐阅读：《CRFL: Certifiably Robust Federated Learning against Backdoor Attacks》《Deep Learning with Differential Privacy》。


__AI_ANNOTATION_SUMMARY__ = """
CRFL.__init__: 配置范数阈值与噪声参数，初始化 CRFL 聚合器状态。
CRFL.aggregate: 对客户端梯度执行均值、裁剪与加噪，生成鲁棒联邦更新。
"""
