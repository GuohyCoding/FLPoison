# -*- coding: utf-8 -*-

from fl.client import Client
from global_utils import actor
from attackers.pbases.mpbase import MPBase
from attackers import attacker_registry


@attacker_registry
@actor('attacker', 'model_poisoning', 'non_omniscient')
class SignFlipping(MPBase, Client):
    """符号翻转攻击器：在非全知场景下直接反转本地更新符号以扰乱聚合。

    该攻击源自 ICML 2018 论文《Asynchronous Byzantine Machine Learning (the case of SGD)》，
    通过提交取反后的梯度或权重更新，使聚合方向偏离真实下降方向。

    属性:
        无额外属性，仅依赖 `MPBase` 和 `Client` 提供的公共成员。
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        """初始化 SignFlipping 攻击客户端。

        概述:
            调用 `Client` 构造函数继承联邦学习基础能力，无需额外参数。

        参数:
            args (argparse.Namespace): 运行配置，需包含标准客户端所需字段。
            worker_id (int): 当前攻击客户端编号。
            train_dataset (Dataset): 本地训练数据集。
            test_dataset (Dataset): 本地测试数据集。

        返回:
            None。

        异常:
            AttributeError: 当 `args` 缺失必需字段时由基类自动抛出。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)。

        费曼学习法:
            (A) 该函数只是把攻击者注册为常规客户端，不做额外设置。
            (B) 类比参赛者只需领取参赛号，不需携带特殊装备。
            (C) 步骤拆解:
                1. 调用 `Client.__init__` 完成数据加载与通信接口初始化。
            (D) 示例:
                >>> attacker = SignFlipping(args, worker_id=0, train_dataset=train, test_dataset=test)
            (E) 边界条件与测试建议: 确保 `args` 满足 `Client` 的输入要求；可测试初始化后 `update` 属性是否由父类维护。
            (F) 背景参考: 联邦学习客户端抽象、符号翻转攻击基本概念。
        """
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)

    def non_omniscient(self):
        """返回符号翻转后的更新向量，用于非全知攻击接口。

        概述:
            直接将当前本地更新 `self.update` 取相反数并返回。

        参数:
            无。

        返回:
            numpy.ndarray 或 Tensor: 与 `self.update` 同形状的取反更新。

        异常:
            AttributeError: 当 `self.update` 未被父类计算时。

        复杂度:
            时间复杂度 O(d)，d 为模型参数维度；空间复杂度 O(1)（共享缓冲区）。

        费曼学习法:
            (A) 函数把更新方向完全反向，迫使聚合器沿反方向移动。
            (B) 类比团队合作中有人故意向相反方向使劲拖后腿。
            (C) 步骤拆解:
                1. 读取父类计算得到的 `self.update`。
                2. 对其取负号，得到反向更新向量。
                3. 返回该向量供服务器聚合。
            (D) 示例:
                >>> flipped = attacker.non_omniscient()
                >>> np.allclose(flipped, -attacker.update)
                True
            (E) 边界条件与测试建议: 确保 `self.update` 已由训练流程更新；可测试取反后范数不变、方向相反。
            (F) 背景参考: 符号翻转攻击、鲁棒联邦聚合分析。
        """
        # 直接对本地更新取反，破坏聚合方向。
        return - self.update


# __AI_ANNOTATION_SUMMARY__
# 类 SignFlipping: 提供符号翻转模型投毒攻击的客户端实现。
# 方法 __init__: 调用父类初始化客户端上下文，无额外配置。
# 方法 non_omniscient: 输出取反后的更新向量以实施符号翻转攻击。
