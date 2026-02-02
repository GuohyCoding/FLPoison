# -*- coding: utf-8 -*-

import torch

from global_utils import actor
from attackers.pbases.mpbase import MPBase
from attackers import attacker_registry
from fl.client import Client


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class IPM(MPBase, Client):
    """IPM（Inner Product Manipulation）攻击：通过提交缩放的负均值梯度来抵消聚合方向。

    该实现参考 UAI 2020 论文《Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation》，
    针对带鲁棒性的聚合算法（如 Krum、Bulyan、几何中位数）调节缩放系数，利用全知者假设构造对抗性更新。

    属性:
        default_attack_params (dict): 默认攻击参数，包含 `scaling_factor` 与 `attack_start_epoch`。
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        """初始化 IPM 攻击客户端并设置缩放因子。

        概述:
            继承联邦客户端上下文，写入默认缩放系数及攻击起始轮次，并允许外部覆盖。

        参数:
            args (argparse.Namespace): 运行配置，需包含 `num_adv`、`num_clients`、`algorithm` 等字段。
            worker_id (int): 当前攻击者编号。
            train_dataset (Dataset): 本地训练数据集（攻击不直接使用，但保持接口）。
            test_dataset (Dataset): 本地测试数据集。

        返回:
            None。

        异常:
            AttributeError: 若 `args` 缺失字段时由基类自动抛出。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)。

        费曼学习法:
            (A) 该函数为 IPM 攻击者设定默认缩放参数与攻击起始轮次。
            (B) 类比在比赛中提前调整“反向力”，等合适时间点再施加。
            (C) 步骤拆解:
                1. 调用 `Client.__init__` 继承通信与模型状态。
                2. 设定默认 `scaling_factor` 与 `attack_start_epoch`。
                3. 调用 `update_and_set_attr` 合并外部配置。
            (D) 示例:
                >>> attacker = IPM(args, worker_id=0, train_dataset=train, test_dataset=test)
                >>> attacker.scaling_factor
                0.1
            (E) 边界条件与测试建议: 确保 `scaling_factor` 为正数；可测试默认参数能被覆写。
            (F) 背景参考: Inner Product Manipulation 攻击、鲁棒聚合算法原理。
        """
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        # scale=0.1, 0.5, 1, 2, 100 均可破坏 FedAvg 与 GeometricMedian；0.1 等小尺度可破坏 Krum。
        self.default_attack_params = {
            'scaling_factor': 0.1,
            "attack_start_epoch": None,
        }
        self.update_and_set_attr()

    def omniscient(self, clients):
        """在全知者假设下构造 IPM 攻击更新，提交缩放的负均值梯度。

        概述:
            若达到攻击启动轮次，计算所有良性客户端更新的均值，取其相反方向并按缩放系数放大，
            最终为所有攻击者复制该向量，实现方向抵消。

        参数:
            clients (List[Client]): 当前参与训练的客户端列表，须包含 `category` 与 `update`。

        返回:
            numpy.ndarray 或 None: 若未到攻击轮次返回 `None`；否则返回形状 `(num_adv, d)` 的攻击更新矩阵。

        异常:
            ValueError: 当良性客户端集合为空时，`np.mean` 会触发警告或错误。

        复杂度:
            时间复杂度 O(m * d)，m 为良性客户端数，d 为参数维度；空间复杂度 O(d)。

        费曼学习法:
            (A) 函数让攻击者提交与良性更新相反方向的向量，从而削弱聚合效果。
            (B) 类比接力赛中偷偷往反方向跑，以抵消团队整体前进。
            (C) 步骤拆解:
                1. 检查是否到达攻击起始轮次；未到则保持潜伏。
                2. 收集所有良性客户端的更新向量并求均值，估计真实梯度方向。
                3. 取均值的相反方向并乘以缩放系数，获得攻击向量。
                4. 将该向量复制 `num_adv` 份，供所有攻击者提交。
            (D) 示例:
                >>> attack = attacker.omniscient(clients)
                >>> attack.shape
                (attacker.args.num_adv, len(attacker.update))
            (E) 边界条件与测试建议: 确保存在良性客户端；测试
                1) 未到攻击轮次返回 `None`；
                2) 攻击轮次生成的向量方向与均值相反且缩放正确。
            (F) 背景参考: Inner Product Manipulation 论文、鲁棒聚合对抗分析。
        """
        if self.attack_start_epoch is not None and self.global_epoch <= 2 + self.attack_start_epoch:
            return None
        device = self.args.device
        benign_updates = torch.stack(
            [
                i.update.detach().to(device)
                if torch.is_tensor(i.update)
                else torch.as_tensor(i.update, device=device)
                for i in clients
                if i.category == "benign"
            ],
            dim=0,
        )
        mean = benign_updates.mean(dim=0)
        attack_vec = -float(self.scaling_factor) * mean
        # repeat attack vector for all attackers
        return attack_vec.unsqueeze(0).repeat(self.args.num_adv, 1)


# __AI_ANNOTATION_SUMMARY__
# 类 IPM: 通过内积操控构造缩放负梯度的模型投毒攻击器。
# 方法 __init__: 初始化默认缩放因子与攻击起始轮次。
# 方法 omniscient: 生成缩放负均值梯度，为所有攻击者复制提交。
