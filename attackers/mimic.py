# -*- coding: utf-8 -*-

import torch
from global_utils import actor
from attackers.pbases.mpbase import MPBase
from attackers import attacker_registry
from fl.client import Client


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class Mimic(MPBase, Client):
    """Mimic 攻击：模仿某个固定良性客户端的更新以持续放大其影响力。

    基于 ICLR 2022 论文《Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing》，
    攻击者在全知场景下选择一个良性客户端，复制其更新提交，使聚合器偏向该客户端的方向。

    属性:
        default_attack_params (dict): 默认设置包含 `choice`，表示被模仿的良性客户端索引。
        algorithm (str): 默认设定为 `FedSGD`，与论文设置保持一致。
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        """初始化 Mimic 攻击者，并指定要模仿的良性客户端。

        概述:
            继承联邦客户端上下文，设定默认的模仿目标索引与假定聚合算法。

        参数:
            args (argparse.Namespace): 运行配置，需包含 `num_adv`、客户端列表等信息。
            worker_id (int): 当前攻击客户端编号。
            train_dataset (Dataset): 本地训练数据集（攻击不直接使用）。
            test_dataset (Dataset): 本地测试数据集。

        返回:
            None。

        异常:
            AttributeError: 当 `args` 缺少必要字段时由基类抛出。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)。

        费曼学习法:
            (A) 该函数设定 Mimic 攻击默认模仿哪位良性客户端。
            (B) 类比在考试中选定一个学霸并复制 TA 的答案。
            (C) 步骤拆解:
                1. 调用 `Client.__init__` 获取联邦训练上下文。
                2. 设置默认参数 `choice=0`，表示模仿第一个良性客户端。
                3. 调用 `update_and_set_attr` 允许外部覆盖该索引。
                4. 记录默认聚合算法为 `FedSGD`。
            (D) 示例:
                >>> attacker = Mimic(args, worker_id=0, train_dataset=train, test_dataset=test)
                >>> attacker.choice
                0
            (E) 边界条件与测试建议: 确保 `choice` 小于参与客户端数量；
                建议测试默认值是否可被覆写。
            (F) 背景参考: Mimic 攻击、Bucketing 聚合策略。
        """
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {'choice': 0}
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

    def omniscient(self, clients):
        """全知场景下复制指定良性客户端的更新，作为所有攻击者的提交。

        概述:
            在所有客户端中选定索引 `choice`，取其更新向量，
            并复制 `num_adv` 份提交，以偏移聚合结果。

        参数:
            clients (List[Client]): 当前参与训练的客户端列表。

        返回:
            numpy.ndarray: 形状 `(num_adv, d)` 的更新矩阵。

        异常:
            AssertionError: 当 `choice` 超出客户端数量时触发。

        复杂度:
            时间复杂度 O(d)，d 为模型参数维度；空间复杂度 O(d)。

        费曼学习法:
            (A) 函数让每个攻击者都提交与某个良性客户端完全相同的更新。
            (B) 类比考试中多人抄写同一份答案，增加该答案被计入的权重。
            (C) 步骤拆解:
                1. 检查 `choice` 是否落在客户端索引范围内。
                2. 获取被选中客户端的更新向量。
                3. 使用 `np.tile` 将该更新复制 `num_adv` 份。
            (D) 示例:
                >>> mimic_updates = attacker.omniscient(clients)
                >>> mimic_updates.shape
                (attacker.args.num_adv, len(attacker.update))
            (E) 边界条件与测试建议: 确保 `clients` 中包含足够多的良性客户端；
                建议测试 1) choice 超界时触发断言；2) 返回矩阵各行一致。
            (F) 背景参考: Mimic 攻击策略、联邦聚合偏置分析。
        """
        assert self.choice < len(clients), f"choice {self.choice} is out of range"
        device = self.args.device
        update = clients[self.choice].update
        if torch.is_tensor(update):
            attack_vec = update.detach().to(device)
        else:
            attack_vec = torch.as_tensor(update, device=device)
        # repeat attack vector for all attackers
        return attack_vec.unsqueeze(0).repeat(self.args.num_adv, 1)


# __AI_ANNOTATION_SUMMARY__
# 类 Mimic: 模仿固定良性客户端更新的模型投毒攻击器。
# 方法 __init__: 设置默认模仿对象索引与聚合算法。
# 方法 omniscient: 复制指定客户端的更新并批量提交。
