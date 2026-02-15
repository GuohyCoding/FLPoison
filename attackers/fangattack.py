# -*- coding: utf-8 -*-

import numpy as np
import torch
from copy import deepcopy

from global_utils import actor
from attackers.pbases.mpbase import MPBase
from aggregators.krum import krum
from attackers import attacker_registry
from fl.client import Client


@attacker_registry
@actor('attacker', 'omniscient')
class FangAttack(MPBase, Client):
    """Fang 攻击：针对 Krum 聚合的模型投毒方法，利用部分全局信息构造可被选中的恶意更新。

    该实现基于 USENIX Security 2020 论文《Local Model Poisoning Attacks to Byzantine-Robust Federated Learning》，
    假设攻击者了解各恶意客户端的提交更新，并能模拟 Krum 聚合流程，以此构造被 Krum 选中的恶意向量。

    属性:
        default_attack_params (dict): 默认攻击参数，目前包含 `stop_threshold` 用于二分搜索停止条件。
        algorithm (str): 假定服务器聚合算法名称，默认为 `FedAvg`，用于确定扰动基准。
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        """初始化 FangAttack 攻击者，配置默认参数并继承客户端上下文。

        概述:
            调用 `Client` 构造函数获取联邦学习运行所需状态，设置停止阈值并注册假定聚合算法。

        参数:
            args (argparse.Namespace): 运行配置，需包含 `num_adv`、`algorithm` 等字段。
            worker_id (int): 当前恶意客户端编号。
            train_dataset (Dataset): 本地训练数据集。
            test_dataset (Dataset): 本地测试数据集。

        返回:
            None。

        异常:
            AttributeError: 若 `args` 缺失必要字段时由基类自动抛出。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)。

        费曼学习法:
            (A) 该函数让 Fang 攻击者具备基本参数与运行上下文。
            (B) 类比在战术推演前先设定默认作战阈值与情景假设。
            (C) 步骤拆解:
                1. 调用父类 `Client.__init__`，继承数据与训练接口。
                2. 设置默认的攻击超参数 `stop_threshold`。
                3. 调用 `update_and_set_attr` 合并外部配置。
                4. 记录聚合算法名称，便于后续确定扰动基准。
            (D) 示例:
                >>> attacker = FangAttack(args, worker_id=0, train_dataset=train, test_dataset=test)
                >>> attacker.stop_threshold
                1e-05
            (E) 边界条件与测试建议: 确保 `num_adv` ≥ 2 等先验条件满足；可测试默认参数是否正确写入。
            (F) 背景参考: 《Local Model Poisoning Attacks to Byzantine-Robust Federated Learning》。
        """
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {'stop_threshold': 1.0e-5}
        self.update_and_set_attr()
        self.algorithm = "FedAvg"

    def omniscient(self, clients):
        """在全知者假设下生成 Fang 攻击向量，使 Krum 选择恶意更新。

        概述:
            先估计攻击方向，再通过二分搜索找到使首个攻击者被 Krum 选中的扰动幅度，
            随后让其他攻击者提交相同（或近似）恶意向量作为支持者。

        参数:
            clients (List[Client]): 当前参与训练的客户端列表，需包含 `category` 与 `update` 属性。

        返回:
            numpy.ndarray: 形状为 `(num_adv, d)` 的攻击更新矩阵。

        异常:
            AssertionError: 当攻击者数量不足（≤1）时触发。

        复杂度:
            时间复杂度 O(a * b * d)，其中 a 为攻击者数量，b 为二分搜索迭代次数，d 为参数维度；
            空间复杂度 O(a * d)。

        费曼学习法:
            (A) 该函数让恶意客户端计算一种更新，使 Krum 聚合器误选恶意方向。
            (B) 类比作弊者观察同伙的作业答案（benign 假装），再调节自己的错误答案幅度直到老师选中它。
            (C) 步骤拆解:
                1. 收集所有攻击者的当前更新（视为“伪装的良性”），计算平均方向 `est_direction`。
                2. 根据聚合算法确定扰动基准（FedAvg 使用全局权重，FedSGD 使用 0）。
                3. 模拟增加支持者数量，从 1 到 `num_adv-1`，尝试让 Krum 选中恶意向量。
                4. 在每次模拟中使用二分搜索调整 `lambda_value`，不断缩小扰动幅度直到 Krum 切换选择或达到阈值。
                5. 找到合适的 `lambda_value` 后，构造首个攻击者的恶意更新向量。
                6. 令其他攻击者复制该向量（或在小球内扰动），形成支持者提交。
            (D) 示例:
                >>> crafted = attacker.omniscient(clients)
                >>> crafted.shape
                (attacker.args.num_adv, len(attacker.update))
            (E) 边界条件与测试建议: 要求攻击者数量大于 1；建议测试
                1) lambda 搜索能收敛；2) 输出矩阵各行相同且非零。
            (F) 背景参考: Krum 聚合算法原理、Byzantine 鲁棒聚合相关文献。
        """
        # 使用已获取的 update 与各类联邦算法兼容，不直接依赖权重或梯度。
        device = self.update.device if torch.is_tensor(self.update) else (
            self.global_weights_vec.device if torch.is_tensor(self.global_weights_vec) else torch.device("cpu")
        )
        before_attack_update = torch.stack(
            [
                c.update.detach().to(device=device)
                if torch.is_tensor(c.update)
                else torch.as_tensor(c.update, device=device, dtype=torch.float32)
                for c in clients
                if c.category == "attacker"
            ],
            dim=0,
        )
        attacker_updates = torch.zeros(
            (self.args.num_adv, before_attack_update.shape[1]),
            device=device,
            dtype=before_attack_update.dtype,
        )
        # 估计攻击方向：以攻击者提交的平均更新为参考，取符号方向。
        est_direction = torch.sign(torch.mean(before_attack_update, dim=0))

        # 根据聚合算法确定扰动基准（FedAvg 使用全局权重，FedSGD 则以 0 为基准）。
        if self.args.algorithm == "FedAvg":
            if torch.is_tensor(self.global_weights_vec):
                perturbation_base = self.global_weights_vec.detach().to(
                    device=device, dtype=before_attack_update.dtype
                )
            else:
                perturbation_base = torch.as_tensor(
                    self.global_weights_vec, device=device, dtype=before_attack_update.dtype
                )
        else:
            perturbation_base = torch.zeros_like(est_direction)

        # 模拟支持者数量，逐步寻找能被 Krum 选中的扰动大小。
        simulation_attack_number = 1
        assert self.args.num_adv > 1, "FangAttack requires more than 1 attacker"
        while simulation_attack_number < self.args.num_adv:
            lambda_value = 1.0
            # 构造模拟更新矩阵，前 num_adv 行使用攻击者的原始更新。
            simulation_updates = torch.empty(
                (self.args.num_adv + simulation_attack_number, before_attack_update.shape[1]),
                device=device,
                dtype=before_attack_update.dtype,
            )
            simulation_updates[:self.args.num_adv] = before_attack_update

            while True:
                # 在模拟矩阵后部填入同一恶意向量，模拟额外支持者。
                simulation_updates[self.args.num_adv:self.args.num_adv +
                                   simulation_attack_number] = perturbation_base - lambda_value * est_direction

                # 调用 Krum，对模拟集合进行选择，返回被选中向量的索引。
                min_idx = krum(simulation_updates,
                               simulation_attack_number, return_index=True)
                # 若 Krum 选择了恶意向量或 λ 已降至停止阈值以下，则结束搜索。
                if min_idx >= self.args.num_adv or lambda_value <= self.stop_threshold:
                    break
                # 否则缩小 λ，继续二分搜索。
                lambda_value *= 0.5

            simulation_attack_number += 1
            if min_idx >= self.args.num_adv:
                break

        # 设置首个攻击者的恶意更新。
        attacker_updates[0] = perturbation_base - lambda_value * est_direction

        # 让其余攻击者提交相同向量作为支持者，确保 Krum 更倾向于选择恶意方向。
        for i in range(1, self.args.num_adv):
            attacker_updates[i] = attacker_updates[0]

        # XXX：测试
        # print("malicious_updates: ", attacker_updates)
        return attacker_updates

    def sample_vectors(self, epsilon, w0_prime, num_byzantine):
        """在 ε 球内为支持者采样近邻向量（备用方法）。

        概述:
            持续采样随机向量，将其限制在以 `w0_prime` 为中心、半径 ε 的球内，
            生成除首个攻击者外的若干支持者更新（未在主流程中启用）。

        参数:
            epsilon (float): 球面半径，限定支持者扰动幅度。
            w0_prime (numpy.ndarray): 首要攻击者的恶意更新向量。
            num_byzantine (int): 恶意客户端总数（包含首个攻击者）。

        返回:
            numpy.ndarray: 形状为 `(num_byzantine-1, d)` 的支持者更新矩阵。

        异常:
            ValueError: 当 `num_byzantine` < 2 时无法生成支持者向量。

        复杂度:
            时间复杂度 O(k * d)，k 为采样次数；空间复杂度 O((num_byzantine-1) * d)。

        费曼学习法:
            (A) 该函数提供一种可选机制，在小球内为支持者制造“略有不同”的恶意更新。
            (B) 类比抛撒迷你干扰器，让每个同伙的信号都在主信号周围轻微波动。
            (C) 步骤拆解:
                1. 初始化空列表用于存储符合条件的向量。
                2. 随机采样向量，范围为 [w0_prime - ε, w0_prime + ε]。
                3. 计算与 `w0_prime` 的距离，若不超过 ε 则加入列表。
                4. 重复上述步骤直到收集到 `num_byzantine-1` 个向量。
                5. 将列表堆叠为矩阵返回。
            (D) 示例:
                >>> supports = attacker.sample_vectors(0.01, attacker_updates[0], attacker.args.num_adv)
            (E) 边界条件与测试建议: 若 ε 太小可能采样时间过长；可测试
                1) 返回矩阵形状是否正确；2) 每个向量距中心不超过 ε。
            (F) 背景参考: 球体采样（Sphere Sampling）、向量扰动方法。
        """
        if num_byzantine < 2:
            raise ValueError("num_byzantine must be at least 2 to sample supporters.")
        nearby_vectors = []
        while len(nearby_vectors) < num_byzantine - 1:
            random_vector = w0_prime + np.random.uniform(-epsilon, epsilon, w0_prime.shape)
            if np.linalg.norm(random_vector - w0_prime) <= epsilon:
                nearby_vectors.append(random_vector)
        return np.stack(nearby_vectors, axis=0)


# __AI_ANNOTATION_SUMMARY__
# 类 FangAttack: 针对 Krum 聚合的 Fang 模型投毒攻击实现。
# 方法 __init__: 初始化攻击参数与客户端上下文，记录假定聚合算法。
# 方法 omniscient: 通过模拟 Krum 选择构造恶意更新并生成支持者向量。
# 方法 sample_vectors: 在 ε 球内为支持者采样近邻恶意向量（备用）。
