# -*- coding: utf-8 -*-

import numpy as np
import torch
from attackers.pbases.dpbase import DPBase
from attackers.pbases.mpbase import MPBase
from datapreprocessor.edge_dataset import EdgeDataset
from global_utils import actor
from fl.models.model_utils import model2vec, vec2model
from attackers import attacker_registry
from .synthesizers import DatasetSynthesizer
from fl.client import Client

# TODO: test asr when pixel-based backdoor attack's bug is fixed


@attacker_registry
@actor('attacker', 'data_poisoning', 'model_poisoning', 'non_omniscient')
class EdgeCase(MPBase, DPBase, Client):
    """EdgeCase 边缘样本后门攻击器，结合稀有样本、PGD 投影与模型缩放实现多重投毒。

    本实现基于 NeurIPS 2020 论文《Attack of the Tails: Yes, You Really Can Backdoor Federated Learning》，
    通过引入尾部分布数据（EdgeDataset）并在每轮训练后执行 PGD 投影，与缩放型模型替换配合提升后门成功率。

    属性:
        default_attack_params (dict): 攻击默认参数，包括投毒比例、PGD 半径、缩放因子等。
        synthesizer (DatasetSynthesizer): 将原始训练集与边缘样本混合的合成器。
        poisoned_set (Tuple[Dataset, Dataset]): 缓存的投毒训练/测试数据集，用于按需替换。
        train_loader (Iterator): 根据投毒标志返回批量数据的迭代器。
        algorithm (str): 假定的聚合算法名称，默认设置为 `FedOpt`。

    说明:
        文件顶部的 TODO 暗示当前实现尚未重新测试 ASR（攻击成功率），待像素后门流程修复后需要补充验证。
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        """初始化 EdgeCase 攻击客户端，配置投毒流程与 PGD/缩放策略。

        概述:
            继承联邦客户端上下文，设置默认参数，构建边缘样本合成器，并初始化投毒版训练数据加载器。

        参数:
            args (argparse.Namespace): 运行配置，需包含 `batch_size`、`num_workers`、`epochs`、`algorithm` 等字段。
            worker_id (int): 当前恶意客户端的唯一编号。
            train_dataset (Dataset): 原始本地训练集。
            test_dataset (Dataset): 原始本地测试集。

        返回:
            None。

        异常:
            AttributeError: 当 `args` 缺少必需字段或父类初始化失败时自动抛出。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)（不含合成器构建与数据加载成本）。

        费曼学习法:
            (A) 该函数为攻击者准备运行所需的所有“装备”和数据流水线。
            (B) 类比一位厨师要做特殊料理：先登记厨房权限，再配好独门酱料，最后安排烹饪顺序。
            (C) 步骤拆解:
                1. 调用 `Client.__init__`，继承联邦通信与训练接口。
                2. 设置 `default_attack_params`，提供投毒比例、PGD 半径与模型缩放等默认策略。
                3. 调用 `update_and_set_attr`，将默认值与外部参数融合为实例属性。
                4. 运行 `define_synthesizer`，生成投毒数据集并缓存。
                5. 调用 `get_dataloader`，以投毒模式获取训练数据迭代器。
                6. 指定 `algorithm`，便于后续缩放逻辑与聚合策略保持一致。
            (D) 示例:
                >>> attacker = EdgeCase(args, worker_id=0, train_dataset=train_ds, test_dataset=test_ds)
                >>> attacker.poisoning_ratio
                0.8
            (E) 边界条件与测试建议: 若边缘数据集缺失或 target_label 无样本需额外处理；
                建议单元测试验证 1) 默认初始化是否成功；2) 修改 `poisoning_ratio` 后 `poisoned_set` 中边缘样本比例是否变化。
            (F) 背景参考: 《Attack of the Tails》原文、联邦学习后门攻击综述。
        """
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        """
        poisoning_ratio: ratio of edge data in the training dataset
        epsilon: Radius the l2 norm ball in PGD attack. For PGD with replacement, 0.25 for mnist, 0.083 for cifar10, coming from the paper
        projection_type: l_2 or l_inf
        l2_proj_frequency: projection frequency
        """
        # 默认投毒与 PGD/缩放参数，均可通过配置覆盖。
        self.default_attack_params = {
            "poisoning_ratio": 0.8,
            "epsilon": 0.25,
            "PGD_attack": True,
            "projection_type": "l_2",
            "l2_proj_frequency": 1,
            "scaling_attack": True,
            "scaling_factor": 50,
            "target_label": 1,
        }
        # 将默认值与传入配置整合为实例属性。
        self.update_and_set_attr()

        # 构建外部边缘样本合成器，生成投毒数据资源。
        self.define_synthesizer()
        # 获取训练数据迭代器，默认处于投毒模式（poison_epochs=True）。
        self.train_loader = self.get_dataloader(
            train_dataset, train_flag=True, poison_epochs=True)
        # 指定当前假定的服务端优化算法，用于缩放策略分支。
        self.algorithm = "FedOpt"

    def define_synthesizer(self):
        """创建边缘样本合成器并缓存投毒训练/测试数据集。

        概述:
            将原始训练集与 `EdgeDataset` 中的稀有样本按比例混合，生成后门训练集与后门测试集。

        参数:
            无。

        返回:
            None。

        异常:
            ValueError: 当 `poisoning_ratio` 不在 [0, 1] 范围等合成器预期外情况。

        复杂度:
            时间复杂度 O(n)，n 为混合后的样本总数；空间复杂度 O(n)。

        费曼学习法:
            (A) 该方法负责把原始数据与边缘样本“搅拌”成可用于攻击的训练/测试集。
            (B) 类比把普通汤底与浓缩酱汁混合，形成具有特殊味道的基底。
            (C) 步骤拆解:
                1. 利用目标标签实例化 `EdgeDataset`，获取边缘样本。
                2. 创建 `DatasetSynthesizer`，按投毒比例混合原始训练集与边缘样本。
                3. 调用 `get_poisoned_set(train=True/False)`，分别获得训练与测试用的投毒数据。
                4. 将两份数据集缓存到 `self.poisoned_set`，方便后续快速切换。
            (D) 示例:
                >>> attacker.define_synthesizer()
                >>> len(attacker.poisoned_set[0][0])
                # 返回投毒训练集中样本数量
            (E) 边界条件与测试建议: 确保 EdgeDataset 已正确下载/生成；
                建议测试投毒训练集含有目标标签边缘样本，测试集是否完全由边缘样本构成。
            (F) 背景参考: EdgeDataset 构建方法、数据投毒混合策略相关章节。
        """
        self.synthesizer = DatasetSynthesizer(
            self.args,
            self.train_dataset,
            EdgeDataset(self.args, self.target_label),
            self.poisoning_ratio,
        )
        # 缓存投毒训练集与投毒测试集，后续根据 train_flag 切换。
        self.poisoned_set = (
            self.synthesizer.get_poisoned_set(train=True),
            self.synthesizer.get_poisoned_set(train=False),
        )

    def get_dataloader(self, dataset, train_flag, poison_epochs=None):
        """返回结合投毒标记的数据迭代器，训练模式支持无限循环。

        概述:
            当 `poison_epochs` 为 True 时使用缓存的投毒数据集，否则回退至原始数据集；训练阶段返回无限迭代器，测试阶段遍历一次即停。

        参数:
            dataset (Dataset): 默认数据集，当不启用投毒时直接使用。
            train_flag (bool): 是否为训练模式；训练模式下返回无限生成器。
            poison_epochs (Optional[bool]): 当前轮次是否启用投毒，缺省时视为 False。

        返回:
            Iterator[Tuple[Tensor, Tensor]]: 批量的图像与标签迭代器。

        异常:
            ValueError: 当请求投毒但 `poisoned_set` 尚未初始化时可能触发。

        复杂度:
            时间复杂度 O(len(data)) 每次完整遍历，空间复杂度 O(batch_size)。

        费曼学习法:
            (A) 函数决定在给定阶段端出“带毒菜品”还是“清淡菜品”，并控制端菜方式。
            (B) 类比服务员按照档期安排特别菜单：特定时段推出加料菜，其余时间提供常规菜。
            (C) 步骤拆解:
                1. 若 `poison_epochs` 为 None，则默认不启用投毒。
                2. 根据 `poison_epochs` 选择投毒数据或原始数据。
                3. 构建 `DataLoader`，训练模式启用打乱与多线程读取。
                4. 对训练模式使用 `while True` 形成无限迭代器，匹配外层 epoch 控制。
                5. 测试模式仅遍历一轮数据后退出循环。
            (D) 示例:
                >>> loader = attacker.get_dataloader(train_ds, train_flag=True, poison_epochs=True)
                >>> images, labels = next(loader)
            (E) 边界条件与测试建议: 确保 `poisoned_set` 已准备完成；
                建议测试 1) 训练模式可重复迭代；2) 测试模式仅输出一轮数据并停止。
            (F) 背景参考: PyTorch `DataLoader` 使用方法、联邦学习本地训练循环设计。
        """
        # EdgeCase attack is this kind of attack using external prepared backdoor dataset
        poison_epochs = False if poison_epochs is None else poison_epochs
        # 当启用投毒时，按训练/测试标志选择对应的投毒数据集。
        data = self.poisoned_set[1 - train_flag] if poison_epochs else dataset
        # 构建数据加载器，训练阶段随机打乱，测试阶段保持顺序。
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=self.args.batch_size,
            shuffle=train_flag,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        while True:  # train mode for infinite loop with training epoch as the outer
            for images, targets in dataloader:
                yield images, targets
            if not train_flag:
                # test mode for test dataset
                break  # 测试阶段仅遍历一次，避免无限循环。

    def step(self, optimizer, **kwargs):
        """在标准优化步后执行 PGD 投影，限制参数偏移至 ε 球内。

        概述:
            调用父类完成常规梯度更新，再按 L∞ 或 L2 约束将模型参数投影回指定范围，最后回写模型。

        参数:
            optimizer (torch.optim.Optimizer): 当前使用的优化器实例。
            **kwargs: 需包含 `cur_local_epoch`（int）表示本地轮次索引。

        返回:
            None。

        异常:
            KeyError: 当 `kwargs` 中缺少 `cur_local_epoch` 字段时触发。

        复杂度:
            时间复杂度 O(d)，d 为参数向量维度；空间复杂度 O(d) 用于暂存模型向量。

        费曼学习法:
            (A) 该函数保证模型在更新后不会离全局模型太远，从而隐蔽地携带后门。
            (B) 类比攀岩时系安全绳：每次前进都要检测绳长，超过限制就把人拉回。
            (C) 步骤拆解:
                1. 记录当前本地轮次索引，供 L2 投影按频率触发。
                2. 调用 `super().step(optimizer)` 执行常规梯度更新。
                3. 使用 `model2vec` 将模型参数向量化，计算与全局权重的差值 `w_diff`。
                4. 若为 L∞ 投影，逐元素裁剪差值至 [-ε, ε] 范围。
                5. 若为 L2 投影，在设定频率或最后一轮检查差值范数，超出半径则归一化缩放。
                6. 使用 `vec2model` 将投影后的参数重新写入模型。
            (D) 示例:
                >>> attacker.step(optimizer, cur_local_epoch=0)
            (E) 边界条件与测试建议: 应确保 `epsilon` > 0 且 `PGD_attack` 为 True；
                建议测试 1) L∞ 模式参数是否被正确裁剪；2) L2 模式在指定频率执行缩放。
            (F) 背景参考: Projected Gradient Descent、模型替换攻击相关章节。
        """
        # PGD after step at each local epoch
        # normal step
        # 当前本地轮次索引用于决定 L2 投影的触发频率。
        cur_local_epoch = kwargs["cur_local_epoch"]
        super().step(optimizer)

        # get the updated model
        # 将更新后的模型转换为向量形式，便于计算偏移。
        model_update = model2vec(self.model)
        w_diff = model_update - self.global_weights_vec

        # PGD projection
        if self.projection_type == "l_inf":
            # L∞ 投影：逐元素截断参数偏移，防止超出 ε 范围。
            smaller_idx = np.less(w_diff, -self.epsilon)
            larger_idx = np.greater(w_diff, self.epsilon)
            model_update[smaller_idx] = self.global_weights_vec[smaller_idx] - self.epsilon
            model_update[larger_idx] = self.global_weights_vec[larger_idx] + self.epsilon
        elif self.projection_type == "l_2":
            # L2 投影：周期性检查范数，必要时按比例缩放到 ε 球面上。
            w_diff_norm = np.linalg.norm(w_diff)
            if (cur_local_epoch % self.l2_proj_frequency == 0 or
                    cur_local_epoch == self.local_epochs - 1) and w_diff_norm > self.epsilon:
                model_update = self.global_weights_vec + self.epsilon * w_diff / w_diff_norm

        # load the model_update to the model after PGD projection
        # 将投影后的参数写回模型，保证更新受限于 ε 范围。
        vec2model(model_update, self.model)

    def non_omniscient(self):
        """执行缩放式模型替换，将本地更新按需放大后提交。

        概述:
            当 `scaling_attack` 启用时，对本地更新应用缩放因子；对 FedAvg 采用权重偏移放大，其他算法直接缩放更新向量。

        参数:
            无。

        返回:
            numpy.ndarray 或 Tensor: 经缩放处理的本地更新。

        异常:
            AttributeError: 当实例缺少 `update` 或 `global_weights_vec` 属性时触发。

        复杂度:
            时间复杂度 O(d)，d 为模型参数维度；空间复杂度 O(1)。

        费曼学习法:
            (A) 函数决定是否把攻击者的更新“调大音量”再提交给服务器。
            (B) 类比歌手在合唱中暗自调高麦克风，关键时刻让自家声部盖过他人。
            (C) 步骤拆解:
                1. 判断 `scaling_attack` 开关，决定是否执行放大。
                2. 若为 FedAvg，则在全局权重基础上放大偏移量，保持向量语义为权重。
                3. 若为其他算法，直接对更新向量乘以 `scaling_factor`。
                4. 若未开启缩放，则返回原始更新以降低可检测性。
            (D) 示例:
                >>> scaled_update = attacker.non_omniscient()
            (E) 边界条件与测试建议: 需确保 `poison_epochs` 设置匹配投毒周期、`scaling_factor` 合理；
                建议测试 1) 缩放开启时更新幅度显著增大；2) 关闭缩放后与原始更新一致。
            (F) 背景参考: 模型替换攻击（Model Replacement Attack）、联邦聚合策略原理。
        """
        # scaling attack (model replacement attacks)
        # non_omniscient function is after the get_local_update function
        if self.scaling_attack:
            scaled_update = self.global_weights_vec + self.scaling_factor * (
                self.update - self.global_weights_vec
            ) if self.args.algorithm == "FedAvg" else self.scaling_factor * self.update
        else:
            scaled_update = self.update
        return scaled_update


# __AI_ANNOTATION_SUMMARY__
# 类 EdgeCase: 结合边缘样本、PGD 投影与模型缩放的联邦后门攻击器。
# 方法 __init__: 初始化攻击参数与数据管线，设定默认聚合算法。
# 方法 define_synthesizer: 构建并缓存投毒训练/测试数据集。
# 方法 get_dataloader: 按投毒标志返回训练或测试迭代器，训练模式无限循环。
# 方法 step: 在标准更新后执行 PGD 投影以限制参数偏移。
# 方法 non_omniscient: 基于缩放因子放大或保留本地更新，实现模型替换攻击。
