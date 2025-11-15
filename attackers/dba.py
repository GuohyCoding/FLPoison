# -*- coding: utf-8 -*-

import torch

from attackers import attacker_registry
from attackers.pbases.dpbase import DPBase
from fl.client import Client
from global_utils import actor

from .synthesizers import PixelSynthesizer


@attacker_registry
@actor('attacker', 'data_poisoning')
class DBA(DPBase, Client):
    """DBA（Distributed Backdoor Attack）分布式后门攻击器，为每个客户端部署局部触发器并在推理阶段合成全局触发器。

    该实现基于《DBA: Distributed Backdoor Attacks Against Federated Learning》提出的策略，
    通过在训练阶段轮流植入局部触发器，使聚合模型在推理阶段面对组合触发器时输出攻击者设定的目标标签。

    属性:
        default_attack_params (dict): 攻击默认参数集合，涵盖触发器尺寸、缩放因子、目标标签等。
        trigger (Tensor): 合成器内部使用的触发器张量，包含各局部触发器。
        trigger_nums (int): 局部触发器数量，默认 4。
        train_loader (DataLoader): 依据投毒日程构建的本地训练数据加载器。
        algorithm (str): 假定的服务器优化算法名称，默认 FedOpt。
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        """初始化 DBA 攻击客户端并配置分布式后门所需资源。

        概述:
            调用基类初始化联邦客户端上下文，写入默认攻击参数，搭建触发器合成器，并设置投毒轮次与训练数据加载器。

        参数:
            args (argparse.Namespace): 运行配置，需包含 `epochs`、`algorithm`、`batch_size` 等字段。
            worker_id (int): 当前客户端编号。
            train_dataset (Dataset): 客户端本地训练数据集。
            test_dataset (Dataset): 客户端本地测试数据集。

        返回:
            None。

        异常:
            AttributeError: 当 `args` 缺失必要字段或父类初始化失败时触发。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)（不含数据加载器构建成本）。

        费曼学习法:
            (A) 此方法为 DBA 攻击者准备默认设置、局部触发器及投毒日程。
            (B) 好比一支合唱队每人负责一句暗号：排练时各唱自己的部分，演出时合成完整暗号。
            (C) 步骤拆解:
                1. 调用 `Client.__init__` 继承数据与通信接口，保证框架兼容。
                2. 定义 `default_attack_params`，提供触发器因子、缩放倍数、投毒节奏等默认设定。
                3. 调用 `update_and_set_attr` 合并外部配置，形成实例属性。
                4. 标记 `algorithm`，便于与聚合策略保持一致。
                5. 调用 `define_synthesizer` 构建局部触发器及分布式注入逻辑。
                6. 通过 `generate_poison_epochs` 计算需要投毒的训练轮次。
                7. 使用 `get_dataloader` 创建带投毒调度的训练加载器。
            (D) 示例:
                >>> attacker = DBA(args, worker_id=2, train_dataset=train_ds, test_dataset=test_ds)
                >>> attacker.scaling_factor
                100
            (E) 边界条件与测试建议: 若 `trigger_factor` 维度不为 3 或取值不合理，将导致触发器构建失败；
                建议测试 1) 默认初始化通过；2) 修改 `attack_strategy` 时 `poison_epochs` 是否正确更新。
            (F) 背景参考: 《DBA: Distributed Backdoor Attacks Against Federated Learning》、联邦学习后门攻击综述。
        """
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {"attack_model": "all2one", "scaling_factor": 100, "trigger_factor": [
            8, 2, 0], "poisoning_ratio": 0.32, "source_label": 2, "target_label": 7, "attack_strategy": "continuous", "single_epoch": 0, "poison_frequency": 5, "attack_start_epoch": None}
        # 合并默认参数与外部传参，生成可直接访问的实例属性。
        self.update_and_set_attr()
        # 指定当前假定的服务器端优化算法，供上层模块使用。
        self.algorithm = "FedOpt"
        # 构建局部触发器合成器，并覆盖其注入逻辑以实现分布式后门。
        self.define_synthesizer()
        # 基于攻击策略生成投毒轮次列表，确保仅在指定轮次加毒。
        poison_epochs = self.generate_poison_epochs(
            self.attack_strategy, self.args.epochs, self.single_epoch, self.poison_frequency, self.attack_start_epoch)
        # 构造带投毒控制的训练数据加载器。
        self.train_loader = self.get_dataloader(
            train_dataset, train_flag=True, poison_epochs=poison_epochs)

        # self.num_benign_epoch, self.num_poison_epoch = 1, 10
        # self.benign_lr, self.poison_lr = 0.1, 0.05
        # self.benign_optimizer, self.optimizer = self.get_optimizer_scheduler(
        #     self.benign_lr)[0], self.get_optimizer_scheduler(self.poison_lr)[0]

    def define_synthesizer(self):
        """定义 DBA 分布式触发器合成器，并重写后门注入逻辑。

        概述:
            构建若干局部触发器张量，初始化像素合成器，并覆写其 `implant_backdoor` 方法以支持分布式拼接触发器。

        参数:
            无。

        返回:
            None: 通过副作用设置触发器与合成器属性。

        异常:
            ValueError: 当 `trigger_factor` 提供的尺寸参数非法时可能触发。

        复杂度:
            时间复杂度 O(k)，其中 k 为局部触发器数量；空间复杂度 O(k * w)，w 为触发器宽度。

        费曼学习法:
            (A) 该函数为每个客户端准备局部触发器，并告诉合成器如何把它们拼成全局后门。
            (B) 类比把一幅拼图分给四名队员各自携带，最后合起来呈现完整图案。
            (C) 步骤拆解:
                1. 解析 `trigger_factor`，确定局部触发器尺寸、间距与位移。
                2. 创建 `trigger_nums` 个局部触发器张量，初始为全 1 补丁。
                3. 构建 `PixelSynthesizer`，注入目标标签、投毒比例等参数。
                4. 读取合成器内部实际触发器（可能经过归一化等处理）以保持一致性。
                5. 保存原始 `implant_backdoor` 方法，供局部触发器调用。
                6. 将合成器的 `implant_backdoor` 覆写为分布式版本，使其在训练/推理阶段植入不同组合。
            (D) 示例:
                >>> attacker.define_synthesizer()
                >>> attacker.trigger_nums
                4
            (E) 边界条件与测试建议: 需确保 `trigger_factor` 内容满足 [size, gap, shift] 结构；
                建议测试 1) 构造后 `trigger` 形状正确；2) 覆写后的 `implant_backdoor` 可在训练/推理两种分支运行。
            (F) 背景参考: 可查阅像素级后门触发器设计、DBA 论文中触发器布置章节。
        """
        # 构建四个局部触发器，最终全局触发器由它们拼接组成。
        self.trigger_nums = 4
        # trigger_factor = [size, gap, shift]，用于控制触发器形状与放置偏移。
        self.trigger_size, self.gap, self.shift = self.trigger_factor
        # 初始化局部触发器张量，默认填充为亮色像素。
        self.trigger = torch.ones(
            (self.trigger_nums, 1, 1, self.trigger_size))
        # 初始化像素合成器，负责在图像上植入局部触发器并调整标签。
        self.synthesizer = PixelSynthesizer(
            self.args, self.trigger, attack_model=self.attack_model, target_label=self.target_label, poisoning_ratio=self.poisoning_ratio, source_label=self.source_label, single_epoch=self.single_epoch)
        # get synthesizer-transformed trigger for subsequently overwriting implant_distributed_backdoor
        # 合成器可能对触发器做归一化处理，因此取回最终版本以确保一致。
        self.trigger = self.synthesizer.trigger
        # implant the distributed backdoor via overwriting the synthesizer's implant_backdoor method
        # 保存原始的单触发器植入函数，便于复用。
        self.implant_single_backdoor = self.synthesizer.implant_backdoor
        # 覆写合成器的植入函数，实现训练阶段单触发器、推理阶段多触发器的逻辑。
        self.synthesizer.implant_backdoor = self.implant_distributed_backdoor

    def implant_distributed_backdoor(self, image, label, **kwargs):
        """在图像中植入分布式后门，训练阶段单触发器、推理阶段多触发器。

        概述:
            根据 `train` 标志判断当前阶段：若为训练则选择对应客户端的局部触发器，
            若为推理则依次嵌入全部局部触发器，模拟全局触发器效果。

        参数:
            image (Tensor): 输入图像张量。
            label (Tensor): 对应标签张量。
            **kwargs: 需包含 `train` (bool) 表示是否为训练阶段，`worker_id` (int) 表示当前客户端编号。

        返回:
            Tuple[Tensor, Tensor]: 经过后门植入的图像与标签。

        异常:
            KeyError: 当 `kwargs` 缺少 `train` 或 `worker_id` 时触发。

        复杂度:
            时间复杂度 O(k)，其中 k 为插入的触发器数量；空间复杂度 O(1)（复用输入张量）。

        费曼学习法:
            (A) 该函数决定在不同阶段该植入哪个触发器组合。
            (B) 类比演出彩排：排练时每位演员只穿自己的道具，正式演出时所有道具一起登场。
            (C) 步骤拆解:
                1. 从 `kwargs` 中读取 `train` 与 `worker_id`，判断阶段及当前客户端身份。
                2. 若处于训练阶段，计算该客户端对应的触发器索引，确保人人只植入自己的局部触发器。
                3. 调用原始单触发器植入方法，将选定触发器贴到图像上并修改标签。
                4. 若处于推理或服务器侧阶段，遍历全部局部触发器依次植入，组合成全局触发器。
                5. 返回修改后的图像与标签。
            (D) 示例:
                >>> poisoned_img, poisoned_label = attacker.implant_distributed_backdoor(img, lbl, train=True, worker_id=1)
            (E) 边界条件与测试建议: 需保证 `worker_id` 小于客户端数量，否则触发器选择可能重复；
                建议测试 1) 训练分支仅植入一个触发器；2) 推理分支植入完四个触发器后标签是否更新。
            (F) 背景参考: 建议阅读 DBA 论文中关于分布式触发器调度的章节以及联邦学习数据流机制。
        """
        train, worker_id = kwargs['train'], kwargs['worker_id']
        # 2. implement the backdoor logic
        if train:
            # implant one of the four local trigger at the time of local training
            # 计算当前客户端对应的触发器索引，实现循环分配。
            trigger_idx = worker_id % self.trigger_nums
            # 根据触发器索引设置其在图像中的位置。
            self.setup_trigger_position(trigger_idx)
            # 调用原始植入函数，仅嵌入所选局部触发器。
            image, label = self.implant_single_backdoor(
                image, label, trigger=self.trigger[trigger_idx])
        else:
            # for global backdoor in server, embed all four local triggers to the image
            # 推理阶段需要嵌入所有局部触发器，组合成全局后门。
            for trigger_idx in range(self.trigger_nums):
                self.setup_trigger_position(trigger_idx)
                image, label = self.implant_single_backdoor(
                    image, label, trigger=self.trigger[trigger_idx])
            #     print(self.trigger_position)
            # print(image)

        return image, label

    def setup_trigger_position(self, trigger_idx):
        """根据触发器索引计算其在图像上的放置位置。

        概述:
            结合触发器尺寸、间距与偏移量，确定局部触发器在图像中的起始行列坐标，从而避免相互重叠。

        参数:
            trigger_idx (int): 当前触发器的索引，范围 [0, trigger_nums)。

        返回:
            None: 将位置记录在 `self.trigger_position`，供合成器使用。

        异常:
            ValueError: 当索引越界或 `trigger` 尚未初始化时可能触发。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)。

        费曼学习法:
            (A) 该函数决定某个局部触发器应该贴在图像的哪个角落。
            (B) 类似把四张贴纸贴在海报的四个象限，需要根据编号计算位置。
            (C) 步骤拆解:
                1. 根据论文设置默认间距与偏移（MNIST 使用 4,2,0，CIFAR10 使用 6,3,0，此处默认 MNIST）。
                2. 获取触发器宽度，为后续计算列偏移提供基础。
                3. 利用整除与取模，将编号映射到二维网格，计算行起点与列起点。
                4. 将结果写入 `self.trigger_position`，供合成器在植入时定位。
            (D) 示例:
                >>> attacker.setup_trigger_position(2)
                >>> attacker.trigger_position
                (3, 0)  # 默认 gap=2、shift=0 且触发器位于下左象限
            (E) 边界条件与测试建议: 若图像尺寸不足以容纳触发器与间距，需要提前检查；
                建议测试 1) 四个索引的位置信息互不重叠；2) 修改间距后位置是否更新。
            (F) 背景参考: 可查阅 DBA 论文中触发器布局参数设置章节。
        """
        # 论文推荐参数：MNIST 使用 gap=2, shift=0；CIFAR10 使用 gap=3, shift=0，此处默认 MNIST 配置。
        self.gap = 2
        self.shift = 0
        # 触发器宽度用于确定列偏移。
        width = self.trigger.shape[-1]
        # 整除用于区分上/下行，乘以 (1+gap) 确保触发器之间有间隔。
        row_starter = (trigger_idx // 2) * (1+self.gap) + self.shift
        # 取模区分左右列，再加上触发器宽度与间距形成水平偏移。
        column_starter = (trigger_idx % 2) * (width + self.gap) + self.shift
        self.trigger_position = (row_starter, column_starter)

    # def local_training(self, model=None, train_loader=None, optimizer=None, criterion_fn=None, local_epochs=None):
    #     if self.global_epoch in self.poison_epochs:
    #         # benign training
    #         self.train_loader = self.get_dataloader(
    #             self.train_dataset, train_flag=True, poison_epochs=False)
    #         super().local_training(model, train_loader,
    #                                 self.benign_optimizer, criterion_fn, self.num_benign_epoch)
    #         # poison training
    #         self.train_loader = self.get_dataloader(
    #             self.train_dataset, train_flag=True, poison_epochs=True)
    #         super().local_training(model, train_loader,
    #                                    self.optimizer, criterion_fn, self.num_poison_epoch)
    #     else:
    #         return super().local_training()

    def non_omniscient(self):
        """在非全知攻击假设下调整本地更新幅度，实现缩放型模型投毒。

        概述:
            当当前全局轮次处于投毒阶段时，对本地更新施加 `scaling_factor` 放大；
            否则返回常规更新，以保持隐蔽性。

        参数:
            无。

        返回:
            Tensor: 缩放后的更新向量。

        异常:
            AttributeError: 若实例缺少 `update` 或 `global_weights_vec` 属性时触发。

        复杂度:
            时间复杂度 O(d)，d 为参数向量维度；空间复杂度 O(1)。

        费曼学习法:
            (A) 该函数决定在特定轮次将更新放大，以增强攻击影响力。
            (B) 类似在关键比赛时把话筒音量调大，平时保持正常音量以免被发现。
            (C) 步骤拆解:
                1. 检查当前全局轮次是否在 `poison_epochs` 中，确定是否需要放大。
                2. 若需要放大：对 FedAvg 使用偏移缩放（全局权重 + 缩放因子 * 差值），其它算法直接缩放更新向量。
                3. 若不需要放大：直接返回原始更新。
                4. 输出处理后的更新供服务端聚合。
            (D) 示例:
                >>> scaled = attacker.non_omniscient()
            (E) 边界条件与测试建议: 确保 `poison_epochs` 与 `scaling_factor` 已配置；
                建议测试 1) 在投毒轮次放大有效；2) 非投毒轮次返回原始更新。
            (F) 背景参考: 可阅读模型投毒攻击中的缩放策略、FedAvg 聚合原理。
        """
        if self.global_epoch in self.poison_epochs:
            # scale
            # FedAvg 需要在全局权重基础上放大偏移，保障提交的是权重向量而非梯度。
            scaled_update = self.global_weights_vec + self.scaling_factor * \
                (self.update - self.global_weights_vec) if self.args.algorithm == "FedAvg" else self.scaling_factor * self.update
        else:
            # 非投毒轮次保持原状，降低可检测性。
            scaled_update = self.update
        return scaled_update


# __AI_ANNOTATION_SUMMARY__
# 类 DBA: 分布式后门攻击器，实现局部触发器拼接与更新缩放。
# 方法 __init__: 初始化 DBA 客户端并配置默认攻击参数与投毒调度。
# 方法 define_synthesizer: 构造局部触发器并覆写植入逻辑以支持分布式后门。
# 方法 implant_distributed_backdoor: 区分训练/推理阶段植入局部或全局触发器。
# 方法 setup_trigger_position: 依据索引计算局部触发器的位置坐标。
# 方法 non_omniscient: 在投毒轮次放大本地更新，增强模型投毒效果。
