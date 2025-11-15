# -*- coding: utf-8 -*-

import torch
from fl.client import Client
from attackers.pbases.mpbase import MPBase
from attackers.pbases.dpbase import DPBase
from global_utils import actor
from fl.models.model_utils import model2vec
from attackers import attacker_registry
from sklearn.metrics.pairwise import cosine_distances
from attackers.synthesizers.pixel_synthesizer import PixelSynthesizer


@attacker_registry
@actor('attacker', 'data_poisoning', 'model_poisoning', 'non_omniscient')
class ModelReplacement(MPBase, DPBase, Client):
    """模型替换攻击器：结合像素触发器与更新缩放，实现经典的“约束-缩放”式后门攻击。

    该实现参考 AISTATS 2020 论文《How to Backdoor Federated Learning》，
    通过在本地损失中加入异常检测项以限制更新形状，并在提交阶段对更新进行大幅缩放以替换全局模型。

    属性:
        default_attack_params (dict): 默认攻击参数，包含缩放因子、损失权重及标签映射等。
        synthesizer (PixelSynthesizer): 像素级触发器生成器。
        train_loader (Iterator): 根据投毒调度返回的训练数据加载器。
        algorithm (str): 默认假定的服务器端优化算法，设置为 `FedOpt`。
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        """初始化模型替换攻击者，配置触发器与投毒调度。

        概述:
            设定缩放比例与损失权重，构建像素触发器合成器，并将投毒调度映射到本地训练数据加载器。

        参数:
            args (argparse.Namespace): 运行配置，需包含 `epochs`、`batch_size`、`num_workers`、`device` 等字段。
            worker_id (int): 当前恶意客户端编号。
            train_dataset (Dataset): 本地训练数据集，用于构造投毒数据。
            test_dataset (Dataset): 本地测试数据集（保持接口一致）。

        返回:
            None。

        异常:
            AttributeError: 当 `args` 缺失必要字段时由基类自动抛出。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)（不含数据加载器构建）。

        费曼学习法:
            (A) 该函数准备模型替换攻击所需的参数、触发器与数据调度。
            (B) 类比一位厨师先配好酱料（默认参数），布置暗号（触发器），再安排何时上菜（投毒轮次）。
            (C) 步骤拆解:
                1. 调用 `Client.__init__` 完成联邦客户端的基础初始化。
                2. 写入默认攻击参数（缩放因子、标签映射、投毒频率等）。
                3. 通过 `update_and_set_attr` 合并外部配置。
                4. 调用 `define_synthesizer` 创建像素触发器合成器。
                5. 计算投毒轮次并生成带投毒调度的训练数据加载器。
                6. 记录默认聚合算法名称，便于后续缩放逻辑使用。
            (D) 示例:
                >>> attacker = ModelReplacement(args, worker_id=0, train_dataset=train, test_dataset=test)
                >>> attacker.scaling_factor
                50
            (E) 边界条件与测试建议: 确保 `scaling_factor` 与 `alpha` 合理；可测试
                1) 默认参数能否被外部覆盖；2) 投毒调度是否按 `poison_frequency` 触发。
            (F) 背景参考: 模型替换攻击、联邦学习后门综述。
        """
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        """
        scaling_factor: estimated scaling factor, num_clients / global_lr, 50/1=50 in our setting
        alpha: the weight of the classification loss in the total loss
        """
        self.default_attack_params = {
            'scaling_factor': 50,
            "alpha": 0.5,
            "attack_model": "all2one",
            "poisoning_ratio": 0.32,
            "target_label": 6,
            "source_label": 3,
            "attack_strategy": "continuous",
            "single_epoch": 0,
            "poison_frequency": 5,
        }
        # 合并默认攻击参数与外部配置。
        self.update_and_set_attr()

        # 初始化像素触发器合成器并生成投毒数据集。
        self.define_synthesizer()
        poison_epochs = self.generate_poison_epochs(
            self.attack_strategy, self.args.epochs, self.single_epoch, self.poison_frequency)
        # 构建根据投毒调度返回的训练数据加载器。
        self.train_loader = self.get_dataloader(
            train_dataset, train_flag=True, poison_epochs=poison_epochs)
        # 指明当前假定的聚合算法，供缩放逻辑参考。
        self.algorithm = "FedOpt"

    def define_synthesizer(self):
        """创建像素触发器合成器，用于生成投毒样本。

        概述:
            构造固定大小的触发器补丁，并实例化 `PixelSynthesizer` 以插入触发器、翻转标签。

        参数:
            无。

        返回:
            None。

        异常:
            ValueError: 当触发器尺寸或标签配置非法时。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)。

        费曼学习法:
            (A) 该方法准备一个可反复使用的像素触发器和对应的投毒逻辑。
            (B) 类比制作隐形印章，并让助手按计划在作业角落盖章。
            (C) 步骤拆解:
                1. 创建尺寸为 5×5 的全 1 触发器张量，作为亮色补丁。
                2. 实例化 `PixelSynthesizer`，指定攻击模式、标签与投毒比例。
                3. 将合成器保存为实例属性，供训练流程调用。
            (D) 示例:
                >>> attacker.define_synthesizer()
                >>> attacker.trigger.shape
                torch.Size([1, 5, 5])
            (E) 边界条件与测试建议: 确保触发器大小不超过输入图像尺寸；
                建议测试 1) 合成器是否正确贴入触发器；2) 标签是否按配置翻转。
            (F) 背景参考: 像素触发器设计、后门数据合成。
        """
        # for pixel-type trigger, specify the trigger tensor
        self.trigger = torch.ones((1, 5, 5))
        self.synthesizer = PixelSynthesizer(
            self.args, self.trigger,
            attack_model=self.attack_model,
            target_label=self.target_label,
            poisoning_ratio=self.poisoning_ratio,
            source_label=self.source_label,
            single_epoch=self.single_epoch,
        )

    def criterion_fn(self, y_pred, y_true, **kwargs):
        """带异常检测项的本地损失函数，抑制过于突兀的模型更新。

        概述:
            在交叉熵损失基础上加入模型向量与全局模型的余弦距离，
            通过权重 `alpha` 控制分类损失与异常损失的平衡。

        参数:
            y_pred (Tensor): 模型输出的 logits，形状为 (batch_size, num_classes)。
            y_true (Tensor): 样本真实标签。
            **kwargs: 预留扩展参数，当前未使用。

        返回:
            Tensor: 融合分类与异常检测项的标量损失。

        异常:
            RuntimeError: 当输入张量维度不匹配或存在 NaN 时由 PyTorch 抛出。

        复杂度:
            时间复杂度 O(d)，d 为模型参数维度；空间复杂度 O(1)。

        费曼学习法:
            (A) 函数在普通分类损失外，再加一项“别离全队太远”的惩罚。
            (B) 类比行军时既要保持正确方向（分类准确），又要靠近队伍中心（小距离）。
            (C) 步骤拆解:
                1. 使用 `model2vec` 展平本地模型参数。
                2. 计算与全局模型向量之间的余弦距离，作为异常项。
                3. 计算交叉熵损失作为分类项。
                4. 使用 `alpha` 加权组合两部分损失。
            (D) 示例:
                >>> loss = attacker.criterion_fn(outputs, labels)
            (E) 边界条件与测试建议: 确保 `alpha` 在 [0, 1]；测试 1) 模型未更新时异常项近 0；2) 调整 `alpha` 的影响。
            (F) 背景参考: 余弦距离、模型替换攻击的约束策略。
        """
        # constrain: cosine distance between model2vec(self.model) and self.global_weights_vec
        cosine_dist = cosine_distances(
            model2vec(self.model).reshape(1, -1),
            self.global_weights_vec.reshape(1, -1),
        )
        classification_loss = torch.nn.CrossEntropyLoss()(y_pred, y_true)
        # 组合分类损失与异常检测项，提升隐蔽性。
        return self.alpha * classification_loss + (1 - self.alpha) * torch.from_numpy(cosine_dist).to(self.args.device)

    def non_omniscient(self):
        """非全知场景下的缩放式更新提交，实现模型替换效果。

        概述:
            在提交更新时，将本地更新向量 `(X - G^t)` 按 `scaling_factor` 放大；
            对于 FedAvg，需在全局权重基础上操作，保持形状一致。

        参数:
            无。

        返回:
            numpy.ndarray 或 Tensor: 缩放后的恶意更新。

        异常:
            AttributeError: 当缺少 `update` 或 `global_weights_vec` 属性时。

        复杂度:
            时间复杂度 O(d)，d 为参数维度；空间复杂度 O(1)。

        费曼学习法:
            (A) 函数把看似正常的更新放大到足以覆盖全局模型，从而实现替换。
            (B) 类比在合唱中把麦克风音量调到最大，压过其他声音。
            (C) 步骤拆解:
                1. 判断当前聚合算法是否为 FedAvg。
                2. 若是 FedAvg，则在全局权重基础上放大偏移量。
                3. 否则直接对更新向量乘以缩放因子。
                4. 返回缩放后的更新供服务器聚合。
            (D) 示例:
                >>> scaled = attacker.non_omniscient()
            (E) 边界条件与测试建议: 确保 `scaling_factor` 合理；测试
                1) FedAvg 与其他算法路径是否正确；2) 缩放后范数显著增大。
            (F) 背景参考: 模型替换攻击、联邦聚合机制。
        """
        # scale
        # gamma = self.args.num_clients/self.optimizer.param_groups[0]['lr'] # however, adversaries don't know num_clients
        # self.update = X - G^t
        if self.args.algorithm == "FedAvg":
            scaled_update = self.global_weights_vec + self.scaling_factor * (
                self.update - self.global_weights_vec
            )
        else:
            scaled_update = self.scaling_factor * self.update
        return scaled_update


# __AI_ANNOTATION_SUMMARY__
# 类 ModelReplacement: 结合像素触发器与缩放替换的模型/数据双重攻击器。
# 方法 __init__: 初始化默认参数，构建触发器与投毒调度。
# 方法 define_synthesizer: 创建固定像素触发器并实例化合成器。
# 方法 criterion_fn: 融合分类损失与余弦距离的约束损失函数。
# 方法 non_omniscient: 根据聚合算法放大更新以替换全局模型。
