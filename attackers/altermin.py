# -*- coding: utf-8 -*-

import torch.nn.functional as F
import copy
from datapreprocessor.data_utils import Partition, dataset_class_indices, get_transform
import torch
from .labelflipping import LabelFlipping
from global_utils import actor
from fl.models.model_utils import model2vec, model2vec, vec2model
from attackers.pbases.mpbase import MPBase
from attackers.pbases.dpbase import DPBase
from attackers import attacker_registry
from fl.client import Client
import numpy as np


@attacker_registry
@actor('attacker', "data_poisoning", 'model_poisoning')
class AlterMin(MPBase, DPBase, Client):
    """AlterMin（Alternating Minimization）混合投毒攻击器，交替优化潜伏目标与显式目标以规避防御并提升攻击成功率。

    该实现基于 ICML 2019 论文《Analyzing Federated Learning Through an Adversarial Lens》，
    将数据投毒（标签翻转）与模型投毒（梯度放大）结合，分阶段优化隐蔽损失与目标损失，从而在联邦学习中实现稳健的恶意更新。

    属性:
        synthesizer (Callable): 来源于 LabelFlipping 的数据投毒合成器，用于构造被翻转标签的样本。
        poisoned_loader (DataLoader): 按照配置采样的恶意数据迭代器，用于目标优化阶段。
        algorithm (str): 当前攻击器假定的服务端优化器名称，默认为 FedOpt。
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        """初始化 AlterMin 攻击客户端并构建所需的数据与参数结构。

        概述:
            继承 MPBase、DPBase 与 Client 的能力，设置默认攻击参数，构建数据投毒器并初始化本地数据加载器。

        参数:
            args (argparse.Namespace): 运行时配置，需包含 `batch_size`、`num_workers`、`device`、`num_clients` 等字段。
            worker_id (int): 当前客户端的唯一编号。
            train_dataset (Dataset): 客户端本地原始训练数据集。
            test_dataset (Dataset): 客户端本地评估数据集（亦作为恶意样本采样源）。

        返回:
            None: 构造函数仅进行状态初始化及资源分配。

        异常:
            AttributeError: 当 `args` 缺失必要字段或基类初始化失败时触发。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)（不含数据加载器构建成本）。

        费曼学习法:
            (A) 该方法把 AlterMin 攻击者的默认设置与数据管线全部搭建好。
            (B) 类比一位渗透者先准备伪装衣（默认参数）、再安排内应（数据投毒器），最后把潜伏与攻击路线规划好。
            (C) 步骤拆解:
                1. 调用 Client.__init__ 继承通信与训练接口，确保与框架兼容。
                2. 设定 default_attack_params 提供攻击默认策略，便于后续覆盖。
                3. 调用 update_and_set_attr 将默认值与外部配置融合为实例属性。
                4. 指定 algorithm，标记假定的服务端优化器，方便日志与兼容性处理。
                5. 调用 define_synthesizer 构造标签翻转攻击器，准备恶意样本生成机制。
                6. 调用 init_poisoned_loader 采样并缓存恶意数据用于目标优化。
                7. 构建常规训练数据加载器，支撑隐蔽目标训练过程。
            (D) 示例:
                >>> attacker = AlterMin(args, worker_id=1, train_dataset=train_ds, test_dataset=test_ds)
                >>> attacker.boosting_factor
                10
            (E) 边界条件与测试建议: 若 `poisoned_sample_cnt` 大于可采样的源样本数量将导致抽样异常；可测试
                1) 能否成功初始化并生成 `poisoned_loader`；2) 默认属性是否写入。
            (F) 推荐背景: 建议阅读联邦学习攻击综述以及《Analyzing Federated Learning Through an Adversarial Lens》原文。
        """
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        """
        attack_model: ["sample-targeted", "class-targeted"]
        poisoned_sample_cnt: number of poisoned samples,
        boosting_factor: boosting factor boosting_factor, variable `lambda` in paper
        rho: the weight of distance loss in the loss function
        benign_epochs: Benign training epochs for malicious agent
        malicous_epochs: malicious training epochs
        """
        self.default_attack_params = {"attack_model": "targeted", "source_label": 3, "target_label": 7, "poisoned_sample_cnt": 1,
                                      "boosting_factor": 10, "rho": 1e-4, "benign_epochs": 10, "malicous_epochs": 1}
        self.update_and_set_attr()
        self.algorithm = "FedOpt"
        # initialize the synthesizer for data loader
        self.define_synthesizer()
        self.init_poisoned_loader()
        self.train_loader = self.get_dataloader(
            self.train_dataset, train_flag=True, poison_epochs=False)

    def define_synthesizer(self,):
        """定义 AlterMin 攻击所依赖的标签翻转数据合成器。

        概述:
            基于 LabelFlipping 攻击器实例化并对其源/目标标签进行定制，使 AlterMin 能够生成指定类别的恶意样本。

        参数:
            无。

        返回:
            None: 方法通过副作用为实例属性 `synthesizer` 赋值。

        异常:
            AttributeError: 若实例缺失 `source_label` 或 `target_label` 属性时会触发。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)。

        费曼学习法:
            (A) 该函数负责准备一个能把原始样本标签翻转成攻击目标的“伪装器”。
            (B) 好比给摄影师配一套滤镜，把看到的颜色换成敌人误以为的颜色。
            (C) 步骤拆解:
                1. 实例化 LabelFlipping 攻击器，继承其数据加工能力。
                2. 将 AlterMin 当前配置的源标签与目标标签写入 LabelFlipping，确保投毒方向一致。
                3. 将得到的 `synthesizer` 存入实例，以供后续数据加载器使用。
            (D) 示例:
                >>> attacker.define_synthesizer()
                >>> poisoned = attacker.synthesizer(image, label)
            (E) 边界条件与测试建议: 若标签配置超出数据集类别范围，将导致后续投毒失败；可测试 `synthesizer`
                在单样本上的输出是否更换标签。
            (F) 背景参考: 可查看经典的数据投毒方法，如“Label Flipping Attack”以及《Adversarial Machine Learning》相关章节。
        """
        # define a type of data poisoning attack
        dpa = LabelFlipping(
            self.args, self.worker_id, self.train_dataset, self.test_dataset)
        # note that other parameters are using the default values of LabelFlipping. If you want to change them, you can do it here
        dpa.source_label, dpa.target_label = self.source_label, self.target_label
        self.synthesizer = dpa.synthesizer

    def init_poisoned_loader(self):
        """初始化恶意数据加载器，将部分样本转换为投毒样本。

        概述:
            从指定类别的样本中抽取固定数量，应用投毒合成器后构建 DataLoader，为后续目标优化提供数据源。

        参数:
            无。

        返回:
            None: 通过副作用创建 `poisoned_loader`。

        异常:
            ValueError: 当 `poisoned_sample_cnt` 大于可用样本时，`np.random.choice` 在 `replace=False` 下会抛错。

        复杂度:
            时间复杂度 O(k)，其中 k 为抽样数量；空间复杂度 O(k)。

        费曼学习法:
            (A) 这个函数为攻击者准备一小批“被污染”的训练数据。
            (B) 类似在干净水池中悄悄取几杯水，加入染料后装瓶备用。
            (C) 步骤拆解:
                1. 根据攻击模式确定可选样本索引（针对特定源类或任意类）。
                2. 随机抽取指定数量的样本索引，保证不重复。
                3. 获取训练阶段使用的图像变换，以保持数据前处理一致。
                4. 构造 Partition 子集并调用 `poison_setup` 应用标签翻转。
                5. 使用 DataLoader 封装生成的投毒数据，便于迭代训练。
            (D) 示例:
                >>> attacker.poisoned_sample_cnt = 2
                >>> attacker.init_poisoned_loader()
                >>> len(attacker.poisoned_loader.dataset)
                2
            (E) 边界条件与测试建议: 当攻击模式为 targeted 但测试集中缺少 source_label 时需提前检测；
                建议测试 1) 正常抽样是否成功；2) 当抽样数量达到上限时能否优雅报错。
            (F) 背景参考: 可阅读联邦学习中数据投毒章节，以及 Partition 数据抽样策略。
        """
        # sample a small poisoned dataset from the test dataset
        posioned_indices = dataset_class_indices(
            self.test_dataset, class_label=self.source_label if self.attack_model == "targeted" else None)
        indices = np.random.choice(
            posioned_indices, self.poisoned_sample_cnt, replace=False)
        train_trans, _ = get_transform(self.args)
        poisoned_dataset = Partition(self.test_dataset, indices, train_trans)
        poisoned_dataset.poison_setup(self.synthesizer)
        self.poisoned_loader = torch.utils.data.DataLoader(
            poisoned_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)

    def local_training(self):
        """执行 AlterMin 本地训练流程，交替优化隐蔽损失与目标损失。

        概述:
            先在良性数据上最小化隐蔽损失，再判断攻击成功率，若未达标则在恶意数据上执行目标优化并对更新进行放大。

        参数:
            无。

        返回:
            Tuple[float, float]: 良性训练阶段的准确率与损失。

        异常:
            RuntimeError: 若模型训练过程中出现梯度计算错误，将沿用 PyTorch 异常抛出。

        复杂度:
            时间复杂度 O(E_b * B + E_m * B_p)，其中 E_b/E_m 分别为 benign 与 malicious 轮次，B/B_p 为对应数据量。
            空间复杂度 O(|θ|)，取决于模型参数维度。

        费曼学习法:
            (A) 函数在“隐蔽训练”后，根据是否成功攻击决定要不要强化恶意更新。
            (B) 好比间谍先伪装练习不露馅，再查看任务是否完成，若未成功就大胆行动并把结果放大。
            (C) 步骤拆解:
                1. 调用基类 `local_training`，以自定义 `stealth_loss` 在良性数据上训练若干轮，积累隐蔽性。
                2. 使用 `test` 在投毒数据上评估攻击成功率（ASR）和目标损失。
                3. 若 ASR 损失大于零，说明尚未完全成功，需继续目标优化：
                    a. 记录优化前的参数向量，为后续放大计算差值。
                    b. 在恶意数据上再训练若干轮，聚焦于目标损失。
                    c. 记录优化后的参数向量，并计算差分。
                    d. 通过 `boosting_factor` 放大全部差分，形成增强后的参数。
                    e. 使用 `vec2model` 将放大后的向量写回模型，实现显式 boosting。
            (D) 示例:
                >>> acc, loss = attacker.local_training()
                >>> print(acc, loss)
            (E) 边界条件与测试建议: 若 `poisoned_loader` 为空会导致目标阶段失败；建议测试
                1) 仅隐蔽阶段是否正常结束；2) ASR 损失大于零时是否正确执行 boosting。
            (F) 背景参考: 推荐阅读交替最小化（Alternating Minimization）与模型投毒放大策略的相关章节。
        """
        # 1. benign training with stealth objectives (normal+distance loss) on benign data
        train_acc, train_loss = super().local_training(
            criterion_fn=self.stealth_loss, local_epochs=self.benign_epochs)
        # 2. get asr (attack success rate) loss, if asr loss > 0, do targeted objective optimization
        asr, asr_loss = self.test(self.model, self.poisoned_loader)
        # optimize the target objective if malicious loss is not zero yet
        if asr_loss > 0.0:
            # 3. get weights before and after malicious training
            pre_train_weights = copy.deepcopy(model2vec(self.model))
            # malicious training with adversarial objectives on poisoned data
            super().local_training(train_loader=self.cycle(
                self.poisoned_loader), local_epochs=self.malicous_epochs)
            # 记录恶意训练后的模型参数，用于计算增益差异。
            post_train_weights = copy.deepcopy(model2vec(self.model))

            # 5. check target objective condition. If not, do targeted objective optimization, explicit boosting
            boosted_weights = pre_train_weights + self.boosting_factor * \
                (post_train_weights - pre_train_weights)
            # 6. load the boosted weights to the model parameters
            vec2model(boosted_weights, self.model)
        return train_acc, train_loss

    def stealth_loss(self, y_pred, y_true):
        """隐蔽损失函数：交叉熵损失与模型漂移惩罚的加权组合。

        概述:
            在标准交叉熵上施加模型参数与全局参数向量的 L2 距离，约束恶意更新偏差以提高隐蔽性。

        参数:
            y_pred (Tensor): 模型输出的 logits，形状为 (batch_size, num_classes)。
            y_true (Tensor): 样本真实标签，形状为 (batch_size,)。

        返回:
            Tensor: 标量损失值。

        异常:
            RuntimeError: 当输入张量形状不匹配或包含 NaN 时可能抛出。

        复杂度:
            时间复杂度 O(|θ|)，因需计算参数向量差值；空间复杂度 O(|θ|)。

        费曼学习法:
            (A) 该函数在普通分类损失外，加上一项“不要离全局太远”的惩罚。
            (B) 好比走在大部队里，一边遵守集体步伐（交叉熵），一边提醒自己别离队太远（距离惩罚）。
            (C) 步骤拆解:
                1. 调用 `model2vec` 把当前模型参数展平为向量，便于与全局向量比较。
                2. 计算与 `global_weights_vec` 的 L2 距离，衡量偏离程度。
                3. 调用交叉熵损失函数处理常规分类目标。
                4. 将两者按权重 `rho` 合并，得到最终隐蔽损失。
            (D) 示例:
                >>> loss = attacker.stealth_loss(y_pred, y_true)
            (E) 边界条件与测试建议: 确保 `global_weights_vec` 已更新；可测试
                1) 当模型与全局参数一致时距离项为零；2) 修改 `rho` 对总损失的影响。
            (F) 背景参考: 可阅读正则化技巧、模型投毒中的距离惩罚策略。
        """
        # 计算模型与全局参数的向量距离，约束恶意更新幅度。
        distance_loss = np.linalg.norm(
            model2vec(self.model) - self.global_weights_vec, 2)
        # 将标准交叉熵损失与距离惩罚组合，得到隐蔽损失。
        return torch.nn.CrossEntropyLoss()(y_pred, y_true) + self.rho * distance_loss

    def client_test(self, model=None, test_dataset=None, poison_epochs=False):
        """扩展客户端测试逻辑，可选择在恶意样本上评估攻击表现。

        概述:
            根据 `poison_epochs` 标志决定在常规测试集或投毒数据上评估模型，返回准确率与损失。

        参数:
            model (Optional[nn.Module]): 指定要评估的模型，不提供时使用当前模型。
            test_dataset (Optional[Dataset]): 指定评估数据集，不提供时使用默认测试集。
            poison_epochs (bool): 若为 True，则在缓存的投毒样本上评估攻击信心。

        返回:
            Tuple[float, float]: 测试准确率与损失；当 `poison_epochs=True` 时损失固定为 0。

        异常:
            RuntimeError: 当模型前向传播失败或数据加载异常时可能抛出。

        复杂度:
            时间复杂度 O(n)，其中 n 为评估样本数；空间复杂度 O(1)（额外内存常数级）。

        费曼学习法:
            (A) 该函数灵活选择评估对象：要么检测正常性能，要么衡量恶意样本的欺骗程度。
            (B) 类似教练既要看运动员常规赛发挥，也要单独检查关键战术是否奏效。
            (C) 步骤拆解:
                1. 若传入模型或数据集，则复制引用以避免修改默认状态。
                2. 当 `poison_epochs=True` 时，拼接恶意数据批次并调用 `malicious_samples_confidence` 计算目标置信度。
                3. 否则，获取常规测试集 DataLoader 并调用 `test` 检查性能。
                4. 返回准确率（或置信度）与对应损失。
            (D) 示例:
                >>> acc, loss = attacker.client_test(poison_epochs=False)
                >>> asr, _ = attacker.client_test(poison_epochs=True)
            (E) 边界条件与测试建议: 若 `poisoned_loader` 未初始化将失败；建议测试
                1) 常规测试路径；2) 恶意评估路径，验证损失为 0。
            (F) 背景参考: 可阅读联邦学习评估指标与攻击成功率度量的相关章节。
        """
        model = self.new_if_given(model, self.model)
        test_dataset = self.new_if_given(test_dataset, self.test_dataset)
        if poison_epochs:
            # 聚合恶意样本批次以便统一评估目标置信度。
            images, targets = map(lambda x: torch.cat(
                x, dim=0), zip(*self.poisoned_loader))
            # confidence of the target class
            test_acc, test_loss = self.malicious_samples_confidence(
                images, targets, model), 0
        else:
            test_loader = self.get_dataloader(
                test_dataset, train_flag=False, poison_epochs=False)
            test_acc, test_loss = self.test(model, test_loader)

        return test_acc, test_loss

    def malicious_samples_confidence(self, images, targets, model):
        """计算恶意样本集合在目标类别上的置信度或预测准确率。

        概述:
            使用 softmax 概率评估给定模型对恶意样本的输出，以衡量攻击对目标标签的吸引力。

        参数:
            images (Tensor): 恶意样本的图像张量，形状为 (n, C, H, W)。
            targets (Tensor): 恶意样本的真实目标标签（已投毒），形状为 (n,)。
            model (nn.Module): 待评估模型。

        返回:
            float: 单样本时为目标类别概率，多样本时为预测等于目标标签的比例。

        异常:
            RuntimeError: 当模型前向传播失败或张量位于不同设备时可能抛出。

        复杂度:
            时间复杂度 O(n * |θ|)（前向传播成本），空间复杂度 O(n)。

        费曼学习法:
            (A) 该函数衡量模型在恶意样本上“有多相信目标标签”。
            (B) 好比测量迷惑法术是否奏效：看守卫是否把伪装者认成指定身份。
            (C) 步骤拆解:
                1. 将模型切换到评估模式并关闭梯度，确保稳定预测。
                2. 对输入图像执行前向传播并通过 softmax 得到各类别概率。
                3. 若只有单个样本，直接取目标标签对应的概率。
                4. 若包含多个样本，统计预测类别与目标标签一致的比例。
            (D) 示例:
                >>> conf = attacker.malicious_samples_confidence(images, targets, attacker.model)
            (E) 边界条件与测试建议: 输入需位于同一设备；可测试
                1) 单样本时返回概率值；2) 多样本时返回正确率。
            (F) 背景参考: 建议复习 softmax 与分类置信度概念，以及攻击成功率（ASR）。
        """
        model.eval()
        with torch.no_grad():
            # get the probabilities of each class by softmax
            probabilities = F.softmax(
                model(images.to(self.args.device)), dim=1)
            if len(images) == 1:  # confidence of single malicious sample
                target_confidences = probabilities[0, targets[0].item()].item()
            elif len(images) > 1:  # multiple malicious samples
                target_confidences = torch.sum(
                    targets == np.argmax(probabilities.cpu(), axis=1)) / len(images)

        return target_confidences


# __AI_ANNOTATION_SUMMARY__
# 类 AlterMin: 混合型 AlterMin 攻击器，交替最小化隐蔽与目标损失完成投毒。
# 方法 __init__: 初始化攻击参数与数据管线，准备隐蔽与恶意训练资源。
# 方法 define_synthesizer: 构建标签翻转合成器以生成投毒样本。
# 方法 init_poisoned_loader: 从测试集抽样并构建恶意数据迭代器。
# 方法 local_training: 执行隐蔽训练与目标 boosting，输出基础训练指标。
# 方法 stealth_loss: 融合交叉熵与参数距离的隐蔽损失函数。
# 方法 client_test: 在常规或恶意数据上评估模型表现。
# 方法 malicious_samples_confidence: 计算恶意样本在目标类别上的置信度或准确率。
