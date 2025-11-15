# -*- coding: utf-8 -*-

import copy
import numpy as np
import torch
from attackers.pbases.dpbase import DPBase
from attackers.pbases.mpbase import MPBase
from .synthesizers import PixelSynthesizer
from datapreprocessor.data_utils import subset_by_idx
from global_utils import actor
from fl.models.model_utils import gradient2vector, model2vec, vec2model
from attackers import attacker_registry
from fl.client import Client


@attacker_registry
@actor('attacker', 'data_poisoning', 'model_poisoning')
class Neurotoxin(MPBase, DPBase, Client):
    """Neurotoxin 攻击器：利用罕见更新坐标隐藏后门并结合梯度裁剪保持持久性。

    基于 ICML 2022 论文《Neurotoxin: Durable Backdoors in Federated Learning》，
    通过挑选良性客户端很少更新的坐标（梯度掩码）嵌入后门，同时对梯度范数进行裁剪以规避检测。

    属性:
        default_attack_params (dict): 默认攻击参数，包含采样数量、Top-k 比例、梯度范数阈值及标签翻转设定。
        synthesizer (PixelSynthesizer): 像素触发器生成器。
        grad_mask_vec (np.ndarray): 训练过程中生成的梯度掩码向量。
        algorithm (str): 默认聚合算法，设置为 `FedSGD`。
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        """初始化 Neurotoxin 攻击客户端并配置默认参数。

        概述:
            设定梯度掩码所需的采样规模、Top-k 比例与范数阈值，构建触发器合成器，并生成投毒训练数据加载器。

        参数:
            args (argparse.Namespace): 运行配置，需提供 `batch_size`、`num_workers`、`device`、`epochs` 等信息。
            worker_id (int): 当前恶意客户端编号。
            train_dataset (Dataset): 本地训练数据集。
            test_dataset (Dataset): 本地测试数据集。

        返回:
            None。

        异常:
            AttributeError: 当 `args` 缺失必要字段时由基类自动抛出。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)（不含后续训练与采样开销）。

        费曼学习法:
            (A) 该函数为 Neurotoxin 攻击者准备默认参数、触发器与投毒数据加载器。
            (B) 类比特工准备隐藏线路：先选好掩护（掩码参数），再布置暗号（触发器），最后安排行动时间表（投毒轮次）。
            (C) 步骤拆解:
                1. 调用 `Client.__init__` 初始化联邦客户端上下文。
                2. 设置默认攻击参数（采样量、Top-k 比例、范数阈值、标签翻转策略等）。
                3. 使用 `update_and_set_attr` 合并外部配置。
                4. 调用 `define_synthesizer` 构建像素触发器与投毒逻辑。
                5. 根据攻击策略生成投毒轮次，并通过 `get_dataloader` 构建训练数据加载器。
                6. 指定默认聚合算法为 `FedSGD`。
            (D) 示例:
                >>> attacker = Neurotoxin(args, worker_id=1, train_dataset=train_ds, test_dataset=test_ds)
                >>> attacker.norm_threshold
                0.2
            (E) 边界条件与测试建议: 确保 `num_sample` 不大于训练数据量；可测试默认参数可否覆盖与投毒调度是否生效。
            (F) 背景参考: 《Neurotoxin》原论文、联邦学习后门综述。
        """
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        """
        num_sample: number of benign samples to train for calculating the gradient mask
        topk_ratio: ratio of top-k smallest absolute gradient values
        norm_threshold: clipping threshold of gradient norm 
        """
        self.default_attack_params = {
            'num_sample': 64,
            'topk_ratio': 0.1,
            'norm_threshold': 0.2,
            "attack_model": "all2one",
            "poisoning_ratio": 0.32,
            "target_label": 6,
            "source_label": 1,
            "attack_strategy": "continuous",
            "single_epoch": 0,
            "poison_frequency": 5,
        }
        self.update_and_set_attr()

        self.define_synthesizer()
        poison_epochs = self.generate_poison_epochs(
            self.attack_strategy, self.args.epochs, self.single_epoch, self.poison_frequency)
        self.train_loader = self.get_dataloader(
            train_dataset, train_flag=True, poison_epochs=poison_epochs)
        self.algorithm = 'FedSGD'

    def define_synthesizer(self):
        """构建 Neurotoxin 的像素触发器合成器。

        概述:
            创建固定大小的白色像素触发器，并实例化 `PixelSynthesizer`，以支持后续标签翻转与触发器植入。

        参数:
            无。

        返回:
            None。

        异常:
            ValueError: 当触发器尺寸与数据集不匹配时（由合成器内部抛出）。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)。

        费曼学习法:
            (A) 该方法准备一个可插入图像的触发器工具。
            (B) 类比制作统一的贴纸模板，之后按同样方式贴在照片角落。
            (C) 步骤拆解:
                1. 创建尺寸为 5×5 的触发器张量（默认全 1 表示亮块）。
                2. 实例化 `PixelSynthesizer`，填入攻击模式、标签映射与投毒比例。
                3. 将生成的合成器保存为实例属性，以在训练过程中复用。
            (D) 示例:
                >>> attacker.define_synthesizer()
                >>> attacker.trigger.shape
                torch.Size([1, 5, 5])
            (E) 边界条件与测试建议: 确保输入图像尺寸大于触发器；可测试合成器是否正确翻转标签与贴入触发器。
            (F) 背景参考: 像素触发器后门、数据投毒流程。
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

    def local_training(self):
        """执行本地训练，先生成梯度掩码再进行有约束的后门训练。

        概述:
            先基于全局模型与干净样本计算梯度掩码，再调用父类 `local_training` 进行带掩码的后门训练。

        参数:
            无。

        返回:
            Tuple[float, float]: 训练准确率与损失。

        异常:
            ValueError: 当梯度掩码计算失败或数据不足时（由 `get_gradient_mask` 抛出）。

        复杂度:
            时间复杂度 O(n * d)，n 为采样样本数，d 为模型参数维度；空间复杂度 O(d)。

        费曼学习法:
            (A) 函数先找出哪些参数少被别人动过，再在这些位置进行后门训练。
            (B) 类比潜入图书馆先观察哪些书很少被借，再在这些书里藏线索。
            (C) 步骤拆解:
                1. 调用 `get_gradient_mask` 生成稀疏坐标掩码。
                2. 将掩码存入实例属性，供后续优化步骤使用。
                3. 调用父类 `local_training`，在 `step` 中应用掩码与梯度裁剪。
                4. 返回训练准确率与损失。
            (D) 示例:
                >>> acc, loss = attacker.local_training()
            (E) 边界条件与测试建议: 确保 `num_sample` 足够大以稳定估计；建议测试掩码是否更新、训练流程是否正常返回指标。
            (F) 背景参考: Neurotoxin 攻击流程、梯度掩码思想。
        """
        # 1. get unfrequently-used gradient mask via global model and clean data
        self.grad_mask_vec = self.get_gradient_mask()
        # 2. backdoor training while apply the gradient mask and PGD in step()
        train_acc, train_loss = super().local_training()
        return train_acc, train_loss

    def get_gradient_mask(self):
        """计算梯度掩码，选取良性客户端少更新的坐标。

        概述:
            从全局模型出发，用采样的干净数据执行一次良性训练，获取梯度向量，
            并选取其绝对值最小的 Top-k 坐标作为掩码。

        参数:
            无。

        返回:
            numpy.ndarray: 梯度掩码向量，与模型参数维度一致，取值为 0 或 1。

        异常:
            ValueError: 当采样数据为空或 `topk_ratio` 无法得到有效 k 时。

        复杂度:
            时间复杂度 O(n * d)，n 为采样样本数，d 为模型参数维度；空间复杂度 O(d)。

        费曼学习法:
            (A) 该函数找出“别人很少碰的旋钮”，之后只在这些位置动手脚。
            (B) 类比在旅店观察哪些房门很少被打扫，再把秘密藏在这些房间里。
            (C) 步骤拆解:
                1. 随机抽取 `num_sample` 个干净样本构成子数据集。
                2. 克隆当前模型，使用抽样数据做一次良性训练。
                3. 将更新转换为梯度向量。
                4. 计算梯度绝对值最小的 Top-k 坐标，生成掩码向量。
                5. 返回掩码用于后续训练步骤。
            (D) 示例:
                >>> mask = attacker.get_gradient_mask()
                >>> mask.dtype
                float64
            (E) 边界条件与测试建议: 确保 k>0 且小于参数维度；可测试 1) 掩码中 1 的数量与 `topk_ratio` 一致；2) 采样数据量不足时提示补充。
            (F) 背景参考: 梯度稀疏性、Neurotoxin 掩码机制。
        """
        # benign training to get the frequently-updated coordinates by benign clients
        # 1. sample `self.num_sample` clean data, and do benign training on global model
        sample_indices = np.random.choice(
            range(len(self.train_dataset)), size=self.num_sample, replace=False)
        sampled_dataset = subset_by_idx(
            self.args, self.train_dataset, sample_indices)
        sampled_loader = torch.utils.data.DataLoader(
            sampled_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        # 深拷贝当前模型，避免污染原模型参数。
        new_model = copy.deepcopy(self.model)
        new_optimizer, lr_scheduler = self.get_optimizer_scheduler(new_model)
        super().train(new_model, iter(sampled_loader), new_optimizer)

        # 2 get vectorized gradient from the above training
        grad_vec = gradient2vector(new_model)

        # 3. get the indices of top-k smallest absolute gradient value as the gradient mask
        k = int(len(grad_vec) * self.topk_ratio)
        if k <= 0:
            raise ValueError("topk_ratio 太小，无法产生有效的梯度掩码。")
        idx = np.argpartition(np.abs(grad_vec), k)
        grad_mask_vec = np.zeros(len(grad_vec))
        grad_mask_vec[idx[:k]] = 1.0
        return grad_mask_vec

    def step(self, optimizer, **kwargs):
        """在每个优化步中应用梯度掩码与梯度范数裁剪。

        概述:
            先按掩码筛选梯度，再执行父类优化步骤，随后对模型偏移进行 L2 裁剪，防止更新过大。

        参数:
            optimizer (torch.optim.Optimizer): 当前使用的优化器。
            **kwargs: 兼容父类接口的扩展参数。

        返回:
            None。

        异常:
            AttributeError: 当 `grad_mask_vec` 尚未初始化时。

        复杂度:
            时间复杂度 O(d)，空间复杂度 O(d)。

        费曼学习法:
            (A) 函数在每步优化前戴上“滤镜”，只保留少数坐标，再确保步子不迈太大。
            (B) 类比蒙着眼走迷宫：先用手杖限定可行方向，再控制每次迈步长度。
            (C) 步骤拆解:
                1. 使用 `apply_grad_mask` 将梯度限制在掩码坐标上。
                2. 调用父类 `step` 执行常规更新。
                3. 获取模型参数向量，与全局权重比较得到偏移。
                4. 依据 `norm_threshold` 对偏移做 L2 裁剪。
                5. 将裁剪后的参数写回模型。
            (D) 示例:
                >>> attacker.step(optimizer)
            (E) 边界条件与测试建议: 确保 `grad_mask_vec` 已由 `local_training` 生成；可测试裁剪前后范数是否受限。
            (F) 背景参考: 梯度裁剪、投影梯度下降（PGD）思想。
        """
        # project gradient with gradient mask
        if not hasattr(self, "grad_mask_vec"):
            raise AttributeError("梯度掩码未生成，请先调用 local_training。")
        self.apply_grad_mask(self.model.parameters(), self.grad_mask_vec)
        super().step(optimizer)
        # gradient norm clipping
        model_params_vec = model2vec(self.model)
        weight_diff = model_params_vec - self.global_weights_vec
        scale = np.minimum(1, self.norm_threshold /
                           np.linalg.norm(weight_diff))
        weight_diff *= scale
        vec2model(self.global_weights_vec + weight_diff,
                  self.model)

    def apply_grad_mask(self, parameters, grad_mask_vec):
        """对模型参数梯度应用掩码，仅保留指定坐标。

        概述:
            遍历模型参数，将梯度与掩码对应区段相乘，屏蔽未选择的更新坐标。

        参数:
            parameters (Iterable[Tensor]): 模型参数迭代器。
            grad_mask_vec (np.ndarray): 与模型参数总维度一致的掩码向量。

        返回:
            None。

        异常:
            ValueError: 当掩码长度与参数元素总数不一致时。

        复杂度:
            时间复杂度 O(d)，空间复杂度 O(1)。

        费曼学习法:
            (A) 函数像给梯度披上一张筛子，只让掩码位置的梯度通过。
            (B) 类比厨房过滤汤料，只让特定调料落入锅中。
            (C) 步骤拆解:
                1. 初始化指针 `current_pos`，记录掩码截取位置。
                2. 遍历每个参数张量，计算其元素数量。
                3. 按数量切片掩码，并重塑为参数形状。
                4. 将参数梯度与掩码相乘，屏蔽不需要的梯度分量。
                5. 更新指针，处理下一个参数。
            (D) 示例:
                >>> attacker.apply_grad_mask(model.parameters(), mask_vec)
            (E) 边界条件与测试建议: 确保掩码长度等于参数总元素数；可测试掩码中为 0 的坐标梯度是否归零。
            (F) 背景参考: 梯度稀疏化、坐标选择策略。
        """
        params = list(parameters)
        current_pos = 0
        total_elements = sum(param.numel() for param in params)
        if total_elements != len(grad_mask_vec):
            raise ValueError("梯度掩码长度与参数元素总数不一致。")
        # 逐个参数切片掩码并与梯度相乘。
        for param in params:
            numel = param.numel()  # get the number of element of param
            mask_slice = grad_mask_vec[current_pos:current_pos + numel]
            mask_tensor = torch.from_numpy(mask_slice.reshape(param.shape)).to(param.device)
            if param.grad is not None:
                param.grad *= mask_tensor
            current_pos += numel


# __AI_ANNOTATION_SUMMARY__
# 类 Neurotoxin: 持久后门攻击器，利用梯度掩码与范数裁剪隐藏后门。
# 方法 __init__: 初始化默认攻击参数、触发器与投毒训练管线。
# 方法 define_synthesizer: 构建像素触发器合成器。
# 方法 local_training: 先生成梯度掩码，再执行受限后门训练。
# 方法 get_gradient_mask: 使用干净样本估计罕见坐标并生成掩码。
# 方法 step: 在每步优化时应用掩码与范数裁剪。
# 方法 apply_grad_mask: 将掩码逐段作用于参数梯度。
