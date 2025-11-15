"""
数据投毒攻击基础类 DPBase：封装通用的触发器定义、数据加载与测试流程。

提供生成投毒 epoch、按需毒化数据迭代器、以及带有投毒测试的客户端评估方法，
便于不同数据投毒攻击在此基础上扩展具体逻辑。
"""
import torch
from attackers.pbases.pbase import PBase
from attackers.synthesizers.dataset_synthesizer import DatasetSynthesizer
from datapreprocessor.data_utils import (
    Partition,
    get_transform,
    dataset_class_indices,
)
from global_utils import frac_or_int_to_int


class DPBase(PBase):
    """
    数据投毒攻击基类：为具体数据投毒策略提供公共工具方法。
    """

    def define_synthesizer(self):
        """
        定义触发器与相关投毒组件。

        派生类应在此方法中构建 `self.synthesizer`、设置触发器模式、
        并初始化与数据投毒相关的成员变量。

        复杂度:
            时间复杂度 / 空间复杂度取决于具体继承实现。
        """
        raise NotImplementedError

    def get_dataloader(self, dataset, train_flag, poison_epochs=None):
        """
        根据训练/测试阶段及给定投毒计划返回迭代器，按需在指定轮次对批次执行投毒。

        参数:
            dataset (torch.utils.data.Dataset): 客户端可访问的数据集。
            train_flag (bool): True 表示训练模式（无限循环）；False 表示测试模式（单轮）。
            poison_epochs (bool | list[int] | None): 控制哪些 epoch 投毒。
                - None: 默认 False，表示不投毒；
                - bool: True 为所有 epoch 投毒，False 为完全不投毒；
                - list[int]: 仅在列表包含的全局轮次执行投毒。

        返回:
            生成器: 每次迭代返回 (images, targets)，若需要投毒则为投毒后的批次。

        复杂度:
            时间复杂度 O(|dataset|)；空间复杂度 O(batch_size)。
        """
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=train_flag,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        # 将 poison_epochs 统一为函数接口，便于按轮次判断是否投毒。
        poison_epochs = False if poison_epochs is None else poison_epochs
        if isinstance(poison_epochs, bool):
            def poisoning_epoch(_):  # 全局轮次是否投毒
                return poison_epochs
        elif isinstance(poison_epochs, list):
            def poisoning_epoch(epoch):  # 在指定轮次投毒
                return epoch in poison_epochs
        else:
            raise ValueError("poison_epochs 参数类型不受支持。")

        while True:
            for images, targets in dataloader:
                if poisoning_epoch(self.global_epoch):
                    # 对当前批次执行触发器注入，返回投毒样本。
                    yield self.synthesizer.backdoor_batch(
                        images,
                        targets,
                        train=train_flag,
                        worker_id=self.worker_id,
                    )
                else:
                    # 不投毒时直接返回原始样本。
                    yield images, targets

            if not train_flag:
                # 测试模式下仅迭代一轮。
                break

    def generate_poison_epochs(
        self,
        attack_strategy,
        epochs,
        single_epoch,
        poison_frequency,
        attack_start_epoch=None,
    ):
        """
        根据攻击策略生成投毒轮次列表。

        参数:
            attack_strategy (str): 投毒策略，可选 'single-shot'、'fixed-frequency'、'continuous'。
            epochs (int): 全局训练总轮次。
            single_epoch (int): 单次投毒策略下的目标轮次。
            poison_frequency (int | float): 固定频率投毒的周期，可为整数或比例。
            attack_start_epoch (int | None): 连续投毒的起点轮次，缺省表示从 0 开始。

        返回:
            list[int] | bool: 投毒轮次列表；连续投毒时返回 True 表示全程投毒。

        异常:
            ValueError: attack_strategy 不支持时抛出。

        复杂度:
            时间复杂度 O(epochs)；空间复杂度 O(k)，k 为投毒轮次数量。
        """
        if attack_strategy == "continuous":
            return list(range(attack_start_epoch, epochs)) if attack_start_epoch is not None else True
        if attack_strategy == "single-shot":
            if not isinstance(single_epoch, int):
                raise ValueError("single_epoch 应为整数。")
            return [single_epoch]
        if attack_strategy == "fixed-frequency":
            freq = frac_or_int_to_int(poison_frequency, epochs)
            return list(range(0, epochs, freq))
        raise ValueError("attack strategy not supported")

    def client_test(self, model=None, test_dataset=None, poison_epochs=False):
        """
        基于给定模型与数据集执行测试，可用于评估良性精度或投毒攻击成功率。

        参数:
            model (torch.nn.Module | None): 若提供则使用该模型，否则复制 self.model。
            test_dataset (torch.utils.data.Dataset | None): 若提供则使用该数据集，否则使用 self.test_dataset。
            poison_epochs (bool | list[int]): 控制测试阶段是否投毒；默认 False。

        返回:
            tuple[float, float]: (测试精度, 测试损失)。

        复杂度:
            时间复杂度取决于测试集大小；空间复杂度 O(batch_size)。
        """
        model = self.new_if_given(model, self.model)
        test_dataset = self.new_if_given(test_dataset, self.test_dataset)

        # 若合成器非数据级且攻击为定向攻击，需要构造投毒后的测试集以评估 ASR。
        if (
            not isinstance(self.synthesizer, DatasetSynthesizer)
            and self.attack_model == "targeted"
        ):
            poisoned_indices = dataset_class_indices(
                test_dataset, class_label=self.source_label
            )
            test_trans = get_transform(self.args)[1]
            poisoned_testset = Partition(test_dataset, poisoned_indices, test_trans)
            poisoned_testset.poison_setup(self.synthesizer)
            poisoned_loader = torch.utils.data.DataLoader(
                poisoned_testset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True,
            )
        else:
            # 其他情况复用 get_dataloader，按 poison_epochs 控制是否注入后门。
            poisoned_loader = self.get_dataloader(
                test_dataset, train_flag=False, poison_epochs=True if poison_epochs else False
            )

        test_acc, test_loss = self.test(model, poisoned_loader)
        return test_acc, test_loss


# 费曼学习法解释（DPBase.define_synthesizer）
# (A) 功能概述：派生类在此定义触发器和投毒相关变量。
# (B) 类比说明：像为实验准备材料，需要在开工前先声明工具与参数。
# (C) 步骤拆解：由子类负责实现，因此此处抛出 NotImplementedError 作为约束。
# (D) 示例：
#     >>> class MyAttack(DPBase):
#     ...     def define_synthesizer(self):
#     ...         self.synthesizer = CustomSynth()
# (E) 边界条件与测试建议：未实现会抛异常；测试时应确保子类正确覆盖。
# (F) 参考：面向对象设计中的抽象方法模式。


# 费曼学习法解释（DPBase.get_dataloader）
# (A) 功能概述：按指定策略对数据批次执行投毒并提供迭代器。
# (B) 类比说明：像课堂上按课表安排实验，有的课需要加入特殊材料，有的则按常规进行。
# (C) 步骤拆解：
#     1. 构建 PyTorch DataLoader。
#     2. 将 poison_epochs 统一为判断函数。
#     3. 训练模式下无限循环，测试模式仅一轮。
#     4. 若当前 epoch 需要投毒，则调用合成器注入后门。
# (D) 最小示例：
#     >>> loader = attack.get_dataloader(dataset, True, poison_epochs=[0,5])
# (E) 边界条件与测试建议：poison_epochs 类型必须正确；建议测试训练与测试模式的行为差异。
# (F) 参考：联邦投毒攻击流程、PyTorch DataLoader 用法。


# 费曼学习法解释（DPBase.generate_poison_epochs）
# (A) 功能概述：根据策略生成需要投毒的轮次列表或标志。
# (B) 类比说明：像制定课程安排，决定哪些周上实验课、哪些周上理论课。
# (C) 步骤拆解：
#     1. 若策略为连续，返回布尔 True 或指定起点后的连续列表。
#     2. 单次策略返回含一个元素的列表。
#     3. 固定频率策略将频率转换为整数步长并返回等差序列。
# (D) 最小示例：
#     >>> epochs = attack.generate_poison_epochs("fixed-frequency", 50, None, 0.2)
# (E) 边界条件与测试建议：频率转换需正确；策略字符串错误应抛异常；建议测试三种策略的结果是否符合预期。
# (F) 参考：联邦投毒攻击调度策略相关文献。


# 费曼学习法解释（DPBase.client_test）
# (A) 功能概述：在测试阶段评估模型性能或攻击成功率。
# (B) 类比说明：像考试时既要看正常题成绩，也要看针对性题目的表现。
# (C) 步骤拆解：
#     1. 根据传入参数选择模型与测试集。
#     2. 若攻击为定向且合成器非数据级，构造带触发器的测试集。
#     3. 否则复用 get_dataloader 控制投毒。
#     4. 运行测试并返回精度与损失。
# (D) 最小示例：
#     >>> acc, loss = attack.client_test(model=global_model, poison_epochs=True)
# (E) 边界条件与测试建议：合成器类型与攻击模式需匹配；建议分别测试 targeted 与 non-targeted 场景。
# (F) 参考：联邦投毒攻击评估方法、ASR（Attack Success Rate）计算。


__AI_ANNOTATION_SUMMARY__ = """
DPBase.define_synthesizer: 抽象方法，提醒子类实现触发器与投毒变量初始化。
DPBase.get_dataloader: 根据轮次计划返回可投毒的数据批次迭代器。
DPBase.generate_poison_epochs: 按不同策略生成投毒轮次列表或连续标志。
DPBase.client_test: 构建投毒测试集并评估模型精度或攻击成功率。
"""
