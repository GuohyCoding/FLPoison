# -*- coding: utf-8 -*-

from functools import partial

from attackers.pbases.dpbase import DPBase
from .synthesizers import Synthesizer
from global_utils import actor
from attackers import attacker_registry
from fl.client import Client


@attacker_registry
@actor('attacker', 'data_poisoning')
class LabelFlipping(DPBase, Client):
    """标签翻转攻击器：通过替换标签实现数据投毒，可模拟多种映射策略。

    支持 `targeted`、`all2one`、`all2all` 与 `random` 四种标签替换模型，可按策略生成投毒数据。

    属性:
        default_attack_params (dict): 默认攻击参数，包括投毒比例、标签映射与调度策略。
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        """初始化标签翻转攻击客户端，并构建投毒数据管线。

        概述:
            设定默认标签映射参数，生成标签翻转合成器，并按调度策略构造投毒训练数据加载器。

        参数:
            args (argparse.Namespace): 运行配置，需包含 `epochs`、`batch_size`、`num_workers` 等字段。
            worker_id (int): 当前攻击客户端编号。
            train_dataset (Dataset): 本地训练数据集。
            test_dataset (Dataset): 本地测试数据集。

        返回:
            None。

        异常:
            AttributeError: 当 `args` 缺失必要字段时由基类抛出。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)（不含数据加载器构建）。

        费曼学习法:
            (A) 构造一个能按策略翻转标签并按计划注入数据的攻击客户端。
            (B) 类比厨师准备好“陷阱食材”并安排何时端上桌。
            (C) 步骤拆解:
                1. 调用 `Client.__init__` 获取联邦训练上下文。
                2. 写入默认的标签映射与投毒调度配置。
                3. 调用 `update_and_set_attr` 合并外部参数。
                4. 运行 `define_synthesizer` 创建标签翻转合成器。
                5. 计算需要投毒的训练轮次。
                6. 基于调度结果构造投毒训练数据加载器。
            (D) 示例:
                >>> attacker = LabelFlipping(args, worker_id=0, train_dataset=train, test_dataset=test)
                >>> attacker.attack_model
                'targeted'
            (E) 边界条件与测试建议: 确保 `source_label` 与 `target_label` 在数据集中存在；
                建议测试投毒调度是否正确（如 `poison_frequency`）。
            (F) 背景参考: 标签翻转攻击、联邦学习数据投毒策略。
        """
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        """
        attack_model: all2one, all2all, targeted, random
        source label: the label to be flipped
        target label: the label to be flipped to
        """
        self.default_attack_params = {
            'attack_model': 'targeted',
            'source_label': 2,
            'target_label': 7,
            "attack_strategy": "continuous",
            "single_epoch": 0,
            "poison_frequency": 5,
            "poisoning_ratio": 0.32,
        }
        self.update_and_set_attr()

        self.define_synthesizer()
        poison_epochs = self.generate_poison_epochs(
            self.attack_strategy, self.args.epochs, self.single_epoch, self.poison_frequency)
        self.train_loader = self.get_dataloader(
            train_dataset, train_flag=True, poison_epochs=poison_epochs)

    def define_synthesizer(self):
        """创建标签翻转合成器，并根据策略覆写关键方法。

        概述:
            实例化 `Synthesizer`，将标签翻转逻辑与批量投毒策略绑定，
            确保仅替换标签、不植入触发器。

        参数:
            无。

        返回:
            None。

        异常:
            ValueError: 当标签映射配置非法时可能抛出。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)（不含数据遍历）。

        费曼学习法:
            (A) 设定标签翻转合成器，并根据攻击模式决定哪些批次被翻转。
            (B) 类比给助手一份“替换标签清单”，并指示哪些班次执行。
            (C) 步骤拆解:
                1. 创建 `Synthesizer`，指定攻击模式与标签映射。
                2. 若为 `targeted` 模式，强制每个批次投毒；否则保留投毒比例。
                3. 通过 `partial` 绑定 `train` 参数，控制投毒策略。
                4. 将 `implant_backdoor` 覆写为不做图像触发器植入，仅执行标签翻转。
            (D) 示例:
                >>> attacker.define_synthesizer()
                >>> attacker.synthesizer.attack_model
                'targeted'
            (E) 边界条件与测试建议: 确保 `source_label != target_label`；
                建议测试不同攻击模式下批次选取逻辑。
            (F) 背景参考: 标签翻转策略、数据驱动的后门构造。
        """
        self.synthesizer = Synthesizer(
            self.args, None,
            attack_model=self.attack_model,
            target_label=self.target_label,
            poisoning_ratio=1,
            source_label=self.source_label,
            single_epoch=self.single_epoch,
        )

        # 标签翻转只改标签，不跟随 poisoning_ratio。targeted 模式下所有批次均投毒。
        train = False if self.attack_model == 'targeted' else True
        self.synthesizer.backdoor_batch = partial(
            self.synthesizer.backdoor_batch, train=train)
        # 覆写 implant_backdoor，禁用图像触发器，仅执行标签替换。
        self.synthesizer.implant_backdoor = partial(
            self.synthesizer.implant_backdoor,
            implant_trigger=lambda image, kwargs: None,
        )


# __AI_ANNOTATION_SUMMARY__
# 类 LabelFlipping: 标签翻转数据投毒攻击器。
# 方法 __init__: 初始化标签映射与投毒调度，构造训练数据加载器。
# 方法 define_synthesizer: 创建并定制标签翻转合成器，控制批次投毒与触发器逻辑。
