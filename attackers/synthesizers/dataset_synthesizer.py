"""
数据集级触发注入器 DatasetSynthesizer：直接混合或替换数据集实现投毒。

该类适用于“已预构建投毒数据集”的场景，绕过常规逐批触发器注入流程，
可在 DPBase 的 `get_dataloader` 中配合使用，直接返回混合后的训练集或投毒测试集。
"""
from .synthesizer import Synthesizer


class DatasetSynthesizer(Synthesizer):
    """
    数据集级投毒合成器：通过混合已有投毒样本对数据集进行整体替换。

    注意事项:
        - 使用此合成器需要在 `dpbase.py` 中重写 `get_dataloader`，
          以便直接返回混合后的投毒数据集，而非逐批注入触发器。
    """

    def __init__(self, args, train_dataset, poisoned_dataset, poisoned_ratio) -> None:
        """
        初始化数据集合成器。

        参数:
            args (argparse.Namespace | Any): 全局运行配置。
            train_dataset (torch.utils.data.Dataset): 原始训练数据集。
            poisoned_dataset (Any): 提供投毒样本的数据集或包装器，需实现 mix/get 接口。
            poisoned_ratio (float): 投毒样本比例，用于混合训练集。

        返回:
            None

        复杂度:
            时间复杂度 O(1)；空间复杂度 O(1)（仅保存引用）。
        """
        self.args = args
        self.poisoned_dataset = poisoned_dataset
        self.poisoned_ratio = poisoned_ratio
        self.train_dataset = train_dataset

    def get_poisoned_set(self, train):
        """
        根据训练/测试模式返回对应的投毒数据集。

        参数:
            train (bool): True 表示训练模式，False 表示测试模式。

        返回:
            torch.utils.data.Dataset: 投毒后的训练集或测试集。

        复杂度:
            取决于 `poisoned_dataset` 的实现；此方法本身为 O(1) 调用。
        """
        if train:
            # 对训练集按比例混合投毒样本。
            return self.poisoned_dataset.mix_trainset(
                self.train_dataset, self.poisoned_ratio
            )
        # 返回预置的投毒测试集，用于评估攻击成功率。
        return self.poisoned_dataset.get_poisoned_testset()


# 费曼学习法解释（DatasetSynthesizer.__init__）
# (A) 功能概述：记录原始/投毒数据集及比例，准备后续混合。
# (B) 类比说明：像实验前准备两桶溶液和混合比例，以便随时调配。
# (C) 步骤拆解：
#     1. 保存运行配置与原始训练集。
#     2. 保存投毒数据集包装器。
#     3. 记录投毒比例。
# (D) 最小示例：
#     >>> synth = DatasetSynthesizer(args, clean_ds, poisoned_ds, poisoned_ratio=0.2)
# (E) 边界条件与测试建议：确保 `poisoned_dataset` 提供 `mix_trainset` 与 `get_poisoned_testset` 方法。
# (F) 参考：数据投毒攻击的预混合策略。


# 费曼学习法解释（DatasetSynthesizer.get_poisoned_set）
# (A) 功能概述：根据训练或测试需求返回混合后的投毒数据集。
# (B) 类比说明：像根据不同实验选择混合溶液或原液。
# (C) 步骤拆解：
#     1. 判断当前是训练模式还是测试模式。
#     2. 训练模式下调用 `mix_trainset` 按比例混合投毒样本。
#     3. 测试模式下返回预先构造好的投毒测试集。
# (D) 最小示例：
#     >>> train_set = synth.get_poisoned_set(train=True)
#     >>> test_set = synth.get_poisoned_set(train=False)
# (E) 边界条件与测试建议：投毒比例需合理；确保 `poisoned_dataset` 实现混合逻辑。
# (F) 参考：基于数据集的后门注入方法。


__AI_ANNOTATION_SUMMARY__ = """
DatasetSynthesizer.__init__: 保存原始/投毒数据集与投毒比例，准备混合。
DatasetSynthesizer.get_poisoned_set: 根据训练或测试场景返回投毒混合数据集。
"""
