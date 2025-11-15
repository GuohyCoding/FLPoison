"""
触发器合成器基类 Synthesizer：提供后门投毒批处理与触发器植入的通用接口。

子类可重写触发器植入策略、投毒索引选择逻辑等，实现特定的后门攻击形式。
"""
import random

import torch


class Synthesizer:
    """
    后门触发器合成器基类，为不同触发器类型提供统一的批处理逻辑。
    """

    def __init__(self, args, trigger, **kwargs) -> None:
        """
        初始化合成器并写入配置与触发器张量。

        参数:
            args (argparse.Namespace | Any): 全局运行配置，需包含数据集信息、类别数等。
            trigger (torch.Tensor | Any): 触发器对象，具体含义由子类解释。
            **kwargs: 额外参数，通过 `set_kwargs` 写入实例属性（如 `attack_model`、`target_label` 等）。

        返回:
            None

        复杂度:
            时间复杂度 O(1)；空间复杂度 O(1)。
        """
        self.args = args
        self.trigger = trigger
        self.set_kwargs(kwargs)

    def set_kwargs(self, kwargs):
        """
        将额外参数写入实例属性。

        参数:
            kwargs (dict): 额外配置字典。

        返回:
            None

        复杂度:
            时间复杂度 O(k)，k 为参数数量；空间复杂度 O(1)。
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def backdoor_batch(self, images, labels, train=False, **kwargs):
        """
        对输入批次执行后门投毒：选择部分样本植入触发器并修改标签。

        参数:
            images (torch.Tensor): 输入图像，形状 (N,C,H,W) 或 (C,H,W)。
            labels (torch.Tensor | int): 对应标签。
            train (bool): 是否处于训练模式，决定投毒比例。
            **kwargs: 可包含 `worker_id`、`implant_trigger` 等自定义参数。

        返回:
            tuple[torch.Tensor, torch.Tensor]:
                - 可能已投毒的图像张量；
                - 对应标签张量。

        复杂度:
            时间复杂度 O(N)；空间复杂度 O(N)，N 为批大小。
        """
        worker_id = kwargs.get("worker_id", None)

        if len(images.shape) == 4:
            len_images = images.shape[0]
            poisoned_idx_per_batch = self.setup_poisoned_idx(len_images, train)
            for idx in range(len_images):
                if idx in poisoned_idx_per_batch:
                    images[idx], labels[idx] = self.implant_backdoor(
                        images[idx],
                        labels[idx],
                        train=train,
                        worker_id=worker_id,
                    )
        else:
            images, labels = self.implant_backdoor(
                images,
                labels,
                train=train,
                worker_id=worker_id,
            )
            labels = torch.tensor(labels)
        return images, labels

    def implant_backdoor(self, image, label, **kwargs):
        """
        根据攻击模式修改标签并植入触发器。

        参数:
            image (torch.Tensor): 待投毒图像。
            label (torch.Tensor | int): 原标签。
            **kwargs: 允许覆盖 `implant_trigger`、传入 `worker_id` 等。

        返回:
            tuple[torch.Tensor, torch.Tensor]: (新的图像, 更新后的标签)。

        复杂度:
            时间复杂度 O(1)；空间复杂度 O(1)。
        """
        implant_trigger = kwargs.get("implant_trigger", self.implant_trigger)
        if self.attack_model == "all2one":
            label = self.target_label
            implant_trigger(image, kwargs)
        elif self.attack_model == "random":
            label = torch.tensor(
                random.choice(range(self.args.num_classes)),
                dtype=torch.int64,
            )
            implant_trigger(image, kwargs)
        elif self.attack_model == "all2all":
            label = self.args.num_classes - 1 - label
            implant_trigger(image, kwargs)
        elif self.attack_model == "targeted":
            assert self.source_label != self.target_label, self.args.logger.info(
                f"! Source label: {self.source_label}, Target label: {self.target_label} should not be equal"
            )
            if label == self.source_label:
                label = self.target_label
                implant_trigger(image, kwargs)
        return image, label

    def implant_trigger(self, image, kwargs):
        """
        默认触发器植入函数，支持子类或调用方覆盖。

        参数:
            image (torch.Tensor): 待投毒图像。
            kwargs (dict): 可能包含 `trigger` 或其他自定义参数。

        返回:
            None（就地修改图像）。

        复杂度:
            时间复杂度 O(h * w)；空间复杂度 O(1)。
        """
        trigger = kwargs.get("trigger", self.trigger)
        trigger_height, trigger_width = trigger.shape[-2], trigger.shape[-1]
        row_starter, column_starter = self.trigger_position
        image[
            ...,
            row_starter : self.zero2none(row_starter + trigger_height),
            column_starter : self.zero2none(column_starter + trigger_width),
        ] = trigger

    def setup_poisoned_idx(self, len_images, train, **kwargs):
        """
        生成当前批次需要投毒的样本索引。

        参数:
            len_images (int): 批大小。
            train (bool): 是否训练模式。
            **kwargs: 可包含 `poisoning_len`，用于指定固定的投毒数量。

        返回:
            list[int]: 被选中用于投毒的样本索引。

        复杂度:
            时间复杂度 O(N)（采样）；空间复杂度 O(1)。
        """
        poisoning_len = kwargs.get("poisoning_len", None)
        if poisoning_len is None:
            batch_poisoning_ratio = self.poisoning_ratio if train else 1
            batch_poisoning_len = int(len_images * batch_poisoning_ratio)
        else:
            batch_poisoning_len = poisoning_len

        poisoned_idx_per_batch = random.sample(
            range(len_images), batch_poisoning_len
        )
        return poisoned_idx_per_batch

    def zero2none(self, x):
        """
        将切片右端点为 0 的情况转换为 None，兼容负索引切片。

        参数:
            x (int): 切片右端点。

        返回:
            int | None: 若 x 为 0 返回 None，否则返回 x。
        """
        return x if x != 0 else None


# 费曼学习法解释（Synthesizer.__init__）
# (A) 功能概述：保存配置与触发器张量，并写入额外参数。
# (B) 类比说明：像准备投毒工具箱并把所有材料标好名字。
# (C) 步骤拆解：记录 args、trigger，并调用 `set_kwargs` 绑定额外属性。
# (D) 示例：
#     >>> synth = Synthesizer(args, trigger_tensor, attack_model='targeted')
# (E) 边界条件与测试建议：确保 kwargs 中的键合法；测试属性是否成功注入。
# (F) 参考：面向对象初始化与配置注入模式。


# 费曼学习法解释（Synthesizer.set_kwargs）
# (A) 功能概述：把额外配置写成实例属性。
# (B) 类比说明：像给工具箱里的器材贴上标签，随取随用。
# (C) 步骤拆解：遍历字典，将键值对通过 setattr 写入对象。
# (D) 示例：
#     >>> synth.set_kwargs({'target_label': 1})
#     >>> synth.target_label
#     1
# (E) 边界条件与测试建议：键名冲突会覆盖原属性；建议测试覆盖。
# (F) 参考：Python 动态属性注入。


# 费曼学习法解释（Synthesizer.backdoor_batch）
# (A) 功能概述：按比例选择样本植入触发器并修改标签。
# (B) 类比说明：像在一批作业中随机挑几份偷偷改答案。
# (C) 步骤拆解：
#     1. 判断输入是否为批次。
#     2. 生成需要投毒的样本索引。
#     3. 对选中的样本调用 `implant_backdoor`。
# (D) 示例：
#     >>> imgs, labels = synth.backdoor_batch(batch_imgs, batch_labels, train=True)
# (E) 边界条件与测试建议：当投毒比例为 0 或 1 时需正确处理；测试单样本与批处理两种情况。
# (F) 参考：后门攻击批处理流程。


# 费曼学习法解释（Synthesizer.implant_backdoor）
# (A) 功能概述：根据攻击模式调整标签并植入触发器。
# (B) 类比说明：像根据不同作弊策略修改试卷答案。
# (C) 步骤拆解：
#     1. 选择植入函数（默认 `implant_trigger`）。
#     2. 根据 attack_model 更新标签。
#     3. 对满足条件的样本植入触发器。
# (D) 示例：
#     >>> new_img, new_label = synth.implant_backdoor(img, label, train=True)
# (E) 边界条件与测试建议：需保证 attack_model 合法；针对每种模式编写测试验证标签变换。
# (F) 参考：后门攻击策略分类（all2one、all2all 等）。


# 费曼学习法解释（Synthesizer.implant_trigger）
# (A) 功能概述：将触发器张量叠加到图像的指定位置。
# (B) 类比说明：像把贴纸贴到图片的角落。
# (C) 步骤拆解：
#     1. 获取触发器张量及尺寸。
#     2. 使用切片赋值将触发器区域替换。
# (D) 示例：
#     >>> synth.implant_trigger(img, {})
# (E) 边界条件与测试建议：触发器尺寸与位置需匹配；建议测试负索引切片的正确性。
# (F) 参考：BadNets 触发器植入方法。


# 费曼学习法解释（Synthesizer.setup_poisoned_idx）
# (A) 功能概述：决定当前批次哪些样本需要投毒。
# (B) 类比说明：像在一群人中随机挑选若干人参与计划。
# (C) 步骤拆解：
#     1. 计算需要投毒的数量（训练模式用 poisoning_ratio，测试模式全投毒）。
#     2. 从批次中随机采样对应数量的索引。
# (D) 示例：
#     >>> idxs = synth.setup_poisoned_idx(len_batch, train=True)
# (E) 边界条件与测试建议：poisoning_ratio 为 0 或 1 时是否行为正确；建议测试固定投毒数量的覆盖。
# (F) 参考：后门攻击中投毒比例控制。


# 费曼学习法解释（Synthesizer.zero2none）
# (A) 功能概述：将切片右端点 0 转换为 None，兼容负索引的切片操作。
# (B) 类比说明：像在写区间时，用 None 表示“直到末尾”，避免与 0 混淆。
# (C) 步骤拆解：判断 x 是否为 0，是则返回 None，否则返回 x。
# (D) 示例：
#     >>> synth.zero2none(0)
#     None
# (E) 边界条件与测试建议：仅用于切片操作；建议测试负值、正值输入。
# (F) 参考：Python 切片语义。


__AI_ANNOTATION_SUMMARY__ = """
Synthesizer.__init__: 保存配置与触发器张量并注入额外属性。
Synthesizer.set_kwargs: 将额外配置写入实例属性。
Synthesizer.backdoor_batch: 按投毒比例对批次样本植入触发器并修改标签。
Synthesizer.implant_backdoor: 根据不同攻击模式执行标签修改与触发器植入。
Synthesizer.implant_trigger: 以切片方式将触发器叠加到图像指定区域。
Synthesizer.setup_poisoned_idx: 基于投毒比例随机选择当前批次的投毒样本。
Synthesizer.zero2none: 将切片右端点 0 转换为 None，兼容负索引切片。
"""
