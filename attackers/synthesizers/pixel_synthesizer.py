"""
像素触发器合成器 PixelSynthesizer：直接以张量方式注入预设像素触发器。

适用于已通过代码构造或从文件加载的触发器张量，负责转换其数据类型、归一化并设置默认放置位置。
"""
import torch
from torchvision import transforms

from .synthesizer import Synthesizer


class PixelSynthesizer(Synthesizer):
    """
    像素级触发器合成器：处理已有触发器张量并输出标准化结果。
    """

    def __init__(self, args, trigger, **kwargs) -> None:
        """
        初始化像素触发器合成器。

        参数:
            args (argparse.Namespace | Any): 全局运行配置，需提供 `mean`、`std`、`num_channels` 等属性。
            trigger (torch.Tensor): 触发器张量，可为 (C,H,W) 或 (N,C,H,W) 格式。
            **kwargs: 额外配置，通过 `set_kwargs` 写入实例属性。

        返回:
            None

        复杂度:
            时间复杂度 O(1)；空间复杂度 O(1)，仅保存引用。
        """
        self.args = args
        self.trigger = trigger
        # 将额外参数写入实例，如 attack_model、target_label 等。
        self.set_kwargs(kwargs)

        trigger_height, trigger_width = self.trigger.shape[-2], self.trigger.shape[-1]
        # 默认将触发器放置于图像左下角（BadNets 设定）。
        self.trigger_position = (-trigger_height, -trigger_width)
        self.setup_trigger()

    def setup_trigger(self, trigger=None):
        """
        对触发器张量进行类型转换与归一化，生成最终可叠加的触发器。

        参数:
            trigger (torch.Tensor | None): 可选的外部触发器张量；默认使用 self.trigger。

        返回:
            None

        复杂度:
            时间复杂度 O(H * W)；空间复杂度 O(H * W)，H/W 为触发器尺寸。
        """
        if self.trigger.dtype != torch.float32:
            self.trigger = self.trigger.to(dtype=torch.float32)

        norm_transform = transforms.Normalize(self.args.mean, self.args.std)

        # 若提供多触发器张量 (N,C,H,W)，默认选用第一组触发器并扩展通道。
        if len(self.trigger.shape) == 4:
            selected_trigger = self.trigger.expand(
                -1, self.args.num_channels, *self.trigger.shape[-2:]
            )
        else:
            selected_trigger = self.trigger.expand(
                self.args.num_channels, *self.trigger.shape[-2:]
            )

        # 归一化触发器张量，后续可直接叠加到样本上。
        self.trigger = norm_transform(selected_trigger)


# 费曼学习法解释（PixelSynthesizer.__init__）
# (A) 功能概述：保存触发器张量与配置，并调用预处理生成标准触发器。
# (B) 类比说明：像把贴纸、胶水和贴法说明准备好，随时可贴到目标物体上。
# (C) 步骤拆解：
#     1. 记录运行配置与触发器张量。
#     2. 将额外参数通过 `set_kwargs` 写入实例。
#     3. 根据触发器尺寸设置默认放置位置。
#     4. 调用 `setup_trigger` 完成归一化与通道处理。
# (D) 最小示例：
#     >>> synth = PixelSynthesizer(args, trigger_tensor, attack_model='targeted')
# (E) 边界条件与测试建议：触发器需与数据集通道匹配；建议测试灰度/彩色及多触发器场景。
# (F) 参考：像素级后门触发器研究（如 BadNets）。


# 费曼学习法解释（PixelSynthesizer.setup_trigger）
# (A) 功能概述：将原始触发器张量转换为浮点张量并按数据集均值方差归一化。
# (B) 类比说明：像把贴纸裁好尺寸、上好胶水和底色，确保贴上后色彩融合。
# (C) 步骤拆解：
#     1. 确保触发器数据类型为 float32，避免与模型输入类型不符。
#     2. 创建归一化变换，使用数据集均值方差。
#     3. 若触发器有批维度，按 `num_channels` 扩展或重复。
#     4. 对触发器执行归一化并保存。
# (D) 最小示例：
#     >>> synth.setup_trigger()
#     >>> synth.trigger.shape
# (E) 边界条件与测试建议：通道数必须匹配；多触发器时默认选第一组，可按需改造。
# (F) 参考：图像归一化与触发器叠加流程。


__AI_ANNOTATION_SUMMARY__ = """
PixelSynthesizer.__init__: 保存触发器张量与配置并调用预处理生成标准触发器。
PixelSynthesizer.setup_trigger: 将触发器张量转为 float32、对齐通道并归一化。
"""
