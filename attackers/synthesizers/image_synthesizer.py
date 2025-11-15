"""
图像触发器合成器 ImageSynthesizer：将真实图像触发器嵌入样本以实现后门投毒。

本类负责加载指定触发器图片、根据数据集通道自动识别颜色模式，并输出标准化后的触发器张量及默认放置位置。
"""
from torchvision import transforms
from PIL import Image
from .synthesizer import Synthesizer


class ImageSynthesizer(Synthesizer):
    """
    图像触发器合成器：用于在输入样本上叠加指定的图像触发器。
    """

    def __init__(
        self,
        args,
        trigger_path,
        trigger_size,
        attack_model,
        target_label,
        poisoning_ratio,
        source_label,
    ) -> None:
        """
        初始化触发器合成器，读取触发器图片并存储攻击配置。

        参数:
            args (argparse.Namespace | Any): 运行参数，需提供均值/方差、数据集信息等。
            trigger_path (str): 触发器图像文件路径。
            trigger_size (int): 触发器边长（正方形假设），单位为像素。
            attack_model (str): 攻击模式描述（如 targeted / untargeted）。
            target_label (int): 投毒后希望模型预测的目标标签。
            poisoning_ratio (float): 训练集中被投毒样本所占比例。
            source_label (int | None): 投毒来源标签（定向攻击时使用）。

        返回:
            None

        复杂度:
            时间复杂度 O(1)；空间复杂度 O(1)，仅保存配置引用。
        """
        # 记录基础配置与攻击属性。
        self.args = args
        self.trigger_path = trigger_path
        self.trigger_size = trigger_size
        self.target_label = target_label
        self.poisoning_ratio = poisoning_ratio
        self.source_label = source_label
        self.attack_model = attack_model

        # 加载并预处理触发器图像。
        self.setup_trigger()

    def setup_trigger(self, trigger=None):
        """
        加载并预处理触发器图像，生成标准化张量与默认位置。

        参数:
            trigger (PIL.Image.Image | None): 可选的外部触发器图像，若为 None 则从路径读取。

        返回:
            None

        复杂度:
            时间复杂度 O(H * W)；空间复杂度 O(H * W)，H/W 为触发器尺寸。
        """
        image_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.args.mean, self.args.std)]
        )

        # 根据数据集判断触发器图像模式：彩色 (RGB) 或灰度 (L)。
        mode = "RGB" if self.args.dataset == "CIFAR10" else "L"

        # 默认从 trigger_path 读取触发器；若外部传入 trigger 则使用传入对象。
        trigger_img = trigger or Image.open(self.trigger_path).convert(mode)
        trigger_size = (self.trigger_size, self.trigger_size)
        trigger_img = trigger_img.resize(trigger_size)

        # 转换为张量并存为类属性，便于下游叠加操作使用。
        self.trigger = image_transform(trigger_img)

        # 默认将触发器放置在图像左下角（BadNets 设定）。
        self.trigger_position = (-trigger_size[0], -trigger_size[1])


# 费曼学习法解释（ImageSynthesizer.__init__）
# (A) 功能概述：保存触发器相关配置并加载触发器图像。
# (B) 类比说明：像在实验前准备好试剂瓶、标签和使用说明。
# (C) 步骤拆解：
#     1. 记录攻击参数与触发器路径/大小。
#     2. 保存攻击模式、目标标签、投毒比例等信息。
#     3. 调用 `setup_trigger` 读取并标准化触发器。
# (D) 最小示例：
#     >>> synth = ImageSynthesizer(args, "trigger.png", 4, "targeted", 0, 0.1, 1)
# (E) 边界条件与测试建议：需保证 `trigger_path` 可读；建议测试不同数据集模式下的触发器加载。
# (F) 参考：BadNets 触发器设计、图像预处理流程。


# 费曼学习法解释（ImageSynthesizer.setup_trigger）
# (A) 功能概述：读取触发器图像，调整尺寸并标准化，准备叠加到样本上。
# (B) 类比说明：像把贴纸剪裁到合适大小，再涂上胶水随时可贴。
# (C) 步骤拆解：
#     1. 创建标准化变换，将图像转为张量并按数据集均值方差归一化。
#     2. 判断数据集类型决定使用 RGB 还是灰度模式。
#     3. 读取触发器图像并调整到设定大小。
#     4. 转为张量保存；设定默认放置位置为左下角。
# (D) 最小示例：
#     >>> synth.setup_trigger()
#     >>> synth.trigger.shape  # e.g., torch.Size([3, 4, 4])
# (E) 边界条件与测试建议：触发器尺寸需与原始图像兼容；建议测试不同尺寸、彩色/灰度模式的行为。
# (F) 参考：图像触发器注入方法、PyTorch 预处理 API。


__AI_ANNOTATION_SUMMARY__ = """
ImageSynthesizer.__init__: 保存投毒配置并加载触发器图像。
ImageSynthesizer.setup_trigger: 读取、缩放并标准化触发器图像，设置默认放置位置。
"""
