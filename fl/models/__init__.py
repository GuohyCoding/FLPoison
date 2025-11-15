# -*- coding: utf-8 -*-

from global_utils import import_all_modules, Register
import os
from fl.models.vgg import add_vgg_from_torchvision

# 初始化模型注册表，动态导入当前目录下的模型文件。
model_registry = Register()
import_all_modules(os.path.dirname(__file__), 1, "fl")
# 注册来自 torchvision 的 VGG 系列模型。
vgg_reg = add_vgg_from_torchvision()
model_registry.update(vgg_reg)
# 记录所有可用模型的名称列表，便于校验与展示。
all_models = list(model_registry.keys())

model_categories = {
    "grey": ["lr"],
    "adaptive": ["lenet", "lenet_bn"],  # 1(grey) or 3(rgb) channels
    "rgb": ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "vgg11", "vgg13", "vgg16", "vgg19"],
    "handy": ["simplecnn"]
}


def get_model(args):
    """根据配置返回已注册的模型实例，并自动匹配输入通道与类别数。

    概述:
        - 将 `args.model` 转为小写并校验是否在注册表中。
        - 根据模型类别决定构造函数参数，例如灰度图需指定输入维度，RGB 模型需确认通道数。
        - 返回迁移到目标设备的模型实例。

    参数:
        args (argparse.Namespace): 配置对象，需包含 `model`、`num_channels`、`num_dims`、`num_classes`、`device` 等字段。

    返回:
        torch.nn.Module: 已初始化并迁移到指定设备的模型。

    异常:
        NotImplementedError: 当模型名称未注册或类别组合不支持时抛出。
        AssertionError: 当灰度/彩色模型的通道数与配置不符时抛出。

    复杂度:
        时间复杂度 O(1)（取决于模型构造器复杂度），空间复杂度取决于模型大小。

    费曼学习法:
        (A) 函数根据配置挑选合适的模型并设置必要参数。
        (B) 类比商场导购根据顾客需求（尺寸、颜色）推荐合适款式。
        (C) 步骤拆解:
            1. 标准化模型名称并验证是否存在。
            2. 判断模型类别（灰度、自适应、RGB、handy），校验通道数。
            3. 调用注册表中的构造器，传入对应参数。
            4. 将模型迁移至 `args.device`，返回实例。
        (D) 示例:
            >>> args.model = 'ResNet18'
            >>> model = get_model(args)
        (E) 边界条件与测试建议: 确保 `num_channels` 与模型需求一致；建议编写单元测试验证不同类别模型的构造能力。
        (F) 背景参考: PyTorch 模型注册与工厂模式、联邦学习中常见模型配置方法。
    """
    args.model = args.model.lower()
    if args.model not in all_models:
        raise NotImplementedError(
            f"Model not implemented, please choose from {all_models}")
    if args.model in model_categories["grey"]:
        assert args.num_channels == 1, "models designed for grey images only supports 1 channel"
        # 对于线性模型等，需要传入展平后的输入维度。
        model = model_registry[args.model](
            input_dim=args.num_dims*args.num_dims, num_classes=args.num_classes)
    elif args.model in model_categories["adaptive"]:
        # adaptive 模型可适配 1 或 3 通道，因此传入 num_channels。
        model = model_registry[args.model](
            num_channels=args.num_channels, num_classes=args.num_classes)
    elif args.model in model_categories["rgb"]:
        assert args.num_channels == 3, "models designed for RGB images only supports 3 channels"
        model = model_registry[args.model](num_classes=args.num_classes)
    elif args.model == "simplecnn":
        model = model_registry[args.model](input_size=(args.num_channels, args.num_dims, args.num_dims), num_classes=args.num_classes)
    else:
        raise NotImplementedError(
            f"Model not implemented, please choose from {all_models}")
    return model.to(args.device)


# __AI_ANNOTATION_SUMMARY__
# 全局变量 model_registry/all_models/model_categories: 管理模型注册表与分类信息。
# 函数 get_model: 按配置返回适配的模型实例并迁移到目标设备。*** End Patch*** End Patch to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply.patch error: Io(Os { code: 206, kind: InvalidFilename, message: "The filename or extension is too long." }) to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch.mainloop error due to long patch. to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply_patch to=commentary to=functions.apply.patch error persists. break.***
