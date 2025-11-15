# -*- coding: utf-8 -*-

from torchvision import models


def add_vgg_from_torchvision():
    """收集 torchvision 中所有 VGG 模型，返回供注册表使用的字典。

    概述:
        遍历 `torchvision.models` 中以 `vgg` 开头的模型名称，取出对应构造函数，
        并统一转换为小写键以便注册。

    返回:
        Dict[str, Callable]: 键为小写模型名，值为模型构造函数。

    费曼学习法:
        (A) 函数像在图书馆索引里挑选所有 VGG 系列书目，并整理成清单。
        (B) 类比翻阅目录，找到所有 “VGG” 开头的章节编号，方便后续引用。
        (C) 步骤拆解:
            1. 初始化空字典 `reg`。
            2. 遍历 `torchvision.models.vgg` 下的属性，筛选以 `vgg` 开头的名称。
            3. 通过 `getattr` 获取模型构造函数，并以小写名称作为键存入字典。
            4. 返回整理好的字典，供外部注册使用。
        (D) 示例:
            >>> registry = add_vgg_from_torchvision()
            >>> model = registry['vgg16'](pretrained=False)
        (E) 边界条件与测试建议: 需确保 torchvision 版本包含 VGG 模型；建议测试字典键值是否正确映射。
        (F) 背景参考: torchvision.models 模块、VGG 网络结构。
    """
    reg = {}
    vgg_models = [i for i in models.vgg.__dict__.keys() if i.startswith('vgg')]
    for model_name in vgg_models:
        model_fn = getattr(models, model_name)
        reg[model_name.lower()] = model_fn
    return reg


# __AI_ANNOTATION_SUMMARY__
# 函数 add_vgg_from_torchvision: 遍历 torchvision VGG 模型并返回小写名到构造函数的映射。
