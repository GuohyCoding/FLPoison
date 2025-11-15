# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np
import torch

# ======The two main APIs for model and 1-d numpy array conversion======


def vec2model(vector, model, plus=False, ignorebn=False):
    """将 1D numpy 向量写回模型参数（原地修改）。

    概述:
        遍历模型 state_dict，按顺序从输入向量切片并 reshape，赋值给每一层参数。
        可选择忽略 BatchNorm 统计参数，并支持增量式更新（plus=True）。

    参数:
        vector (np.ndarray): 展平的模型参数向量。
        model (torch.nn.Module): 目标模型实例。
        plus (bool): 若为 True，则执行 in-place 加法，否则执行赋值。
        ignorebn (bool): 是否忽略 BatchNorm 的 running_mean / running_var 等统计量。

    返回:
        None: 函数直接修改传入的模型参数。

    异常:
        ValueError: 当向量长度与模型参数数量不匹配时可能产生 IndexError（隐式）；建议调用前校验。

    复杂度:
        时间 O(n)，空间 O(1)（除新建 tensor 视图外）。

    费曼学习法:
        (A) 函数把平铺的“参数清单”重新装回模型。
        (B) 类比按照说明书顺序将零件装回机器。
        (C) 步骤拆解:
            1. 取得模型 state_dict 及设备信息。
            2. 逐层计算参数元素数，从向量中切片并 reshape。
            3. 将切片 tensor 复制到对应参数上，支持加法或覆盖。
            4. 更新切片索引，继续下一层。
        (D) 示例:
            >>> vec2model(flat_vector, model, plus=False)
        (E) 边界条件与测试建议: 向量长度需匹配；建议写单元测试验证 round-trip 转换。
        (F) 背景参考: 模型参数序列化与联邦学习梯度交换的常用方式。
    """
    curr_idx = 0
    model_state_dict = model.state_dict()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for key, value in model_state_dict.items():
        if ignorebn and any(substring in key for substring in ['running_mean', 'running_var', 'num_batches_tracked']):
            continue
        numel = value.numel()
        param_tensor = torch.from_numpy(
            vector[curr_idx:curr_idx + numel].reshape(value.shape)).to(device=device, dtype=dtype)

        if plus:
            value.copy_(value + param_tensor)  # in-place addition
        else:
            value.copy_(param_tensor)  # in-place assignment
        curr_idx += numel

    # Note that the below method are only suitable for CNN without batch normalization layer
    # vector2parameter(vector, model)


def model2vec(model):
    """将模型参数转换为 1D numpy 向量。

    参数:
        model (torch.nn.Module): 待转换的模型。

    返回:
        np.ndarray: 展平后的参数向量。

    费曼学习法:
        (A) 函数像把机器拆开，按顺序把所有零件排成一列。
        (B) 类比仓库点货：把每个箱子里的货物依次拿出来排好。
    """
    return state2vec(model.state_dict())
    # return parameter2vector(model)


def add_vec2model(vector, model_template):
    """将向量表示的“伪梯度”加到模型副本上，返回更新后的新模型。

    参数:
        vector (np.ndarray): 更新向量。
        model_template (torch.nn.Module): 模型模板，函数内部会深拷贝。

    返回:
        torch.nn.Module: 加完向量后的新模型实例。

    费曼学习法:
        (A) 函数复制一个模型，再把向量加到副本上，得到新模型。
        (B) 类比复印一份报告，然后在复印件上加批注。
    """
    tmp_model = deepcopy(model_template)
    vec2model(vector, tmp_model, plus=True)
    return tmp_model

# ======Below are specific implementations======


def vector2parameter(vector, model):
    """使用向量直接覆盖模型参数（逐个 param.data 赋值）。

    仅适用于不含 BatchNorm 的场景，保留兼容性。
    """
    current_pos = 0
    for param in model.parameters():
        numel = param.numel()  # get the number of elements in param
        param.data = torch.from_numpy(
            vector[current_pos:current_pos + numel].reshape(param.shape)).to(param.device)
        current_pos += numel


def parameter2vector(model):
    """将模型参数转为 1D numpy 向量（不包含 BN 统计量）。"""
    model_parameters = model.parameters()
    return np.concatenate([param.detach().cpu().numpy().flatten() for param in model_parameters])


def set_grad_none(model):
    """将模型所有参数的梯度设为 None，常用于释放显存或重置梯度。"""
    for param in model.parameters():
        param.grad = None


def vector2gradient(vector, model):
    """根据向量重写模型梯度张量（逐层赋值）。

    费曼学习法:
        (A) 函数像根据成绩单重写每位学生的分数。
        (B) 类比在训练前预设梯度，便于模拟攻击或自定义更新。
    """
    current_pos = 0
    parameters = model.parameters()
    for param in parameters:
        numel = param.numel()  # get the number of elements in param
        param.grad = torch.from_numpy(
            vector[current_pos:current_pos + numel].reshape(param.shape)).to(param.device)
        current_pos += numel


def gradient2vector(model):
    """将模型梯度展平成 1D numpy 向量。"""
    parameters = model.parameters()
    return np.concatenate([param.grad.cpu().numpy().flatten() for param in parameters])


def ol_from_vector(vector, model_template, flatten=True, return_type='dict'):
    """从参数向量中提取输出层（最后一层）的权重和偏置。

    参数:
        vector (np.ndarray): 完整模型参数向量。
        model_template (torch.nn.Module): 模板模型，用于确定输出层结构。
        flatten (bool): 是否保持向量形式，不 reshape。
        return_type (str): `'dict'` 或 `'vector'`，决定返回格式。

    返回:
        Dict[str, np.ndarray] 或 np.ndarray: 输出层权重/偏置字典或拼接向量。

    费曼学习法:
        (A) 函数像从“总账本”中剪出最后章节（输出层参数）。
        (B) 类比翻阅书籍，取出最后一章的正文和附录。
        (C) 步骤拆解:
            1. 读取 state_dict 获取输出层权重与偏置形状。
            2. 计算对应在向量中的偏移量。
            3. 切片出权重与偏置，按需 reshape。
            4. 根据 `return_type` 返回字典或拼接向量。
    """
    state_template = model_template.state_dict()
    # Get keys for the last two layers (weight and bias)
    output_layer_keys = list(state_template.keys())[-2:]

    # Get the shapes of the weight and bias
    weight_shape = state_template[output_layer_keys[0]].shape
    bias_shape = state_template[output_layer_keys[1]].shape

    # Calculate sizes
    weight_size = np.prod(weight_shape)
    bias_size = np.prod(bias_shape)

    # Start with the last element of the vector for bias, then weight
    bias = vector[-bias_size:
                  ] if flatten else vector[-bias_size:].reshape(bias_shape)
    weights = vector[-(bias_size + weight_size):-bias_size] if flatten else vector[-(bias_size + weight_size):-
                                                                                   bias_size].reshape(weight_shape)
    if return_type == 'dict':
        return {'weight': weights, 'bias': bias}
    elif return_type == 'vector':
        # !DON'T change the order of weights and bias, as it's the order of the output layer and the order of the state_dict vector
        if flatten:  # concatenate 1d vectors
            return np.concatenate([weights.flatten(), bias.flatten()])
        else:
            # concatenate the weights and bias at axis 1, i.e., column-wise, to produce a 2d array with same number of rows and added bias columns
            return np.concatenate([weights, bias.reshape(bias_size, -1)], axis=1)


def ol_from_model(model, flatten=True, return_type='dict'):
    """基于模型实例提取输出层参数（封装 `ol_from_vector`）。"""
    return ol_from_vector(model2vec(model), model,
                          flatten=flatten, return_type=return_type)


def vec2state(vector, model, plus=False, ignorebn=False, numpy=False):
    """将向量转换为模型 state_dict（深拷贝），支持增量与忽略 BN。

    参数:
        vector (np.ndarray): 参数向量。
        model (torch.nn.Module): 模板模型。
        plus (bool): 是否执行加法更新。
        ignorebn (bool): 是否忽略 BatchNorm 统计量。
        numpy (bool): 返回值是否转为 numpy 格式。

    返回:
        OrderedDict 或 Dict[str, np.ndarray]: 新的 state_dict 或 numpy 字典。

    费曼学习法:
        (A) 类似 `vec2model`，但返回的是新的 state_dict 而不直接修改模型。
        (B) 类比复制一本手册并在副本上写入更新内容。
    """
    curr_idx = 0
    model_state_dict = deepcopy(model.state_dict())
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for key, value in model_state_dict.items():
        if ignorebn and any(substring in key for substring in ['running_mean', 'running_var', 'num_batches_tracked']):
            continue
        numel = value.numel()
        param_tensor = torch.from_numpy(
            vector[curr_idx:curr_idx + numel].reshape(value.shape)).to(device=device, dtype=dtype)

        if plus:
            value.copy_(value + param_tensor)  # in-place addition
        else:
            value.copy_(param_tensor)  # in-place assignment
        curr_idx += numel
    if numpy:
        return {key: value.detach().cpu().numpy() for key, value in model_state_dict.items()}

    return model_state_dict


def state2vec(model_state_dict, ignorebn=False, numpy_flg=False):
    """将 state_dict 转为拼接的 1D numpy 向量。

    参数:
        model_state_dict (OrderedDict): 模型 state dict。
        ignorebn (bool): 是否忽略 BatchNorm 统计量。
        numpy_flg (bool): 值是否已经是 numpy 数组，避免重复转换。

    返回:
        np.ndarray: 展平后的参数向量。

    费曼学习法:
        (A) 函数把 state_dict 里的张量逐个取出展平并拼接。
        (B) 类比将书本里的每页照片按顺序切下来排成一列。
        (C) 步骤拆解:
            1. 根据 `ignorebn` 决定是否过滤 BN 统计量。
            2. 迭代 state_dict，将每个张量展平成 numpy。
            3. `np.concatenate` 拼接成单一向量。
        (D) 示例:
            >>> vec = state2vec(model.state_dict(), ignorebn=True)
        (E) 边界条件与测试建议: 需确保 state_dict 序列稳定；测试 round-trip 是否保持一致性。
        (F) 背景参考: 模型参数序列化技术。
    """
    if numpy_flg:
        arrays = [
            value.flatten()
            for name, value in model_state_dict.items()
            if (True if not ignorebn else all(substring not in name for substring in ['running_mean', 'running_var', 'num_batches_tracked']))
        ]
    else:
        arrays = [
            tensor.detach().cpu().numpy().flatten()
            for name, tensor in model_state_dict.items()
            if (True if not ignorebn else all(substring not in name for substring in ['running_mean', 'running_var', 'num_batches_tracked']))
        ]

    return np.concatenate(arrays)


# __AI_ANNOTATION_SUMMARY__
# 函数 vec2model: 将参数向量写回模型，支持增量更新与忽略 BN。
# 函数 model2vec: 将模型参数转换为 1D 向量。
# 函数 add_vec2model: 返回加上向量更新后的模型副本。
# 函数 vector2parameter: 直接覆盖模型参数（兼容旧实现）。
# 函数 parameter2vector: 将模型参数拼接为向量。
# 函数 set_grad_none: 将所有梯度置为 None。
# 函数 vector2gradient: 用向量重写模型梯度。
# 函数 gradient2vector: 将梯度拼接为向量。
# 函数 ol_from_vector/ol_from_model: 提取输出层权重与偏置。
# 函数 vec2state: 返回向量对应的 state_dict 副本。
# 函数 state2vec: 将 state_dict 展平成 1D numpy 向量。
