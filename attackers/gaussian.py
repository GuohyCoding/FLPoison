# -*- coding: utf-8 -*-

import numpy as np

from global_utils import actor
from attackers.pbases.mpbase import MPBase
from attackers import attacker_registry
from fl.client import Client


@attacker_registry
@actor('attacker', 'model_poisoning', 'non_omniscient')
class Gaussian(MPBase, Client):
    """Gaussian 噪声攻击器：直接以高斯噪声取代本地更新，实现最简单的模型投毒。

    该实现参考 NeurIPS 2017 《Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent》，
    在非全知场景下，用配置的均值与方差生成噪声向量作为提交更新，扰乱聚合结果。

    属性:
        default_attack_params (dict): 默认噪声参数，包含 `noise_mean` 与 `noise_std`。
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        """初始化 Gaussian 攻击客户端，设置噪声参数。

        概述:
            调用基类构造函数以获取联邦学习上下文，并写入噪声分布的默认均值与标准差。

        参数:
            args (argparse.Namespace): 运行配置，允许覆盖噪声参数。
            worker_id (int): 客户端编号。
            train_dataset (Dataset): 本地训练数据集（此攻击不使用，但保持接口一致）。
            test_dataset (Dataset): 本地测试数据集（此攻击不使用，但保持接口一致）。

        返回:
            None。

        异常:
            AttributeError: 若 `args` 缺失必要字段时由基类自动抛出。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)。

        费曼学习法:
            (A) 该函数只是设定好噪声攻击的默认参数。
            (B) 类比设置一台噪声发生器的音量和音调，之后就可以直接播放噪声。
            (C) 步骤拆解:
                1. 调用 `Client.__init__` 完成基础上下文初始化。
                2. 设定默认噪声参数（均值 0、标准差 1）。
                3. 通过 `update_and_set_attr` 将默认值与外部配置合并。
            (D) 示例:
                >>> attacker = Gaussian(args, worker_id=0, train_dataset=train, test_dataset=test)
                >>> attacker.noise_std
                1
            (E) 边界条件与测试建议: 确保噪声标准差为正；可测试不同噪声参数是否正确写入属性。
            (F) 背景参考: 高斯分布、Byzantine 鲁棒学习中的噪声攻击。
        """
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {'noise_mean': 0, 'noise_std': 1}
        self.update_and_set_attr()

    def non_omniscient(self):
        """在非全知场景下生成高斯噪声作为本地更新提交。

        概述:
            根据配置的均值与标准差，采样与当前更新同维度的高斯噪声向量，并以 `float32` 返回。

        参数:
            无。

        返回:
            numpy.ndarray: 与 `self.update` 形状一致的噪声向量。

        异常:
            ValueError: 当 `noise_std` 为负数时，`np.random.normal` 会抛出异常。

        复杂度:
            时间复杂度 O(d)，d 为模型参数维度；空间复杂度 O(d)。

        费曼学习法:
            (A) 函数每次训练轮次都随机生成噪声向量作为恶意更新。
            (B) 类比向对讲机频道持续播放白噪声，干扰正常通信。
            (C) 步骤拆解:
                1. 读取当前更新的形状，确定噪声向量维度。
                2. 调用 `np.random.normal` 采样高斯噪声，使用配置的均值与标准差。
                3. 将结果转换为 `float32`，与联邦系统常用的数据类型保持一致。
                4. 返回噪声向量供框架提交。
            (D) 示例:
                >>> noise = attacker.non_omniscient()
                >>> noise.shape == attacker.update.shape
                True
            (E) 边界条件与测试建议: 标准差需大于 0；可测试 1) 噪声形状与 update 相同；2) 多次调用均值接近 `noise_mean`。
            (F) 背景参考: 高斯噪声模型、Byzantine 鲁棒联邦学习理论。
        """
        noise = np.random.normal(
            loc=self.noise_mean,
            scale=self.noise_std,
            size=self.update.shape,
        ).astype(np.float32)
        return noise


# __AI_ANNOTATION_SUMMARY__
# 类 Gaussian: 以高斯噪声取代本地更新的模型投毒攻击器。
# 方法 __init__: 初始化噪声参数并合并外部配置。
# 方法 non_omniscient: 采样高斯噪声作为提交更新。*** End Patch*** End Patch{"code":400,"stdout":"","stderr":"","error":"Invalid JSON: Expecting value: line 1 column 1 (char 0)"} ***!
