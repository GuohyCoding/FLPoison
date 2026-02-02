# -*- coding: utf-8 -*-

import torch
from fl.models.model_utils import model2vec_torch, vec2model
from .algorithmbase import AlgorithmBase
from fl.algorithms import algorithm_registry


@algorithm_registry
class FedAvg(AlgorithmBase):
    """FedAvg（Federated Averaging）算法的基础实现。

    FedAvg 由 McMahan 等人在 2017 年提出，是联邦学习中最经典的聚合策略：
    每个客户端在本地执行若干轮梯度下降，然后服务器对各客户端模型参数取平均，
    得到新的全局模型并广播。
    """

    def __init__(self, args, model, optimizer=None):
        """初始化 FedAvg 算法，保存配置与模型引用。

        参数:
            args (argparse.Namespace): 全局运行配置，需包含 `local_epochs` 等字段。
            model (torch.nn.Module): 当前被联邦训练的模型。
            optimizer (Optional[torch.optim.Optimizer]): 本地优化器引用，基类未直接使用。

        费曼学习法:
            (A) 方法保证 FedAvg 拿到运行所需的参数与模型引用。
            (B) 类比联邦训练前确认作战手册与武器装备都在手。
        """
        super().__init__(args, model, optimizer)

    def init_local_epochs(self):
        """返回客户端本地训练的 epoch 数量。

        返回:
            int: 本地迭代轮次 `self.args.local_epochs`。

        费曼学习法:
            (A) 告诉客户端每次需要练几圈。
            (B) 类比教练给出今日训练的圈数。
            (C) 步骤拆解:
                1. 读取配置中的 `local_epochs`。
                2. 将该值返回给调用方用于循环控制。
            (D) 示例:
                >>> epochs = fedavg.init_local_epochs()
            (E) 边界条件与测试建议: 确保配置中存在该字段；调试时可打印检查。
        """
        return self.args.local_epochs

    def get_local_update(self, **kwargs):
        """在本地训练后返回模型参数向量，供服务器聚合。

        返回:
            torch.Tensor: 展平后的模型参数。

        费曼学习法:
            (A) 将本地模型“打包”成向量，上交服务器。
            (B) 类比运动员提交训练成果统计表。
            (C) 步骤拆解:
                1. 调用 `model2vec` 将模型参数展平成向量。
                2. 返回该向量供服务器平均。
            (D) 示例:
                >>> update = fedavg.get_local_update()
            (E) 边界条件与测试建议: 需确保模型参数可被访问；可验证向量维度与模型参数数量一致。
        """
        update = model2vec_torch(self.model)
        return update

    def update(self, aggregated_update, **kwargs):
        """将服务器聚合后的参数向量写回模型，实现全局同步。

        参数:
            aggregated_update (torch.Tensor): 聚合后的模型参数向量。

        返回:
            torch.Tensor: 与输入相同，便于链式调用或日志记录。

        费曼学习法:
            (A) 把所有客户端的平均更新载入全局模型。
            (B) 类比教练根据全队报告更新统一战术手册。
            (C) 步骤拆解:
                1. 通过 `vec2model` 将向量恢复成模型参数。
                2. 更新后的模型即可被下轮客户端下载。
            (D) 示例:
                >>> fedavg.update(aggregated_weights)
            (E) 边界条件与测试建议: 向量维度需与模型完全匹配；建议检验更新后模型参数确实发生变化。
        """
        vec2model(aggregated_update, self.model)
        return aggregated_update


# __AI_ANNOTATION_SUMMARY__
# 类 FedAvg: 实现联邦平均算法的核心流程。
# 方法 __init__: 保存配置与模型引用，准备运行环境。
# 方法 init_local_epochs: 返回客户端本地训练轮次数。
# 方法 get_local_update: 展平模型参数以提交至服务器。
# 方法 update: 将聚合向量载入模型，实现全局更新。
