import torch
from fl.models.model_utils import model2vec_torch, vec2model
from .algorithmbase import AlgorithmBase
from fl.algorithms import algorithm_registry


@algorithm_registry
class FedOpt(AlgorithmBase):
    """FedOpt 算法：联邦优化框架的基础实现。

    FedOpt 系列（如 FedAdam、FedYogi）强调对服务器端聚合结果进行额外的优化处理。
    本实现提供基础框架：客户端提交梯度差，服务器将其加到全局权重上。
    子类可在此基础上扩展动量、适应学习率等高级策略。
    """

    def __init__(self, args, model, optimizer=None):
        """初始化 FedOpt 算法。

        参数:
            args (argparse.Namespace): 全局配置对象。
            model (torch.nn.Module): 当前联邦模型。
            optimizer (Optional[torch.optim.Optimizer]): 预留参数，便于子类扩展。

        费曼学习法:
            (A) 函数保存联邦训练所需的配置与模型引用。
            (B) 类比指挥部先确认作战简报与武器已就绪。
        """
        super().__init__(args, model, optimizer)

    def init_local_epochs(self):
        """返回客户端本地训练的轮次数量。

        返回:
            int: `self.args.local_epochs`。

        费曼学习法:
            (A) 告诉客户端需要训练几轮。
            (B) 类比教练规定每日训练圈数。
        """
        return self.args.local_epochs

    def get_local_update(self, **kwargs):
        """计算本地模型与当前全局模型之间的差值向量。

        参数:
            **kwargs: 需包含 `global_weights_vec`，即最新全局模型参数向量。

        返回:
            torch.Tensor: 本地模型参数向量与全局向量的差值。

        费曼学习法:
            (A) 函数把本地“新模型”与全局“旧模型”做减法，求出本地更新量。
            (B) 类比记录自己离团队平均水平的差距。
            (C) 步骤拆解:
                1. 取出全局参数向量。
                2. 获取本地模型参数向量。
                3. 相减得到本地更新量并返回。
            (D) 示例:
                >>> update = fedopt.get_local_update(global_weights_vec=vec)
            (E) 边界条件与测试建议: 确保传入关键字参数完整；测试更新量是否为零向量时模型相同。
        """
        global_weights_vec = kwargs['global_weights_vec']
        model_vec = model2vec_torch(self.model)
        if torch.is_tensor(global_weights_vec):
            global_vec = global_weights_vec.to(
                device=model_vec.device, dtype=model_vec.dtype
            )
        else:
            global_vec = torch.as_tensor(
                global_weights_vec, device=model_vec.device, dtype=model_vec.dtype
            )
        update = model_vec - global_vec
        return update

    def update(self, aggregated_update, **kwargs):
        """服务器端更新全局模型：将聚合更新量加到当前全局向量。

        参数:
            aggregated_update (torch.Tensor): 聚合后的更新向量。
            **kwargs: 需提供 `global_weights_vec`，即当前全局模型向量。

        返回:
            torch.Tensor: 更新后的全局模型向量。

        费曼学习法:
            (A) 将全局旧参数加上聚合更新，得到新的全局模型。
            (B) 类比团队根据所有成员的修正建议，更新统一流程。
            (C) 步骤拆解:
                1. 取出当前全局向量。
                2. 与聚合更新相加得到新权重。
                3. 调用 `vec2model` 写入模型参数。
                4. 返回新权重向量供日志或后续使用。
            (D) 示例:
                >>> new_vec = fedopt.update(aggregated_update=avg_update, global_weights_vec=current_vec)
            (E) 边界条件与测试建议: 向量维度必须匹配；建议验证更新前后参数差值与输入一致。
        """
        global_weights_vec = kwargs['global_weights_vec']
        if torch.is_tensor(global_weights_vec):
            global_vec = global_weights_vec
        else:
            global_vec = torch.as_tensor(
                global_weights_vec,
                device=aggregated_update.device,
                dtype=aggregated_update.dtype,
            )
        global_w = global_vec + aggregated_update
        vec2model(global_w, self.model)
        return global_w


# __AI_ANNOTATION_SUMMARY__
# 类 FedOpt: 联邦优化算法基础框架，输出差值更新并累加到全局模型。
# 方法 __init__: 保存配置与模型引用。
# 方法 init_local_epochs: 返回本地训练轮次。
# 方法 get_local_update: 返回本地模型与全局模型的差值向量。
# 方法 update: 将聚合更新加到全局模型并写回。
