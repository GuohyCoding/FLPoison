"""
Bulyan 聚合器：通过两阶段筛选实现拜占庭鲁棒的坐标级聚合。

该实现遵循论文《The Hidden Vulnerability of Distributed Learning in Byzantium》，
先使用 Krum 等鲁棒聚合器选出候选集合，再在候选集合上进行坐标级 beta-closest-median
筛选，最终平均求得聚合结果。
"""
from aggregators.aggregatorbase import AggregatorBase
from aggregators.krum import krum
import numpy as np
import torch
from aggregators import aggregator_registry


@aggregator_registry
class Bulyan(AggregatorBase):
    """
    Bulyan 聚合器：先以 Krum 选出候选集合，再进行坐标级 beta-closest-median 聚合。

    第一阶段通过 Krum（或其他鲁棒规则）逐步挑选出一个稳定的候选客户端集合；
    第二阶段对该集合的每个坐标依次选择距离坐标中位数最近的 beta 个值，最后对筛选后更新取均值。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 Bulyan 聚合器，配置默认参数并计算 beta 值。

        参数:
            args (argparse.Namespace | Any): 运行配置对象，需要包含
                - num_clients (int): 客户端总数。
                - num_adv (int): 恶意客户端数量估计。
                - defense_params (dict, optional): 用于覆写默认防御参数。
            **kwargs: 预留的额外参数，当前未使用。

        返回:
            None

        异常:
            AttributeError: 当 args 缺少上述必须字段时抛出。

        复杂度:
            时间复杂度 O(1)；空间复杂度 O(1)。
        """
        super().__init__(args)
        """
        enable_check (bool): 是否启用 4f+3 ≤ n 的条件校验，启用后若条件不满足将抛出异常。
        """
        # 设定默认防御参数，允许通过 args.defense_params 覆写。
        self.default_defense_params = {"enable_check": False}
        self.update_and_set_attr()

        # 在已有恶意客户端数量估计的前提下，计算 beta（第二阶段可保留的元素个数）。
        self.beta = self.args.num_clients - 2 * self.args.num_adv

    def aggregate(self, updates, **kwargs):
        """
        执行 Bulyan 聚合流程，对客户端更新进行两阶段鲁棒筛选并求均值。

        参数:
            updates (numpy.ndarray): 客户端上传的更新矩阵，形状 (n, d)。
            **kwargs: 预留参数，当前实现未使用。

        返回:
            numpy.ndarray: 聚合后的更新向量，形状为 (d,)。

        异常:
            ValueError: 当启用条件检查且 4f+3 > n，或 Krum 条件失效时抛出。
            Exception: Krum 函数可能引发的其他异常会向上传递。

        复杂度:
            时间复杂度约 O(n^2 d)，其中 Krum 选择阶段 O(n^2 d)，
            坐标筛选阶段 O(n d log n)；空间复杂度 O(n d)。
        """
        # 可选条件检查：确保满足 4f+3 ≤ n，否则理论保证失效。
        if self.enable_check:
            if 4*self.args.num_adv + 3 > self.args.num_clients:
                raise ValueError(
                    f"num_adv should be meet 4f+3 <= n, got {4*self.args.num_adv+3} > {self.args.num_clients}.")

        # 1. 利用 Krum 迭代选择候选集合
        set_size = self.args.num_clients - 2 * self.args.num_adv
        set_size = max(1, min(len(updates), set_size))
        selected_idx = []
        if torch.is_tensor(updates):
            available = torch.arange(len(updates), device=updates.device)
            while len(selected_idx) < set_size:
                try:
                    remaining = updates[available]
                    krum_idx = krum(
                        remaining, self.args.num_adv, return_index=True
                    )
                except ValueError:
                    if len(selected_idx) > 0:
                        break
                    raise
                except Exception as e:
                    raise e
                chosen = int(available[krum_idx].item())
                selected_idx.append(chosen)
                mask = available != chosen
                available = available[mask]
            selected_idx = torch.as_tensor(selected_idx, device=updates.device, dtype=torch.long)
        else:
            while len(selected_idx) < set_size:
                try:
                    # 从剩余更新中执行 Krum，取得最可信客户端索引。
                    krum_idx = krum(np.delete(
                        updates, selected_idx, axis=0), self.args.num_adv, return_index=True)
                except ValueError:
                    # 若 Krum 条件不再满足，则在已有候选基础上停止；若无候选则继续抛出异常。
                    if len(selected_idx) > 0:
                        break
                    else:
                        raise
                except Exception as e:
                    # 将其他异常直接上抛，便于调用方定位问题。
                    raise e
                # 将选中的索引加入候选集合，准备下一轮挑选。
                selected_idx.append(krum_idx)
            # 将候选索引转换为 NumPy 数组，便于后续索引操作。
            selected_idx = np.array(selected_idx, dtype=np.int64)

        # 若 beta 等于客户端总数或候选数量，直接使用候选集合。
        if len(selected_idx) == 0:
            if torch.is_tensor(updates):
                selected_idx = torch.arange(
                    min(1, len(updates)), device=updates.device, dtype=torch.long
                )
            else:
                selected_idx = np.arange(min(1, len(updates)), dtype=np.int64)

        effective_beta = max(1, min(self.beta, len(selected_idx)))

        if effective_beta == self.args.num_clients or effective_beta == len(selected_idx):
            bening_updates = updates[selected_idx]
        else:
            # 2. 在候选集合上执行坐标级 beta-closest-median 筛选
            # notes: 若改用其他坐标级聚合方式（如 trimmed mean）可在此替换。
            if torch.is_tensor(updates):
                median = torch.median(updates[selected_idx], dim=0).values
                abs_dist = torch.abs(updates[selected_idx] - median)
                _, beta_idx = torch.topk(
                    abs_dist, effective_beta, dim=0, largest=False
                )
                bening_updates = torch.gather(
                    updates[selected_idx], dim=0, index=beta_idx
                )
            else:
                median = np.median(updates[selected_idx], axis=0)
                abs_dist = np.abs(updates[selected_idx] - median)
                # 对每个坐标选取与中位数距离最小的 beta 个元素。
                beta_idx = np.argpartition(
                    abs_dist, effective_beta, axis=0)[:effective_beta]
                bening_updates = np.take_along_axis(
                    updates[selected_idx], beta_idx, axis=0)
        # 对筛选后的更新取均值，得到最终聚合结果。
        if torch.is_tensor(bening_updates):
            return torch.mean(bening_updates, dim=0)
        return np.mean(bening_updates, axis=0)


# 费曼学习法解释（Bulyan.__init__）
# (A) 功能概述：初始化 Bulyan 聚合器并计算两阶段筛选所需的 beta 常数。
# (B) 类比说明：像设置一套筛选机制，先确定允许通过的候选人数，再准备筛选工具。
# (C) 逐步拆解：
#     1. 调用父类构造函数保存全局配置。
#     2. 设定默认防御参数并允许用户覆写。
#     3. 依据客户端总数与恶意估计数计算 beta = n - 2f。
# (D) 最小示例：
#     >>> class Args:
#     ...     num_clients = 10
#     ...     num_adv = 2
#     ...     defense_params = None
#     >>> bulyan = Bulyan(Args())
#     >>> bulyan.beta
#     6
# (E) 边界条件与测试建议：
#     - 缺失 num_clients 或 num_adv 会导致 AttributeError。
#     - 建议测试：1) 默认参数正确注入；2) 用户开启 enable_check 时条件校验是否生效。
# (F) 背景参考：
#     - 背景：Bulyan 属于鲁棒聚合框架的经典算法。
#     - 推荐阅读：《The Hidden Vulnerability of Distributed Learning in Byzantium》《Federated Learning》。


# 费曼学习法解释（Bulyan.aggregate）
# (A) 功能概述：执行 Krum 选集与坐标级筛选，输出鲁棒聚合结果。
# (B) 类比说明：像先请专家初筛合格作品，再让评委挑选最接近平均风格的作品参与最终评审。
# (C) 逐步拆解：
#     1. 可选的条件检查，确保理论前提成立。
#     2. 初始化候选集合大小 set_size = n - 2f。
#     3. 迭代调用 Krum，从剩余客户端中选出最可信者加入候选集合。
#     4. 若 Krum 无法继续（条件不满足）且已有候选，提前终止。
#     5. 根据 beta 判断是否需要坐标级筛选。
#     6. 若需要，计算候选集合的坐标中位数，选出距离最近的 beta 个值。
#     7. 对最终保留的更新求均值，得到聚合结果。
# (D) 最小示例：
#     >>> import numpy as np
#     >>> class Args: num_clients=5; num_adv=1; defense_params=None
#     >>> bulyan = Bulyan(Args())
#     >>> updates = np.random.randn(5, 3)
#     >>> result = bulyan.aggregate(updates)
#     >>> result.shape
#     (3,)
# (E) 边界条件与测试建议：
#     - 若 num_adv 估计值过大，Krum 可能在早期即抛出异常。
#     - 建议测试：1) 纯良性场景下结果与均值接近；2) 插入明显异常向量时能否被过滤。
# (F) 背景参考：
#     - 背景：Bulyan 结合 Krum 等方法，适用于非独立同分布数据中的拜占庭防御。
#     - 推荐阅读：《The Hidden Vulnerability of Distributed Learning in Byzantium》《Robust Statistics》。


__AI_ANNOTATION_SUMMARY__ = """
Bulyan.__init__: 初始化默认参数并计算 beta 常数以支撑 Bulyan 双阶段筛选流程。
Bulyan.aggregate: 结合 Krum 选集与坐标级 beta-closest-median 筛选完成鲁棒聚合。
"""
