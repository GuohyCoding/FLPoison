"""
MeanNorm 聚合器：在聚合前对每个客户端更新按范数裁剪，再求平均。
思路来自原 MXNet 版本的 `mean_norm`，通过限制梯度能量缓解恶意放大导致的爆炸。
"""
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
import torch
from aggregators import aggregator_registry


@aggregator_registry
class MeanNorm(AggregatorBase):
    """
    对每个客户端更新执行 L2 范数裁剪，阈值取“推定良性”更新的平均范数，随后做均值聚合。
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        # eps 用于防止除零，可通过 defense_params 覆盖
        self.default_defense_params = {"eps": 1e-7}
        self.update_and_set_attr()

    def aggregate(self, updates, **kwargs):
        """
        参数:
            updates (np.ndarray | list[np.ndarray]): 形状 [num_clients, dim] 的更新矩阵。
        返回:
            np.ndarray: 裁剪后求均值的全局更新。
        """
        if torch.is_tensor(updates):
            if updates.numel() == 0:
                return updates
            num_clients = updates.shape[0]
            nfake = int(getattr(self.args, "num_adv", 0))
            nfake = max(0, min(nfake, num_clients - 1))
            norms = torch.linalg.norm(updates, dim=1, keepdim=True)
            if nfake > 0:
                sorted_norms, _ = torch.sort(norms.squeeze(-1))
                benign_norm = sorted_norms[: num_clients - nfake].mean()
            else:
                benign_norm = norms.mean()
            capped_norms = torch.minimum(norms, benign_norm)
            clipped_updates = updates * capped_norms / (norms + self.eps)
            return clipped_updates.mean(dim=0)

        updates = np.asarray(updates, dtype=np.float32)
        if updates.size == 0:
            return updates

        num_clients = updates.shape[0]
        
        # 取整后的攻击者数量，避免超过客户端总数
        nfake = int(getattr(self.args, "num_adv", 0))
        nfake = max(0, min(nfake, num_clients - 1))

        norms = np.linalg.norm(updates, axis=1, keepdims=True)

        # 参照 MXNet 版本：估计良性范数，剔除范数最大的 nfake 个更新
        if nfake > 0:
            sorted_idx = np.argsort(norms.squeeze(-1))
            benign_idx = sorted_idx[: num_clients - nfake]
            benign_norm = float(norms[benign_idx].mean())
        else:
            benign_norm = float(norms.mean())

        capped_norms = np.minimum(norms, benign_norm)
        clipped_updates = updates * capped_norms / (norms + self.eps)
        return clipped_updates.mean(axis=0)
