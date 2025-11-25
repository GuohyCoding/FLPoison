# -*- coding: utf-8 -*-

import numpy as np
import torch

from attackers import attacker_registry
from attackers.pbases.mpbase import MPBase
from fl.client import Client
from global_utils import actor


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class PoisonedFL(MPBase, Client):
    """
    PyTorch reimplementation of the PoisonedFL attack.
    The attacker aligns a scaled fixed random direction with the residual
    global update while adaptively shrinking the scaling factor every 50 rounds
    when the aggregated direction drifts away from the fixed sign pattern.
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        # 沿用 MXNet 参考实现的默认放大系数，方便与原结果对齐；脚本参数可覆盖以做消融。
        self.default_attack_params = {"scaling_factor": 10000.0}  # scaling_factor设为8防止溢出
        self.update_and_set_attr()

        self.current_scaling_factor = float(self.scaling_factor)
        # 固定随机方向（±1），只在首次调用时生成，保持攻击方向全程一致以稳定扰动。
        self.fixed_rand = None
        # 记录初始/上一轮/最近 50 轮的全局模型向量，用于计算残差与漂移对齐度。
        self.init_model_vec = None
        self.prev_global_vec = None
        self.last_50_global_vec = None
        # 缓存上一轮的（潜在）恶意梯度，缺失时回退为良性更新，避免无历史信息时的震荡。
        self.last_grad_vec = None

    def omniscient(self, clients):
        attackers = [
            client for client in clients
            if client.category == "attacker"
        ]
        if not attackers:
            return None

        # 当前废播的全局模型向量，作为本轮计算 residual/漂移的基准。
        current_global_vec = torch.from_numpy(
            np.asarray(self.global_weights_vec, dtype=np.float32)
        ).flatten()

        # 首次进入时初始化固定方向与基准快照。
        if self.fixed_rand is None:
            # 与 MXNet 版本一致：随机符号向量，sign 保证只有 ±1。
            self.fixed_rand = torch.sign(torch.randn_like(current_global_vec))
            # 极小概率出现 0，用 where 转成 以保持符号稳定。
            zero_mask = self.fixed_rand == 0
            if torch.any(zero_mask):
                self.fixed_rand = torch.where(
                    zero_mask, torch.ones_like(self.fixed_rand), self.fixed_rand
                )
            self.init_model_vec = current_global_vec.clone()
            self.last_50_global_vec = current_global_vec.clone()

            # XXX
            print("fixed_rand", self.fixed_rand)
            print("fixed_rand.shape[0]", self.fixed_rand.shape[0])

        # history 为连续两轮全局模型的差值，等价于 MXNet 里的 current_model - last_model。
        history_vec = None
        if self.prev_global_vec is not None:
            history_vec = (current_global_vec - self.prev_global_vec).unsqueeze(1)

        # XXX：测试
        # print("prev_global_vec: ", self.prev_global_vec)
        # print("history_vec: ", history_vec)

        # 无论返回与否，都先更新 prev_global，保证下一轮有参照。
        self.prev_global_vec = current_global_vec.clone()

        # 缺少历史或上一轮恶意梯度时，先返回良性更新以维持数值稳定。
        if history_vec is None or self.last_grad_vec is None:
            benign_updates = np.stack(
                [np.array(client.update, copy=True) for client in attackers], axis=0
            ).astype(np.float32)
            self.last_grad_vec = torch.from_numpy(
                np.mean(benign_updates, axis=0)
            ).float()
            # 同步 50 轮快照，确保自适应缩放基于统一的全局节奏。
            if self.global_epoch % 50 == 0:
                self.last_50_global_vec = current_global_vec.clone()
            return benign_updates

        k_95, k_99 = self._get_thresholds(self.fixed_rand.numel())
        sf = float(self.current_scaling_factor)
        eps = 1e-9

        history_norm = torch.norm(history_vec)
        last_grad_norm = torch.norm(self.last_grad_vec)

        # 去掉上一轮梯度方向后得到残差，再按固定符号重新对齐，这对应文里的 fixed direction 投影。
        residual = history_vec.squeeze(1) - self.last_grad_vec * (
            history_norm / (last_grad_norm + eps)
        )
        scale = torch.norm(residual.unsqueeze(1), dim=1)
        deviation = scale * self.fixed_rand / (torch.norm(scale) + eps)
        # XXX：测试
        # print("deviation: ", deviation)

        current_epoch = int(self.global_epoch)
        if current_epoch % 50 == 0:
            # 每 50 轮检查一次固定方向的对齐度，偏移过大则指数衰减 scaling_factor，防止梯度漂移。
            total_update = current_global_vec - self.last_50_global_vec
            replaced = torch.where(total_update == 0, current_global_vec, total_update)
            current_sign = torch.sign(replaced)
            aligned_dim_cnt = int((current_sign == self.fixed_rand).sum().item())
            if aligned_dim_cnt < k_99 and sf * 0.7 >= 0.5:
                sf = sf * 0.7
            lamda_succ = sf * history_norm
        else:
            lamda_succ = sf * history_norm

        # 按固定方向生成恶意更新，并复制到所有攻击者，复用了良性 update 的形状以兼容聚合。
        mal_update = lamda_succ * deviation
        mal_update_np = mal_update.detach().cpu().numpy().astype(np.float32)
        malicious_updates = np.tile(mal_update_np, (len(attackers), 1))

        # 持久化状态，下一轮继续基于同一方向与缩放系数迭代。
        self.current_scaling_factor = sf
        self.last_grad_vec = torch.from_numpy(malicious_updates[0]).float()
        if current_epoch % 50 == 0:
            self.last_50_global_vec = current_global_vec.clone()

        # XXX：测试，恶意更新和良性更新对比
        # print("malicious_updates: ", malicious_updates)
        if current_epoch % 50 == 0:
            benign_clients = [c for c in clients if c.category != "attacker"]
            benign_updates = np.stack([np.array(c.update, copy=True) for c in benign_clients], axis=0)
            benign_norm = np.linalg.norm(benign_updates.mean(axis=0))
            cos = torch.nn.functional.cosine_similarity(
                mal_update.flatten(), history_vec.flatten(), dim=0
            ).item()
            print(
                f"[PoisonedFL debug] epoch={current_epoch}, sf={sf:.2e}, "
                f"hist_norm={history_norm.item():.2e}, mal_norm={torch.norm(mal_update).item():.2e}, "
                f"ratio_mal_hist={torch.norm(mal_update).item()/history_norm.item():.2e}, "
                f"ratio_mal_benign={torch.norm(mal_update).item()/(benign_norm+1e-9):.2e}, "
                f"cos_mal_hist={cos:.3f}"
            )

        return malicious_updates

    @staticmethod
    def _get_thresholds(dim):
        thresholds = {
            1204682: (603244, 603618),
            139960: (70288, 70415),
            717924: (359659, 359948),
            145212: (72919, 73049),
            61706: (31057, 31142),  # lenet MNIST 固定方向维度
        }
        if dim not in thresholds:
            raise NotImplementedError(
                f"Unsupported fixed_rand dimension {dim} for PoisonedFL thresholds."
            )
        return thresholds[dim]
