# -*- coding: utf-8 -*-

import torch
import numpy as np

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
        self.default_attack_params = {"scaling_factor": 10.0}  
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
        device = self.args.device
        if torch.is_tensor(self.global_weights_vec):
            current_global_vec = self.global_weights_vec.detach().flatten().to(device)
        else:
            current_global_vec = torch.as_tensor(
                self.global_weights_vec, dtype=torch.float32, device=device
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
            # self._log_message(f"fixed_rand[:100]={self.fixed_rand[:100].detach().cpu()}")

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
            benign_updates = torch.stack(
                [
                    client.update.detach().to(device)
                    if torch.is_tensor(client.update)
                    else torch.as_tensor(client.update, device=device)
                    for client in attackers
                ],
                dim=0,
            )
            self.last_grad_vec = benign_updates.mean(dim=0)
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
        malicious_updates = mal_update.detach().unsqueeze(0).repeat(len(attackers), 1)

        # if current_epoch % 20 == 0:
        #     # XXX:测试
        #     mal_str = np.array2string(
        #         mal_update[:100].detach().cpu().numpy(),
        #         separator=" ",
        #         max_line_width=1000,
        #         formatter={"float_kind": lambda x: f"{x:.2e}"},
        #     )
        #     self._log_message(f"mal_update={mal_str}")
        #     update = attackers[0].update
        #     if torch.is_tensor(update):
        #         attacker_update = update.detach().cpu().numpy()
        #     else:
        #         attacker_update = np.array(update, copy=True)
        #     benign_update_norm = float(np.linalg.norm(attacker_update))
        #     benign_str = np.array2string(
        #         attacker_update[:100],
        #         separator=" ",
        #         max_line_width=1000,
        #         formatter={"float_kind": lambda x: f"{x:.2e}"},
        #     )
        #     self._log_message(f"benign_update={benign_str}")
        #     mal_update_norm = float(mal_update.norm().item())
        #     ratio = mal_update_norm / (benign_update_norm + 1e-12)
        #     self._log_message(f"mal_benign_l2_ratio={ratio}")
        #     self._log_message(f"lamda_succ={lamda_succ}")

        # 持久化状态，下一轮继续基于同一方向与缩放系数迭代。
        self.current_scaling_factor = sf
        self.last_grad_vec = malicious_updates[0].detach().clone()
        if current_epoch % 50 == 0:
            self.last_50_global_vec = current_global_vec.clone()

        return malicious_updates

    def _log_message(self, msg):
        logger = getattr(self.args, "logger", None)
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)   

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
