# -*- coding: utf-8 -*-

import copy
import numpy as np
import torch
from torch.nn.utils import prune

from attackers import attacker_registry
from attackers.pbases.mpbase import MPBase
from fl.client import Client
from global_utils import actor


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class TMRC(MPBase, Client):
    """
    Targeted Multi-Round Consistency attack.
    Reuses PoisonedFL's fixed random direction but only perturbs weights marked
    as important each round to keep the attack focused on salient parameters.
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        # 继承 Client 以复用联邦训练和全局权重读取的能力；MPBase 提供攻击配置更新等基础功能
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {
            "scaling_factor": 10000.0,
            # ratio of parameters considered important (top-|w|); defaults to 10%
            "importance_ratio": 0.8,
            # cap malicious norm to at most X times benign mean norm to avoid exploding losses
            "max_norm_ratio": 5.0,
            # absolute ceiling on malicious vector norm regardless of benign scale
            "max_attack_norm": 500.0,
            # benign rounds required before fixing the anchor norm (median of first k rounds)
            "anchor_warmup_rounds": 5,
        }
        self.update_and_set_attr()

        # 当前攻击强度，可在多轮中根据对齐情况自适应调整
        self.current_scaling_factor = float(self.scaling_factor)
        # 固定随机方向（元素为±1），只生成一次，确保多轮攻击的方向一致，便于累积偏移
        self.fixed_rand = None
        # 多个全局模型快照：用于估计残差与历史漂移
        self.init_model_vec = None
        self.prev_global_vec = None
        self.last_50_global_vec = None
        # 缓存上一轮的恶意梯度，在缺失时退回 benign 均值
        self.last_grad_vec = None
        # 缓存上一轮的重要参数掩码，便于比较掩码方向变化
        self.prev_importance_mask = None
        # 记录早期 benign 更新范数，锚定攻击尺度，防止攻击跟随自身放大而失控
        self.benign_norm_anchor = None
        self._anchor_samples = []

    def omniscient(self, clients):
        # 只对攻击者客户端注入恶意更新，其他客户端保持正常
        benign_clients = [c for c in clients if c.category != "attacker"]
        attackers = [
            client for client in clients
            if client.category == "attacker"
        ]
        if not attackers:
            return None

        current_global_vec = torch.from_numpy(
            np.asarray(self.global_weights_vec, dtype=np.float32)
        ).flatten()

        if self.fixed_rand is None:
            # 首次调用时确定固定方向，并记录基线模型；方向固定可避免每轮随机方向导致扰动抵消
            self.fixed_rand = torch.sign(torch.randn_like(current_global_vec))
            # 如果有0则替换为1
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

        # Recompute important-weight mask every round to stay targeted
        importance_mask = self._compute_importance_mask(current_global_vec)

        # XXX: show proportion of active weights
        nz = int((importance_mask != 0).sum().item())
        total = importance_mask.numel()
        print(f"[TMRC debug] importance_mask active {nz}/{total} ({nz/total:.4f})")
        if self.prev_importance_mask is not None:
            mask_cos = self._compare_direction(importance_mask, self.prev_importance_mask)
            print(f"[TMRC debug] importance_mask cosine vs prev={mask_cos:.4f}")
        self.prev_importance_mask = importance_mask.detach().clone()


        history_vec = None
        if self.prev_global_vec is not None:
            # history_vec 表示最近一轮的全局参数增量，既是模型演化方向的估计，也是残差计算基准
            history_vec = (current_global_vec - self.prev_global_vec).unsqueeze(1)

            # XXX: compare direction with previous global model
            cos_val = self._compare_direction(current_global_vec, self.prev_global_vec)
            print(f"[TMRC debug] global direction cosine={cos_val:.4f}")

        # 更新上一轮的全局快照，便于下一次计算 history_vec
        self.prev_global_vec = current_global_vec.clone()

        if history_vec is None or self.last_grad_vec is None:
            # 没有历史梯度或残差时，无法构造对齐扰动，先返回攻击者的 benign 均值，避免异常发散
            benign_updates = np.stack(
                [np.array(client.update, copy=True) for client in attackers], axis=0
            ).astype(np.float32)
            self.last_grad_vec = torch.from_numpy(
                np.mean(benign_updates, axis=0)
            ).float()
            if self.global_epoch % 50 == 0:
                self.last_50_global_vec = current_global_vec.clone()
            return benign_updates

        benign_norm_value = 0.0
        if benign_clients:
            benign_updates_np = np.stack(
                [np.array(client.update, copy=True) for client in benign_clients], axis=0
            ).astype(np.float32)
            benign_mean = np.mean(benign_updates_np, axis=0)
            benign_norm_value = float(np.linalg.norm(benign_mean))
            if benign_norm_value > 0:
                self._update_benign_anchor(benign_norm_value)
        active_dim = int(importance_mask.sum().item())
        k_99 = max(1, int(active_dim * 0.99))
        sf = float(self.current_scaling_factor)
        eps = 1e-9

        # 估计历史方向与上一轮梯度的尺度，用于正交残差构造
        history_norm = torch.norm(history_vec)
        last_grad_norm = torch.norm(self.last_grad_vec)

        # 残差 = 真实历史方向 - （对齐到历史范数的上一轮梯度），再用重要掩码筛选
        residual = history_vec.squeeze(1) - self.last_grad_vec * (
            history_norm / (last_grad_norm + eps)
        )
        residual = residual * importance_mask
        # 只在重要维度上做缩放，direction 使用固定随机向量确保方向稳定
        scale = torch.norm(residual.unsqueeze(1), dim=1)
        masked_direction = self.fixed_rand * importance_mask
        deviation = scale * masked_direction / (torch.norm(scale) + eps)

        current_epoch = int(self.global_epoch)
        if current_epoch % 50 == 0:
            # 每 50 轮检查一次全局更新方向与固定方向的对齐程度，若对齐差则衰减 scaling factor，防止过度偏移
            total_update = current_global_vec - self.last_50_global_vec
            replaced = torch.where(total_update == 0, current_global_vec, total_update)
            current_sign = torch.sign(replaced)
            aligned_dim_cnt = int(
                (current_sign[importance_mask.bool()] == self.fixed_rand[importance_mask.bool()]).sum().item()
            )
            if aligned_dim_cnt < k_99 and sf * 0.7 >= 0.5:
                sf = sf * 0.7
            lamda_succ = sf * history_norm
        else:
            lamda_succ = sf * history_norm

        max_ratio = max(float(getattr(self, "max_norm_ratio", 0.0)), 0.0)
        max_attack_norm = float(getattr(self, "max_attack_norm", 0.0))
        base_norm_for_cap = self.benign_norm_anchor
        if (base_norm_for_cap is None or base_norm_for_cap <= 0) and benign_norm_value > 0:
            base_norm_for_cap = benign_norm_value

        cap_value = None
        if max_ratio > 0.0 and base_norm_for_cap is not None and base_norm_for_cap > 0.0:
            cap_value = base_norm_for_cap * max_ratio
        if max_attack_norm > 0.0:
            cap_value = (
                max_attack_norm if cap_value is None else min(cap_value, max_attack_norm)
            )

        clipped = False
        if cap_value is not None and cap_value > 0.0:
            lamda_cap = torch.tensor(
                cap_value,
                dtype=lamda_succ.dtype,
                device=lamda_succ.device,
            )
            if lamda_succ > lamda_cap:
                lamda_succ = lamda_cap
                clipped = True

        if clipped and history_norm.item() > eps:
            sf = lamda_succ.item() / (history_norm.item() + eps)
            self.current_scaling_factor = sf

        # 按残差方向生成恶意更新（仅在重要参数上放大），其余维度将被 benign 均值覆盖
        mal_update = lamda_succ * deviation
        mal_update_np = mal_update.detach().cpu().numpy().astype(np.float32)

        # Merge: keep benign updates on non-important weights, only overwrite important ones
        benign_attacker_updates = np.stack(
            [np.array(client.update, copy=True) for client in attackers], axis=0
        ).astype(np.float32)
        benign_base = np.mean(benign_attacker_updates, axis=0)
        combined_update = benign_base
        # 只在重要索引处替换为恶意更新，其余保持 benign，减少被鲁棒聚合检测到的风险
        important_idx = importance_mask.detach().cpu().numpy().astype(bool)
        combined_update[important_idx] = mal_update_np[important_idx]

        # 所有攻击者共享同一份恶意更新，形成协同攻击
        malicious_updates = np.tile(combined_update, (len(attackers), 1))

        self.current_scaling_factor = sf
        self.last_grad_vec = torch.from_numpy(malicious_updates[0]).float()
        if current_epoch % 50 == 0:
            self.last_50_global_vec = current_global_vec.clone()

        if current_epoch % 50 == 0:
            cos = torch.nn.functional.cosine_similarity(
                mal_update.flatten(), history_vec.flatten(), dim=0
            ).item()
            print(
                f"[TMRC debug] epoch={current_epoch}, sf={sf:.2e}, "
                f"hist_norm={history_norm.item():.2e}, mal_norm={torch.norm(mal_update).item():.2e}, "
                f"ratio_mal_hist={torch.norm(mal_update).item()/history_norm.item():.2e}, "
                f"ratio_mal_benign={torch.norm(mal_update).item()/(benign_norm_value+1e-9):.2e}, "
                f"cos_mal_hist={cos:.3f}"
            )

        return malicious_updates

    def _compare_direction(self, vec_a, vec_b):
        """
        Compare cosine similarity between two vectors after flattening.
        Returns a float cosine value.
        """
        flat_a = vec_a.flatten()
        flat_b = vec_b.flatten()
        return torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=0).item()

    def _compute_importance_mask(self, weight_vec):
        """
        Build a binary mask via PyTorch prune: per-layer L1 unstructured pruning.
        Keep top-|w| proportion (importance_ratio) for weight tensors; other
        parameters (bias/BN stats) default to zero in the mask.
        """
        ratio = float(self.importance_ratio)
        ratio = min(max(ratio, 0.0), 1.0)
        amount = 1.0 - ratio  # portion to prune per layer

        # Work on a model copy to avoid polluting the real model with pruning hooks
        tmp_model = copy.deepcopy(self.model)
        # Load current global weights to align masks with current model state
        from fl.models.model_utils import vec2model  # local import to avoid cycles

        vec2model(weight_vec.detach().cpu().numpy(), tmp_model)

        # 使用 L1 非结构化剪枝得到“重要权重”集合：越大越不易被剪掉，从而保留 high-|w|
        # 这样每轮根据当前模型动态筛选，保证攻击集中在最敏感的参数上
        # Apply unstructured L1 pruning to each module that owns a weight tensor
        for module in tmp_model.modules():
            if hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
                if module.weight is not None and module.weight.numel() > 0:
                    prune.l1_unstructured(module, name="weight", amount=amount)

        pruned_state = tmp_model.state_dict()

        # Rebuild a flat mask aligned with the original state_dict order (global vector order)
        mask_chunks = []
        for name, tensor in self.model.state_dict().items():
            if name.endswith("weight"):
                mask_name = f"{name}_mask"
                if mask_name in pruned_state:
                    mask_tensor = pruned_state[mask_name]
                else:
                    mask_tensor = torch.ones_like(torch.as_tensor(tensor))
            else:
                # Non-weight params (bias/BN stats) stay unperturbed
                mask_tensor = torch.zeros_like(torch.as_tensor(tensor))
            mask_chunks.append(mask_tensor.flatten())

        mask_vec = torch.cat(mask_chunks).to(weight_vec.device, dtype=weight_vec.dtype)
        if mask_vec.sum() <= 0:
            # ensure at least one dimension is active
            mask_vec[0] = 1.0
        return mask_vec

    def _update_benign_anchor(self, benign_norm_value):
        """
        Track an early benign norm anchor so attack magnitude does not
        snowball with the diverging global training dynamics.
        """
        warmup = max(int(getattr(self, "anchor_warmup_rounds", 5)), 1)
        self._anchor_samples.append(float(benign_norm_value))
        if len(self._anchor_samples) > warmup:
            self._anchor_samples = self._anchor_samples[-warmup:]

        if self.benign_norm_anchor is None and len(self._anchor_samples) >= warmup:
            self.benign_norm_anchor = float(np.median(self._anchor_samples))
        elif self.benign_norm_anchor is not None and benign_norm_value < self.benign_norm_anchor:
            # allow anchor to decrease smoothly if benign updates shrink again
            self.benign_norm_anchor = 0.9 * self.benign_norm_anchor + 0.1 * benign_norm_value
