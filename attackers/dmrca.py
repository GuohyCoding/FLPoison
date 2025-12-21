# -*- coding: utf-8 -*-

import numpy as np
import torch

from attackers import attacker_registry
from attackers.pbases.mpbase import MPBase
from fl.client import Client
from global_utils import actor


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient', 'non_omniscient')
class DMRCA(MPBase, Client):
    """
    Dynamic Multi-Round Consistency Attack.
    Tracks global training direction during a warm-up stage, then injects
    dynamically smoothed reverse updates aligned with both global and local
    anti-directions.
    中文提示：先用若干轮“观察期”估计全局训练方向是否稳定，再在稳定后按全局+本地的组合反方向投毒。
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {
            "warmup_rounds": 10,   # 观测/热身轮数：仅记录方向不投毒，越大越稳但起攻越晚
            "beta": 0.1,           # 热身期方向的 EMA 权重：偏大=更贴近最新方向，偏小=更平滑
            "cos_window": 5,       # 判断方向稳定性的滑动窗口长度（最近几轮余弦相似度）
            "cos_threshold": 0.6,  # 视为“方向一致”的余弦下限，越高越难通过稳定性检测
            "cos_epsilon": 0.1,    # 稳定性检测的波动容忍度：窗口内 cos 差值需小于该值

            "gamma": 0.05,         # 动态攻击方向的 EMA 权重：越大越追随当前全局反方向，越小越平滑
            "rho": 0.1,            # 动态方向向静态方向的“拉回”强度：大则更依赖热身确定的静态方向

            "a_global": 1.0,       # 全局攻击方向权重：增大可提升各攻击者之间的一致性
            "b_local": 0.1,        # 本地反方向权重：增大可更贴合自身梯度反方向（针对性更强）

            "alpha": 80000.0,      # 攻击尺度：决定恶意方向放大倍数，越大投毒越猛烈
            "lambda_attack": 1.0   # 恶意/良性混合比例：1 表示全恶意，0 表示全良性
        }
        self.update_and_set_attr()

        # ===== 内部状态 =====
        self.global_round = 0           # 已记录的全局轮次
        self.prev_global_weights = None # 上一轮全局权重向量，便于计算差分
        self.train_dir_ema = None       # 训练方向的指数滑动平均
        self.prev_u = None              # 上一轮单位更新向量 u_t（调试/扩展用）
        self.cos_history = []           # 近期余弦相似度窗口
        self.warmup_done = False        # 是否完成热身期
        self.s_static = None            # 热身期确定的“静态”反方向
        self.s_attack_prev = None       # 上一轮使用的动态攻击方向

    def omniscient(self, clients):
        # 仅依据全局权重轨迹更新内部状态，并返回要分发的（恶意或良性）更新
        attackers = [
            client for client in clients
            if client.category == "attacker"
        ]
        if not attackers:
            return None

        self.global_round += 1

        # 拉平成 torch 向量，后续保持 device 对齐
        w_flat = torch.as_tensor(
            np.asarray(self.global_weights_vec, dtype=np.float32),
            device=self.args.device
        ).flatten()

        # 首轮没有上一轮参考，直接缓存并退出
        if self.prev_global_weights is None:
            self.prev_global_weights = w_flat.clone()
            benign_updates = np.stack(
                [np.array(client.update, copy=True) for client in attackers], axis=0
            ).astype(np.float32)
            return benign_updates

        # 当前全局更新差分 d_t 及其单位向量 u_t
        d_t = w_flat - self.prev_global_weights
        d_norm = torch.norm(d_t)
        if d_norm <= 0:
            self.prev_global_weights = w_flat.clone()
            benign_updates = np.stack(
                [np.array(client.update, copy=True) for client in attackers], axis=0
            ).astype(np.float32)
            return benign_updates
        u_t = d_t / d_norm

        # 训练方向 EMA：热身期用于判断稳定性
        if self.train_dir_ema is None:
            v = u_t.clone()
        else:
            v = (1 - float(self.beta)) * self.train_dir_ema + float(self.beta) * u_t

        # 记录余弦相似度用于稳定性检测
        v_unit = self._normalize(v)
        cos_t = torch.dot(u_t, v_unit).item()
        self.cos_history.append(cos_t)
        if len(self.cos_history) > int(self.cos_window):
            self.cos_history = self.cos_history[-int(self.cos_window):]

        if not self.warmup_done:
            # 热身：若窗口内余弦值足够大且波动足够小，则锁定静态反方向
            if len(self.cos_history) >= int(self.cos_window):
                cos_min = min(self.cos_history)
                cos_max = max(self.cos_history)
                stable = cos_min >= float(self.cos_threshold) and (cos_max - cos_min) <= float(self.cos_epsilon)
                if stable and self.global_round >= int(self.warmup_rounds):
                    self.s_static = -self._normalize(v)
                    self.s_attack_prev = self.s_static.clone()
                    self.warmup_done = True

            self.prev_global_weights = w_flat.clone()
            self.train_dir_ema = v
            self.prev_u = u_t
            benign_updates = np.stack(
                [np.array(client.update, copy=True) for client in attackers], axis=0
            ).astype(np.float32)
            return benign_updates

        # 进入攻击期：取当前全局更新的反方向，必要时翻转以保持与上一轮方向一致
        v_raw = -d_t / (d_norm + 1e-12)
        if torch.dot(v_raw, self.s_attack_prev) < 0:
            v_raw = -v_raw

        # 动态方向平滑，再与静态方向做“拉回”以防漂移
        s_dyn = self._normalize((1 - float(self.gamma)) * self.s_attack_prev + float(self.gamma) * v_raw)
        s_attack = self._normalize((1 - float(self.rho)) * s_dyn + float(self.rho) * self.s_static)
        self.s_attack_prev = s_attack

        self.prev_global_weights = w_flat.clone()
        self.train_dir_ema = v
        self.prev_u = u_t
        # 基于最新的动态攻击方向，构造并返回每个攻击者的更新
        malicious_updates = []
        for client in attackers:
            benign_update = client.update
            malicious_update = self._craft_malicious_update(benign_update)
            malicious_updates.append(malicious_update)
        return np.stack(malicious_updates, axis=0).astype(np.float32)

    def non_omniscient(self):
        # benign 本地更新（未改写前）
        delta = self.update
        delta_flat = torch.as_tensor(delta, device=self.args.device).flatten().float()
        delta_norm = torch.norm(delta_flat)

        # 热身未结束或无方向信息，则保持 benign
        if (not self.warmup_done) or self.s_attack_prev is None or delta_norm <= 0:
            return delta

        # 本地反方向 + 全局动态攻击方向，线性组合
        s_local = -delta_flat / delta_norm
        s_global = self.s_attack_prev

        s_combined = self._normalize(
            float(self.a_global) * s_global + float(self.b_local) * s_local
        )
        if torch.norm(s_combined) <= 0:
            return delta

        # 生成攻击向量，不按 benign 更新范数缩放
        g_attack = float(self.alpha) * s_combined

        # 攻击向量与 benign 混合，提升隐蔽性
        new_update_flat = float(self.lambda_attack) * g_attack + (1 - float(self.lambda_attack)) * delta_flat
        return self._reshape_like(new_update_flat, delta)

    def _craft_malicious_update(self, delta):
        """
        给定单个攻击者的 benign 更新，依据当前全局/本地方向生成恶意更新。
        """
        delta_flat = torch.as_tensor(delta, device=self.args.device).flatten().float()
        delta_norm = torch.norm(delta_flat)

        if (not self.warmup_done) or self.s_attack_prev is None or delta_norm <= 0:
            return np.array(delta, copy=True)

        s_local = -delta_flat / delta_norm
        s_global = self.s_attack_prev

        s_combined = self._normalize(
            float(self.a_global) * s_global + float(self.b_local) * s_local
        )
        if torch.norm(s_combined) <= 0:
            return np.array(delta, copy=True)

        # 保持攻击幅度：不按 benign 更新范数缩放
        g_attack = float(self.alpha) * s_combined

        new_update_flat = float(self.lambda_attack) * g_attack + (1 - float(self.lambda_attack)) * delta_flat
        crafted = self._reshape_like(new_update_flat, delta)
        return np.asarray(crafted, dtype=np.float32)

    def _normalize(self, vec, eps=1e-12):
        norm = torch.norm(vec)
        if norm <= eps:
            return torch.zeros_like(vec)
        return vec / norm

    def _reshape_like(self, flat_tensor, reference):
        if isinstance(reference, torch.Tensor):
            reshaped = flat_tensor.reshape(reference.shape)
            return reshaped.to(device=reference.device, dtype=reference.dtype)

        ref_arr = np.asarray(reference)
        return flat_tensor.detach().cpu().numpy().astype(ref_arr.dtype).reshape(ref_arr.shape)
