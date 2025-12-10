# -*- coding: utf-8 -*-

import numpy as np
import torch

from attackers import attacker_registry
from attackers.pbases.mpbase import MPBase
from fl.client import Client
from global_utils import actor


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class DMRCA(MPBase, Client):
    """
    Dynamic Multi-Round Consistency Attack.
    Tracks global training direction during a warm-up stage, then injects
    dynamically smoothed reverse updates aligned with both global and local
    anti-directions.
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {
            "warmup_rounds": 30,
            "beta": 0.2,           # EMA for training direction during warm-up
            "cos_window": 5,
            "cos_threshold": 0.8,
            "cos_epsilon": 0.05,

            "gamma": 0.02,         # dynamic EMA update for attack direction
            "rho": 0.2,            # pull-back to static direction

            "a_global": 1.0,       # weight for global direction
            "b_local": 0.5,        # weight for local negative direction

            "alpha": 1.0,          # attack scale
            "lambda_attack": 0.7   # attack/benign mixing
        }
        self.update_and_set_attr()

        self.global_round = 0
        self.prev_global_weights = None
        self.train_dir_ema = None
        self.prev_u = None
        self.cos_history = []
        self.warmup_done = False
        self.s_static = None
        self.s_attack_prev = None

    def omniscient(self, clients):
        # Update attack state using global model trajectory; no direct mutation.
        self.global_round += 1

        w_flat = torch.as_tensor(
            np.asarray(self.global_weights_vec, dtype=np.float32),
            device=self.args.device
        ).flatten()

        if self.prev_global_weights is None:
            self.prev_global_weights = w_flat.clone()
            return None

        d_t = w_flat - self.prev_global_weights
        d_norm = torch.norm(d_t)
        if d_norm <= 0:
            self.prev_global_weights = w_flat.clone()
            return None
        u_t = d_t / d_norm

        if self.train_dir_ema is None:
            v = u_t.clone()
        else:
            v = (1 - float(self.beta)) * self.train_dir_ema + float(self.beta) * u_t

        v_unit = self._normalize(v)
        cos_t = torch.dot(u_t, v_unit).item()
        self.cos_history.append(cos_t)
        if len(self.cos_history) > int(self.cos_window):
            self.cos_history = self.cos_history[-int(self.cos_window):]

        if not self.warmup_done:
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
            return None

        v_raw = -d_t / (d_norm + 1e-12)
        if torch.dot(v_raw, self.s_attack_prev) < 0:
            v_raw = -v_raw

        s_dyn = self._normalize((1 - float(self.gamma)) * self.s_attack_prev + float(self.gamma) * v_raw)
        s_attack = self._normalize((1 - float(self.rho)) * s_dyn + float(self.rho) * self.s_static)
        self.s_attack_prev = s_attack

        self.prev_global_weights = w_flat.clone()
        self.train_dir_ema = v
        self.prev_u = u_t
        return None

    def non_omniscient(self):
        delta = self.update
        delta_flat = torch.as_tensor(delta, device=self.args.device).flatten().float()
        delta_norm = torch.norm(delta_flat)

        if (not self.warmup_done) or self.s_attack_prev is None or delta_norm <= 0:
            return delta

        s_local = -delta_flat / delta_norm
        s_global = self.s_attack_prev

        s_combined = self._normalize(
            float(self.a_global) * s_global + float(self.b_local) * s_local
        )
        if torch.norm(s_combined) <= 0:
            return delta

        g_attack = float(self.alpha) * s_combined
        g_attack_norm = torch.norm(g_attack)
        if g_attack_norm > 0:
            g_attack = g_attack * (delta_norm / g_attack_norm)

        new_update_flat = float(self.lambda_attack) * g_attack + (1 - float(self.lambda_attack)) * delta_flat
        return self._reshape_like(new_update_flat, delta)

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
