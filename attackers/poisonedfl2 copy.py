# -*- coding: utf-8 -*-

import functools
import numpy as np
import torch
import math
from captum.attr import Saliency
try:
    from torch.func import functional_call
except Exception:  # 兼容较旧 PyTorch 版本
    from torch.nn.utils.stateless import functional_call

from attackers import attacker_registry
from attackers.pbases.mpbase import MPBase
from fl.client import Client
from global_utils import actor


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class PoisonedFL2(MPBase, Client):
    """
    PyTorch reimplementation of the PoisonedFL attack.
    The attacker aligns a scaled fixed random direction with the residual
    global update while adaptively shrinking the scaling factor every 50 rounds
    when the aggregated direction drifts away from the fixed sign pattern.
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        # Track local cosine similarity between successive updates.
        if self.category == "attacker":
            self.need_local_cos_history = True

        # Reuse PoisonedFL defaults and extend with new hyper-params.
        self.default_attack_params = {
            "scaling_factor": 10.0,
            "early_round": 10,
            "top_k_ratio": 0.05,
            "important_magnitude": 10.0,
            "unimportant_magnitude": 0.1,
            "cos_monitor_start_round": 50,
            "recovery_window": 10,
        }  # scaling_factor follows original default
        self.update_and_set_attr()

        self.current_scaling_factor = float(self.scaling_factor)
        self.cos_monitor_start_round = int(self.cos_monitor_start_round)
        self.force_warmup_rounds_init = int(self.early_round)
        # Only enable forced warm-up after recovery is triggered.
        self.force_warmup_rounds = 0

        # Fixed attack direction; set once after warm-up.
        self.fixed_rand = None
        self.prev_global_vec = None
        self.last_50_global_vec = None
        # 缓存上一轮更新向量，缺失时退回良性更新以保证数值稳定
        self.last_grad = None
        # 历史梯度缓存，仅用于 warm-up 与方向锁定
        self.grad_history = []
        # attack_state 取值：stable -> crashing -> recovering
        self.attack_state = "stable"
        # 保存topk，用于添加水印
        self.topk_mask = None
        self.topk_watermark = None
        self.topk_watermark_socre = 0.0

    def omniscient(self, clients):
        attackers = [
            client for client in clients
            if client.category == "attacker"
        ]
        if not attackers:
            return None

        # 当前全局模型向量，用作残差/漂移的基准
        current_global_vec = torch.as_tensor(
            self.global_weights_vec, device=self.args.device
        ).float().flatten()

        current_epoch = int(self.global_epoch)

        # 第一轮
        if current_epoch == 1:
            self.prev_global_vec = current_global_vec.clone()
            # 返回当前轮良性平均更新
            benign_updates = self._collect_benign_updates(attackers)
            return benign_updates
        
        # 比较水印
        topk_watermark_socre = 0.0
        if current_epoch > 20:
            topk_watermark_socre = self.matched_filter_score(current_global_vec)

        if current_epoch % 20 == 0:
            self._log_message(f"matched_filter_score={topk_watermark_socre}")

        # 计算梯度
        current_grad = None
        prev_global_vec = None
        prev_global_vec = self.prev_global_vec.clone()
        current_grad = current_global_vec - prev_global_vec   #  W_t - W_{t-1} 
        self.prev_global_vec = current_global_vec.clone()

        # Warm-up阶段：前 early_round 轮仅收集历史梯度，返回良性更新
        if current_epoch < int(self.early_round):
            # 只存储梯度，避免把权重当作梯度导致 PCA 方向偏移
            self.grad_history.append(current_grad.detach().cpu())
            self.last_grad = current_grad.clone()
            # 返回当前轮良性平均更新
            benign_updates = self._collect_benign_updates(attackers)
            return benign_updates
         
        # cos监视并改变攻击状态，从50轮开始
        self._update_cos_state_and_maybe_reset(50)
        
        # 再次 warm-up 阶段：这个阶段攻击方向为历史梯度方向的正交方向
        if self.force_warmup_rounds > 0 and current_epoch > 50:
            self.force_warmup_rounds -= 1
            self.grad_history.append(current_grad.detach().cpu())
            # 强制 warm-up 结束后，重置攻击方向
            if self.force_warmup_rounds == 0 and self.attack_state == "recovering":
                self.attack_state = "stable"
                self.current_scaling_factor = float(self.scaling_factor)
                self.fixed_rand = None

         # warm-up 阶段锁定方向：结束 warm-up 后，只要 fixed_rand 为空就重新锁定方向
        if self.fixed_rand is None:
            # 攻击方向：PCA主方向的正交方向并且符合topk权重的参数对比
            pca_direction = self.get_attack_direction_by_pca(current_global_vec)
            topk_vec = self.get_attack_direction_by_weight_importent(current_grad)
            self.fixed_rand = self.getOrthogonalVectorByRotary(pca_direction, topk_vec)
            self.fixed_rand = self.fixed_rand / (self.fixed_rand.norm() + 1e-12)  # 单位向量

            # # 输出结果（分行显示，保留 10 位小数）
            # def _format_vec(vec, width=8):
            #     values = [f"{v:.10f}" for v in vec.tolist()]
            #     lines = [
            #         " ".join(values[i:i + width])
            #         for i in range(0, len(values), width)
            #     ]
            #     return "[\n  " + "\n  ".join(lines) + "\n]"

            # pca_str = _format_vec(pca_direction[:100].detach().cpu())
            # topk_str = _format_vec(topk_vec[:100].detach().cpu())
            # fixed_str = _format_vec(self.fixed_rand[:100].detach().cpu())
            # self._log_message(f"[ORTHO] pca_direction[:100]={pca_str}")
            # self._log_message(f"[ORTHO] topk[:100]={topk_str}")
            # self._log_message(f"[ORTHO] 攻击方向[:100]={fixed_str}")
            # ortho_dot = torch.dot(self.fixed_rand, pca_direction)
            # self._log_message(f"[ORTHO] 攻击方向和PCA方向是否正交(接近0)={float(ortho_dot.item()):.8e}")
    
        k_95, k_99 = self._get_thresholds(self.fixed_rand.numel())
        sf = float(self.current_scaling_factor)
        eps = 1e-9

        # 通过本地客户端更新的幅度设置攻击幅度，随便选取一个攻击者
        update = attackers[0].update
        if torch.is_tensor(update):
            attacker_update = update.detach().cpu().numpy()
        else:
            attacker_update = np.array(update, copy=True)
        attacker_update_norm = float(np.linalg.norm(attacker_update))
        # self._log_message(f"[attacker_update_norm]{attacker_update_norm}")

        # 确保 fixed_rand 是单位向量，然后赋予它梯度的模长
        # deviation = self.fixed_rand * attacker_update_norm
        deviation = self.fixed_rand  # 后面会乘上attacker_update_norm

        if current_epoch % 50 == 0:
            if self.last_50_global_vec is None:
                self.last_50_global_vec = current_global_vec.clone()
            # 每 50 轮检查一次固定方向对齐度，偏离过大则缩放攻击幅度
            total_update = current_global_vec - self.last_50_global_vec
            replaced = torch.where(total_update == 0, current_global_vec, total_update)
            current_sign = torch.sign(replaced)
            aligned_dim_cnt = int((current_sign == self.fixed_rand).sum().item())
            if aligned_dim_cnt < k_99 and sf * 0.8 >= 0.5:
                sf = sf * 0.8
            lamda_succ = sf * attacker_update_norm
            self.last_50_global_vec = current_global_vec.clone()
            # self._log_message(f"[sf]{sf}")
        else:
            lamda_succ = sf * attacker_update_norm

        # 固定方向生成恶意更新，并复制到所有攻击者
        mal_update = lamda_succ * deviation
        # mal_update = sf * deviation

        # XXX:测试
        if current_epoch % 20 == 0:
            mal_str = np.array2string(
                mal_update[:50].detach().cpu().numpy(),
                separator=" ",
                max_line_width=1000,
                formatter={"float_kind": lambda x: f"{x:.2e}"},
            )
            self._log_message(f"mal_update={mal_str}")
            update = attackers[0].update
            if torch.is_tensor(update):
                attacker_update = update.detach().cpu().numpy()
            else:
                attacker_update = np.array(update, copy=True)
            benign_update_norm = float(np.linalg.norm(attacker_update))
            benign_str = np.array2string(
                attacker_update[:50],
                separator=" ",
                max_line_width=1000,
                formatter={"float_kind": lambda x: f"{x:.2e}"},
            )
            self._log_message(f"benign_update={benign_str}")
            mal_update_norm = float(mal_update.norm().item())
            ratio = mal_update_norm / (benign_update_norm + 1e-12)
            self._log_message(f"mal_benign_l2_ratio={ratio}")
            self._log_message(f"sf={sf}")

        # 恶意更新上基于topk添加水印
        eta = 0.01 * mal_update.norm() / math.sqrt(float(self.top_k_ratio))
        mal_update = self.build_topk_watermark(44, mal_update, eta)

        # 持久化状态，下一轮继续基于同一方向与缩放因子
        self.current_scaling_factor = sf
        self.last_grad = current_grad.clone()

        mal_update_np = mal_update.detach().cpu().numpy().astype(np.float32)
        malicious_updates = np.tile(mal_update_np, (len(attackers), 1))

        return malicious_updates
    
    # 判断攻击状态
    def _update_cos_state_and_maybe_reset(self, start_round):
        if self.global_epoch < start_round:
            return    
        if getattr(self, "local_cos", None) is None:
            return

        current_cos = float(self.local_cos)
        prev_state = self.attack_state
        recovery_window = int(getattr(self, "recovery_window", 8))

        # 状态机入口：先处理“触发崩盘”的跃迁，再按状态分发处理逻辑
        if prev_state not in ("crashing", "recovering") and current_cos < 0.4:
            self._enter_crashing(current_cos, recovery_window)
            return

        state_handlers = {
            "crashing": self._handle_crashing_state,
            "recovering": self._handle_recovering_state,
        }
        handler = state_handlers.get(prev_state)
        if handler is not None:
            handler(current_cos, recovery_window)
            return

        # 默认状态：稳定期 (Stable)
        self.attack_state = "stable"

    def _enter_crashing(self, current_cos, recovery_window):
        # 触发崩盘：之前没崩，现在突然 Cos 掉下来了
        self.local_accuracy_history = []
        self.attack_state = "crashing"

        # 记录test accuracy
        current_test_acc = self._eval_global_test_acc()
        if current_test_acc is not None:
            self._push_accuracy(current_test_acc, recovery_window)

        msg = f"[Adaptive] System Crashing Triggered (Cos: {current_cos:.3f}, Acc: {current_test_acc})"
        self._log_message(msg)

    def _handle_crashing_state(self, current_cos, recovery_window):
        # 崩盘监控期：只要之前是 crashing，就必须锁死在这个状态，直到窗口跑完
        current_test_acc = self._eval_global_test_acc()
        if current_test_acc is not None:
            self._push_accuracy(current_test_acc, recovery_window)

        # 只有当数据填满窗口时，才开始判断
        if len(self.local_accuracy_history) >= recovery_window:
            # 先判断是否为攻击生效，若窗口内平均精度极低，认为攻击已完全生效
            if self.local_accuracy_history:
                avg_acc = sum(self.local_accuracy_history) / len(self.local_accuracy_history)
                if avg_acc < 0.15:
                    self.attack_state = "stable"
                    msg = f"[Adaptive] Attack Sucess (Cos: {current_cos:.3f}, Acc: {current_test_acc})"
                    self._log_message(msg)
                    return

            # 判断是否恢复：精度显著上升 OR 精度已经很高
            if self._is_recovering_by_window(recovery_window):
                msg = f"[Adaptive] Recovery Confirmed (Window={recovery_window}). Resetting Attack."
                self._log_message(msg)

                # 切换到恢复模式，force_warmup_rounds不为0重置攻击方向
                self.force_warmup_rounds = self.force_warmup_rounds_init
                self.grad_history = []
                self.attack_state = "recovering"
                return

            # 没恢复（精度低且平，或者还在降），继续保持 Crashing
            self.attack_state = "crashing"
        else:
            # 窗口没满，强制保持 crashing，等待数据积累
            self.attack_state = "crashing"

    def _handle_recovering_state(self, current_cos, recovery_window):
        # 恢复过渡期：状态维持，等待 warm-up 倒计时结束（由外部逻辑控制转为 stable）
        self.attack_state = "recovering"

    def _compute_pca_direction(self, current_global_vec):
        # PCA main direction from historical pseudo-gradients.
        if len(self.grad_history) > 0:
            history_mat = torch.stack(self.grad_history, dim=0).to(
                current_global_vec.device
            ).float()
            history_centered = history_mat - history_mat.mean(dim=0, keepdim=True)
            if history_centered.shape[0] == 1:
                v_pca = history_centered[0]
            else:
                _, _, v_t = torch.linalg.svd(history_centered, full_matrices=False)
                v_pca = v_t[0]
        else:
            v_pca = torch.randn_like(current_global_vec)
        if torch.norm(v_pca) == 0:
            v_pca = torch.randn_like(current_global_vec)
        return v_pca
    
    # 以第一个权重为旋钮进行旋转，缺点：会导致旋钮权重远大于其他权重，最终L2接近旋钮权重的值
    # def getOrthogonalVectorByRotary(self,a, b):
    #     c = b.clone()
    #     for i in range(a.numel()):
    #         if a[i] != 0:
    #             sum_excluding_i = torch.sum(a * c) - a[i] * c[i]
    #             c[i] = -sum_excluding_i / a[i]
    #             break
    #     return c

    # Gram-Schmidt 正交化
    def getOrthogonalVectorByRotary(self, a, b):
        # Project b onto the orthogonal complement of a to avoid a single large coordinate.
        denom = torch.dot(a, a)
        if denom.abs() < 1e-12:
            return b.clone()
        proj = torch.dot(a, b) / denom
        c = b - proj * a
        # If projection collapses (b almost parallel to a), fallback to a random orthogonal.
        if c.norm() < 1e-12:
            rand_vec = torch.randn_like(a)
            proj = torch.dot(a, rand_vec) / denom
            c = rand_vec - proj * a
        return c

    # 构造水印
    def build_topk_watermark(self, seed, mal_update=None, eta=None):
        # Build a reproducible +/-1 watermark on top-k positions; optionally add to mal_update.

        device = self.prev_global_vec.device
        dtype = self.prev_global_vec.dtype
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

        mask = self.topk_mask.to(device=device)
        watermark = torch.zeros_like(self.prev_global_vec, device=device, dtype=dtype)
        num_topk = int(mask.sum().item())
        if num_topk > 0:
            rand = torch.randint(0, 2, (num_topk,), generator=gen, device=device)
            rand = rand.to(dtype=dtype) * 2 - 1
            watermark[mask] = rand
        self.topk_watermark = watermark
        update = mal_update + eta * self.topk_watermark
        # self._log_message(f"mal_update with topk_watermark={update}")

        return update

    def matched_filter_score(self, current_global_vec):
        # Matched filter score for current global update.
        g = current_global_vec - self.prev_global_vec
        # self._log_message(f"g={g}")
        score = torch.dot(g, self.topk_watermark).item()
        # self._log_message(f"matched_filter_score={score}")
        return score

    def get_attack_direction_by_pca(self, current_global_vec):
        # 1) 获得PCA的主方向
        v_pca = self._compute_pca_direction(current_global_vec)
        # 2) 获得PCA的主方向的正交方向作为攻击方向
        ortho_vec = self._compute_orthogonal_sign(v_pca)
        # 3）归一化得到单位向量
        # ortho_unit = ortho_sign / (ortho_sign.norm() + 1e-12)
        # self._log_message(f"[PCA]{ortho_unit}")
        return ortho_vec.detach().clone()

    def get_attack_direction_by_weight_importent(self, current_grad):
        # 1) 求重要权重
        importance_full = self._compute_param_importance()
        # 2) 权重的Top-K mask
        topk_mask = self._compute_topk_mask(importance_full)
        self.topk_mask = topk_mask
        # 3) 求梯度的反方向
        sign_vec = -torch.sign(current_grad)
        weighted_vec = self._apply_importance_weight(sign_vec, topk_mask)
        # # 4）归一化
        # weighted_vec_unit = weighted_vec / (weighted_vec.norm() + 1e-12)
        # self._log_message(f"[WEIGHT]{weighted_vec_unit}")
        return weighted_vec.detach().clone()

    # def get_attack_direction_by_pca(self, current_global_vec):
    #     # 1) PCA main direction from history.
    #     v_pca = self._compute_pca_direction(current_global_vec)
    #     # 2) Captum importance for params.
    #     importance_full = self._compute_param_importance()
    #     # 3) Top-K mask on important params.
    #     topk_mask = self._compute_topk_mask(importance_full)
    #     # 4) Orthogonal attack direction to PCA.
    #     ortho_sign = self._compute_orthogonal_sign(v_pca)
    #     # 5) Weight by importance.
    #     weighted_vec = self._apply_importance_weight(ortho_sign, topk_mask)
    #     return weighted_vec.detach().clone()
    
    def _compute_param_importance(self):
        # Captum saliency for params; align to state_dict vector order.
        prev_training = self.model.training
        self.model.eval()
        images, targets = next(iter(self.train_loader))
        images = images.to(self.args.device)
        targets = targets.to(self.args.device)

        param_meta = []
        for name, param in self.model.named_parameters():
            param_meta.append((name, param.numel(), param.shape))
        param_vec = torch.nn.utils.parameters_to_vector(
            self.model.parameters()
        ).detach().to(self.args.device)
        param_vec = param_vec.requires_grad_(True)
        param_slices = {}
        cursor = 0
        for name, numel, shape in param_meta:
            param_slices[name] = (cursor, cursor + numel, shape)
            cursor += numel

        buffers = dict(self.model.named_buffers())
        forward_fn = functools.partial(
            self._forward_with_params,
            param_slices=param_slices,
            buffers=buffers,
            images=images,
            targets=targets,
        )
        saliency = Saliency(forward_fn)
        param_importance = saliency.attribute(param_vec, abs=True).detach().flatten()
        param_importance = param_importance.cpu()
        if prev_training:
            self.model.train()

        importance_full = []
        for key, value in self.model.state_dict().items():
            if key in param_slices:
                start, end, _ = param_slices[key]
                importance_full.append(param_importance[start:end])
            else:
                importance_full.append(torch.zeros(value.numel()))
        importance_full = torch.cat(importance_full, dim=0)
        return importance_full

    def _compute_topk_mask(self, importance_full):
        total_dim = importance_full.numel()
        k = max(1, int(total_dim * float(self.top_k_ratio)))
        _, topk_idx = torch.topk(importance_full, k)
        topk_mask = torch.zeros(
            total_dim, dtype=torch.bool, device=importance_full.device
        )
        topk_mask[topk_idx] = True
        return topk_mask

    def _compute_orthogonal_sign(self, base_vec):
        # Return a vector orthogonal to base_vec; no sign discretization.
        rand_vec = torch.randn_like(base_vec)
        denom = torch.dot(base_vec, base_vec) + 1e-12
        proj = torch.dot(rand_vec, base_vec) / denom
        ortho_vec = rand_vec - proj * base_vec
        if ortho_vec.norm() < 1e-12:
            rand_vec = torch.randn_like(base_vec)
            proj = torch.dot(rand_vec, base_vec) / denom
            ortho_vec = rand_vec - proj * base_vec
        return ortho_vec

    def _apply_importance_weight(self, ortho_sign, topk_mask):
        weighted_vec = ortho_sign.clone()
        weighted_vec[topk_mask] = (
            weighted_vec[topk_mask] * float(self.important_magnitude)
        )
        weighted_vec[~topk_mask] = (
            weighted_vec[~topk_mask] * float(self.unimportant_magnitude)
        )
        return weighted_vec

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

    def _eval_global_test_acc(self):
        # 使用全局模型在 test_dataset 上评估准确率，但不永久覆盖本地模型参数。
        if self.test_dataset is None or getattr(self, "global_weights_vec", None) is None:
            return None
        local_param_vec = torch.nn.utils.parameters_to_vector(
            self.model.parameters()
        ).detach().clone()
        try:
            # 临时切到全局参数评估收敛趋势。
            self.load_global_model(self.global_weights_vec)
            test_acc, _ = self.client_test(model=self.model, test_dataset=self.test_dataset)
        except Exception:
            return None
        finally:
            # 恢复本地参数，避免影响后续攻击逻辑与更新计算。
            torch.nn.utils.vector_to_parameters(local_param_vec, self.model.parameters())
        return float(test_acc)

    def _collect_benign_updates(self, attackers):
        benign_updates = []
        for client in attackers:
            if torch.is_tensor(client.update):
                update = client.update.detach().cpu().numpy()
            else:
                update = np.array(client.update, copy=True)
            benign_updates.append(update)
        benign_updates = np.stack(benign_updates, axis=0).astype(np.float32)
        benign_mean_vec = torch.from_numpy(
            np.mean(benign_updates, axis=0)
        ).float()
        return benign_updates

    def _is_recovering_by_window(self, window_size):
        # 滑动窗口法判断 recovering：
        # 1) 把窗口拆成前半/后半，比较均值是否“持续抬升”；
        # 2) 要求最后一个点不低于后半段均值，避免短时尖峰误判。
        if window_size < 2:
            return False
        if len(self.local_accuracy_history) < window_size:
            return False

        window = self.local_accuracy_history[-window_size:]
        mid = window_size // 2
        first_half = window[:mid]
        second_half = window[mid:]
        if not first_half or not second_half:
            return False

        first_mean = sum(first_half) / len(first_half)
        second_mean = sum(second_half) / len(second_half)
        return second_mean > first_mean and window[-1] >= second_mean

    def _push_accuracy(self, value, window_size):
        # 只在 crashing 状态记录测试精度，并保持窗口长度恒定。
        self.local_accuracy = float(value)
        self.local_accuracy_history.append(self.local_accuracy)
        if window_size > 0 and len(self.local_accuracy_history) > window_size:
            # 新数据进入时丢弃最早的数据。
            self.local_accuracy_history = self.local_accuracy_history[-window_size:]

    def _log_message(self, msg):
        logger = getattr(self.args, "logger", None)
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)            

    def _forward_with_params(self, flat_params, param_slices, buffers, images, targets):
        # 用可微参数向量构造临时参数字典，保持计算图连通
        params_dict = {}
        for pname, (start, end, shape) in param_slices.items():
            params_dict[pname] = flat_params[start:end].view(shape)
        # 兼容不同 PyTorch 版本的 functional_call 签名
        try:
            logits = functional_call(
                self.model, params_dict, (images,), buffers=buffers
            )
        except TypeError:
            try:
                logits = functional_call(
                    self.model, (params_dict, buffers), (images,)
                )
            except Exception:
                merged = dict(params_dict)
                merged.update(buffers)
                logits = functional_call(self.model, merged, (images,))
        loss = self.criterion_fn(logits, targets)
        return loss.view(1)
