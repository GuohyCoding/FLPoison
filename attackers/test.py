# -*- coding: utf-8 -*-

import os

import numpy as np
import torch
import functools

from copy import deepcopy
from captum.attr import LayerConductance

from attackers import attacker_registry
from attackers.pbases.mpbase import MPBase
from fl.client import Client
from fl.models.model_utils import vec2model
from global_utils import actor

from captum.attr import Saliency
try:
    from torch.func import functional_call
except Exception:  # 兼容较旧 PyTorch 版本
    from torch.nn.utils.stateless import functional_call

@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class Test(MPBase, Client):

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        # 攻击（实验）相关默认参数：可在 config/args 中通过 attack_params 覆盖
        self.default_attack_params = {
            "early_rounds": [5, 10, 15, 20],  # 分别比较前几轮
            "compare_mode": "early_vs_all",   # early_vs_late 或 early_vs_all，和最后几轮比较还是和全局比较
            "late_rounds": 10,
            "enable_sim_curve": True,
            "sim_curve_every": 5,
            "sim_curve_path": "logs/pca/pca_sim_curve_{epochs}.png",
            "enable_weight_curve": True,
            "weight_curve_every": 5,
            "weight_curve_path": "logs/weight/weight_topk_overlap_{epochs}.png",
            "top_k_ratio": 0.05,
            "captum_batch_size": 32,
            "important_magnitude": 100.0,
            "unimportant_magnitude": 0.1,
        }
        
        self.update_and_set_attr()

        # 仅在 CPU 上做 PCA，避免显存占用
        self.prev_global_vec = None
        # 缓存每一轮的全局更新量 Δw_t
        self.history_deltas = []
        # 避免重复计算 PCA
        self.pca_done = False
        # 避免重复计算 Captum 归因重合度
        self.captum_done = False
        # 缓存指定轮次的全局模型向量，用于 Captum 对比（避免保存全部轮次）
        self.captum_weight_cache = {}
        # 复用全局 logger（若存在）
        self.logger = getattr(args, "logger", None)
        # 测试方向是否正交
        self.fixed_rand = None
        self.weight_history = []

        # 比较轮数严格跟随实际训练轮数 epochs
        if not hasattr(args, "epochs"):
            raise ValueError("[PCA-Direction-Test] args.epochs is required to set compare rounds")
        try:
            epochs = int(args.epochs)
        except Exception as exc:
            raise ValueError("[PCA-Direction-Test] args.epochs must be an integer") from exc
        if epochs <= 0:
            raise ValueError("[PCA-Direction-Test] args.epochs must be positive")
        # 注意：Δw 从第 2 轮才有，因此可用的 Δw 数量是 epochs-1
        self.total_epochs = epochs
        self.total_rounds_for_compare = max(1, epochs - 1)
   
        self.early_rounds_list = self._normalize_early_rounds(self.early_rounds)
        if not self.early_rounds_list:
            raise ValueError("[PCA-Direction-Test] early_rounds must contain at least one positive int")
        # self._log(
        #     f"[PCA-Direction-Test] total_rounds_for_compare set to epochs-1={self.total_rounds_for_compare}"
        # )

    def omniscient(self, clients):
        # omniscient 攻击入口：框架会在每轮聚合前调用此方法
        attackers = [
            client for client in clients
            if client.category == "attacker"
        ]
        if not attackers:
            return None
        
        # # -------------正交-------------
        current_global_vec = torch.as_tensor(
            self.global_weights_vec, dtype=torch.float32
        ).detach().to(self.args.device).flatten()

        current_epoch = int(self.global_epoch)

        # # 得到随机方向
        # if self.fixed_rand is None and current_epoch == 1:
        #     self.fixed_rand = torch.sign(torch.randn_like(current_global_vec))
        #     zero_mask = self.fixed_rand == 0
        #     if torch.any(zero_mask):
        #         self.fixed_rand = torch.where(
        #             zero_mask, torch.ones_like(self.fixed_rand), self.fixed_rand
        #         )

        #     b = self.get_attack_direction_by_weight_importent(current_global_vec)
        #     c = self.getOrthogonalVectorByRotary(self.fixed_rand, b)
        #     self._log(f"[ORTHO] b[:200]={b[:200].detach().cpu()}")
        #     self._log(f"[ORTHO] c[:200]={c[:200].detach().cpu()}")
        #     ortho_dot = torch.dot(self.fixed_rand, c)
        #     self._log(f"[ORTHO] bTc={float(ortho_dot.item()):.6e}")

        # # 返回当前轮良性平均更新
        # benign_updates = self._collect_benign_updates(attackers)
        # return benign_updates

        # -------------测试PCA和权重-------------
        # 获取本轮全局模型向量 w_t（来自 server broadcast）
        current_vec = self._to_flat_vec(self.global_weights_vec)
        # 记录当前轮次编号（client.fetch_updates 会自增 global_epoch）
        current_round = int(self.global_epoch)
        # 仅在需要的轮次缓存全局模型向量，供 Captum 对比使用
        self._cache_round_weights(current_round)

        # 首轮没有上一轮参数，无法形成 Δw，直接返回良性更新
        if self.prev_global_vec is None:
            self.prev_global_vec = current_vec.clone()
            return self._collect_benign_updates(attackers)

        # 计算本轮全局更新量 Δw_t = w_t - w_{t-1}
        delta = current_vec - self.prev_global_vec
        self.prev_global_vec = current_vec.clone()

        # 只收集前 total_rounds_for_compare 轮的 Δw
        if len(self.history_deltas) < int(self.total_rounds_for_compare):
            self.history_deltas.append(delta.detach().cpu())

        # 达到指定轮数后只计算一次 PCA 并输出结果
        if (not self.pca_done) and len(self.history_deltas) >= int(self.total_rounds_for_compare):
            self._compare_pca_directions()
            self.pca_done = True
        # 在收集到足够轮次后，计算 Captum Top-K 归因重合度
        if (not self.captum_done) and len(self.history_deltas) >= int(self.total_rounds_for_compare):
            self._compare_captum_overlap()
            self.captum_done = True

        # 本实验不进行投毒：直接返回原样更新
        return self._collect_benign_updates(attackers)

    def _log_message(self, msg):
        logger = getattr(self.args, "logger", None)
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)  
    
    def getOrthogonalVectorByRotary(self,a, b):
        c = b.clone()
        for i in range(a.numel()):
            if a[i] != 0:
                sum_excluding_i = torch.sum(a * c) - a[i] * c[i]
                c[i] = -sum_excluding_i / a[i]
                break
        return c
    
    def get_attack_direction_by_weight_importent(self, current_global_vec):
        # 1) 求重要权重
        importance_full = self._compute_param_importance()
        # 2) 权重的Top-K mask
        topk_mask = self._compute_topk_mask(importance_full)
        # 3) 求梯度的反方向
        sign_vec = -torch.sign(current_global_vec)
        weighted_vec = self._apply_importance_weight(sign_vec, topk_mask)
        # 4）归一化
        weighted_vec_unit = weighted_vec / (weighted_vec.norm() + 1e-12)
        self._log_message(f"[WEIGHT]{weighted_vec_unit}")
        return weighted_vec_unit.detach().clone()
    
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
        if prev_training:
            self.model.train()

        importance_full = []
        for key, value in self.model.state_dict().items():
            if key in param_slices:
                start, end, _ = param_slices[key]
                importance_full.append(param_importance[start:end])
            else:
                importance_full.append(torch.zeros(value.numel(), device=self.args.device))
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
        rand_vec = torch.randn_like(base_vec)
        denom = torch.dot(base_vec, base_vec) + 1e-12
        proj = torch.dot(rand_vec, base_vec) / denom
        ortho_vec = rand_vec - proj * base_vec
        ortho_sign = torch.sign(ortho_vec)
        zero_mask = ortho_sign == 0
        if torch.any(zero_mask):
            ortho_sign = torch.where(
                zero_mask, torch.ones_like(ortho_sign), ortho_sign
            )
        return ortho_sign

    def _apply_importance_weight(self, ortho_sign, topk_mask):
        weighted_vec = ortho_sign.clone()
        weighted_vec[topk_mask] = (
            weighted_vec[topk_mask] * float(self.important_magnitude)
        )
        weighted_vec[~topk_mask] = (
            weighted_vec[~topk_mask] * float(self.unimportant_magnitude)
        )
        return weighted_vec

    def _to_flat_vec(self, obj):
        # 将输入对象统一转成 1D float Tensor，保持在 args.device 上
        if torch.is_tensor(obj):
            vec = obj.detach()
        elif isinstance(obj, np.ndarray):
            vec = torch.from_numpy(obj)
        elif isinstance(obj, dict):
            vec = flatten_state_dict(obj)
        elif hasattr(obj, "state_dict"):
            vec = flatten_state_dict(obj.state_dict())
        else:
            vec = torch.as_tensor(obj)
        return vec.flatten().float().to(self.args.device)

    def _collect_benign_updates(self, attackers):
        # 汇总攻击者客户端当前更新（保持原样），满足框架接口
        def _to_numpy(u):
            if torch.is_tensor(u):
                return u.detach().cpu().numpy()
            return np.array(u, copy=True)

        benign_updates = np.stack(
            [_to_numpy(client.update) for client in attackers], axis=0
        ).astype(np.float32)
        return benign_updates

    def _compare_pca_directions(self):
        total_rounds = int(self.total_rounds_for_compare)
        compare_mode = str(self.compare_mode).lower()

        if total_rounds < 2:
            self._log("[PCA-Direction-Test] Not enough rounds for PCA")
            return

        deltas = self.history_deltas[:total_rounds]
        if len(deltas) < total_rounds:
            self._log("[PCA-Direction-Test] Delta history shorter than total_rounds_for_compare")
            return

        if compare_mode == "early_vs_late":
            late_rounds = min(int(self.late_rounds), total_rounds)
            A_ref = torch.stack(deltas[-late_rounds:], dim=0)
            ref_tag = f"late_{late_rounds}"
        else:
            A_ref = torch.stack(deltas, dim=0)
            ref_tag = f"all_{total_rounds}"

        v_ref, s_ref = self._pca_first_component(A_ref)
        if v_ref is None:
            self._log("[PCA-Direction-Test] PCA failed to produce a reference component")
            return

        for early_rounds in self.early_rounds_list:
            early_rounds = min(int(early_rounds), total_rounds)
            if early_rounds < 1:
                continue

            A_early = torch.stack(deltas[:early_rounds], dim=0)

            v_early, s_early = self._pca_first_component(A_early)
            if v_early is None:
                self._log("[PCA-Direction-Test] PCA failed to produce an early component")
                continue

            sim = float(torch.abs(torch.dot(v_early, v_ref)).item())
            s1_early = float(s_early[0].item()) if s_early is not None else float("nan")
            s1_ref = float(s_ref[0].item()) if s_ref is not None else float("nan")

            msg = (
                f"[PCA] round: {total_rounds} "
                f"early: {early_rounds}"
                f"|cos| of PC1 directions: {sim:.2f} "
                f"early PC1 strength: {s1_early:.2f} "
                f"ref PC1 strength: {s1_ref:.2f}"
            )
            self._log(msg)

        if bool(getattr(self, "enable_sim_curve", False)):
            self._plot_sim_curve(deltas, total_rounds)

    def _plot_sim_curve(self, deltas, total_rounds):
        step = int(getattr(self, "sim_curve_every", 5))
        if step <= 0:
            self._log("[PCA-Direction-Test] sim_curve_every must be positive")
            return

        A_all = torch.stack(deltas, dim=0)
        v_global, _ = self._pca_first_component(A_all)
        if v_global is None:
            self._log("[PCA-Direction-Test] PCA failed to produce a global component")
            return

        rounds = []
        sims = []
        for r in range(step, total_rounds + 1, step):
            A_r = torch.stack(deltas[:r], dim=0)
            v_r, _ = self._pca_first_component(A_r)
            if v_r is None:
                continue
            sim = float(torch.abs(torch.dot(v_r, v_global)).item())
            rounds.append(r)
            sims.append(sim)

        if not rounds:
            self._log("[PCA-Direction-Test] No points to plot for sim curve")
            return

        try:
            import matplotlib.pyplot as plt
        except Exception:
            self._log("[PCA-Direction-Test] matplotlib is not available for plotting")
            return

        plt.figure()
        plt.plot(rounds, sims, marker="o")
        plt.ylim(0.0, 1.0)
        plt.xlabel("Round")
        plt.ylabel("|cos|")
        plt.title(f"PCA Direction Similarity (every {step} rounds)")

        path_template = str(self.sim_curve_path)
        if "{epochs}" in path_template:
            path = path_template.format(epochs=self.total_epochs)
        else:
            path = path_template
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        self._log(f"[PCA-Direction-Test] sim curve saved to {path}")

    def _pca_first_component(self, matrix):
        # 先对数据做中心化：A <- A - mean(A, dim=0)
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
        if matrix.shape[0] < 2 or matrix.shape[1] < 1:
            return None, None

        try:
            # 优先使用 torch.pca_lowrank（速度快、稳定）
            _, svals, vecs = torch.pca_lowrank(matrix, q=1, center=False)
            v1 = vecs[:, 0]
            s_out = svals
        except Exception:
            # 兼容旧版 PyTorch 或 pca_lowrank 不可用时的后备方案
            _, svals, v_h = torch.linalg.svd(matrix, full_matrices=False)
            v1 = v_h[0]
            s_out = svals

        # 归一化 PC1 向量
        norm = torch.norm(v1)
        if norm <= 0:
            return None, None
        v1 = v1 / norm
        return v1, s_out

    def _cache_round_weights(self, round_id):
        # 仅缓存早期轮次与最终轮次，避免无意义的内存占用
        target_final_round = int(self.total_rounds_for_compare)
        keep = (round_id in self.early_rounds_list) or (round_id == target_final_round)
        if bool(getattr(self, "enable_weight_curve", False)):
            step = int(getattr(self, "weight_curve_every", 5))
            if step > 0 and (round_id % step == 0):
                keep = True
        if keep:
            raw = self.global_weights_vec
            if torch.is_tensor(raw):
                raw = raw.detach().cpu()
            vec = np.asarray(raw, dtype=np.float32).copy()
            self.captum_weight_cache[int(round_id)] = vec

    def _build_model_from_vec(self, weight_vec):
        # 基于当前模型结构克隆一个新模型，并写入指定权重向量
        tmp_model = deepcopy(self.model)
        vec2model(weight_vec, tmp_model)
        return tmp_model

    def _compare_captum_overlap(self):
        # 参考 PCA 的逻辑：对早期轮次与“最终可用轮次”进行 Captum 重要性重合度比较
        final_round = int(self.total_rounds_for_compare)
        if final_round not in self.captum_weight_cache:
            self._log("[Captum] Final round weights are missing; skip overlap.")
            return

        # 构造最终模型与早期模型列表
        final_model = self._build_model_from_vec(self.captum_weight_cache[final_round])
        curve_enabled = bool(getattr(self, "enable_weight_curve", False))
        curve_step = int(getattr(self, "weight_curve_every", 5)) if curve_enabled else None
        compare_rounds = list(self.early_rounds_list)
        if curve_enabled and curve_step is not None and curve_step > 0:
            compare_rounds.extend(range(curve_step, final_round + 1, curve_step))
        compare_rounds = sorted(set(compare_rounds))

        early_models = []
        early_rounds = []
        for r in compare_rounds:
            if r in self.captum_weight_cache:
                early_rounds.append(r)
                early_models.append(self._build_model_from_vec(self.captum_weight_cache[r]))

        if not early_models:
            self._log("[Captum] No early round weights are cached; skip overlap.")
            return

        # 取一批测试数据用于归因，确保所有对比使用同一批样本
        test_loader = self.get_dataloader(self.test_dataset, train_flag=False)
        model_pack = {
            "final_model": final_model,
            "early_models": early_models,
            "early_rounds": early_rounds,
            "captum_batch_size": getattr(self, "captum_batch_size", None),
        }
        results = calculate_captum_overlap(
            model_pack,
            test_loader,
            self.args.device,
            float(self.top_k_ratio),
            log_fn=self._log,
            log_rounds=self.early_rounds_list,
        )
        if curve_enabled:
            self._plot_captum_overlap(results, curve_step)

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

    def _plot_captum_overlap(self, results, step):
        if step is None or step <= 0:
            self._log("[Captum] weight_curve_every must be positive")
            return

        try:
            import matplotlib.pyplot as plt
        except Exception:
            self._log("[Captum] matplotlib is not available for plotting")
            return

        results = [(r, v) for r, v in results if int(r) % int(step) == 0]
        if not results:
            self._log("[Captum] no overlap results to plot")
            return

        rounds, overlaps = zip(*sorted(results, key=lambda x: x[0]))
        top_k_ratio = float(self.top_k_ratio)
        if top_k_ratio < 1:
            label = f"Top-K={top_k_ratio:.1%}"
        else:
            label = f"Top-K={int(top_k_ratio)}"

        plt.figure()
        plt.plot(rounds, overlaps, marker="o", label=label)
        plt.ylim(0.0, 1.0)
        plt.xlim(1, int(self.total_epochs))
        plt.xlabel("Round")
        plt.ylabel("Overlap Ratio")
        plt.title("Captum Top-K Overlap vs Final")
        plt.legend()

        path_template = str(self.weight_curve_path)
        if "{epochs}" in path_template:
            path = path_template.format(epochs=self.total_epochs)
        else:
            path = path_template
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        self._log(f"[Captum] overlap curve saved to {path}")

    def _normalize_early_rounds(self, early_rounds):
        if isinstance(early_rounds, (list, tuple)):
            values = list(early_rounds)
        elif isinstance(early_rounds, str):
            values = [v.strip() for v in early_rounds.split(',') if v.strip()]
        else:
            values = [early_rounds]

        normalized = []
        for v in values:
            try:
                iv = int(v)
            except Exception:
                continue
            if iv > 0:
                normalized.append(iv)

        if not normalized:
            return []
        return sorted(set(normalized))

    def _log(self, message):
        # 使用全局 logger 或回退到 print
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)


def flatten_state_dict(state_dict):
    """将 state_dict 中所有浮点张量按顺序拼成一维向量（CPU）。"""
    flat_tensors = []
    for _, value in state_dict.items():
        # 兼容非 Tensor 的权重（例如 list / numpy / 其他类型）
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        # 只保留浮点权重，跳过整数/布尔等 buffer
        if not torch.is_floating_point(value):
            continue
        # 统一拉平成 1D，并搬到 CPU，避免占用 GPU 显存
        flat_tensors.append(value.detach().flatten().cpu())
    if not flat_tensors:
        # 无可用权重时返回空向量
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(flat_tensors, dim=0)


def _get_captum_batch(test_loader, device, batch_size=None):
    # 从 test_loader 取一个批次做归因分析，保证后续所有轮次使用同一批样本
    # 这样能最大限度消除数据采样差异，专注比较“模型内部重要性结构”的相似度
    try:
        images, targets = next(iter(test_loader))
    except StopIteration as exc:
        raise ValueError("test_loader is empty") from exc

    # 允许用更小的 batch 做 Captum 归因，降低显存/时间开销
    if batch_size is not None and batch_size > 0:
        images = images[:batch_size]
        targets = targets[:batch_size]

    # 迁移到指定 device，保持与模型一致
    return images.to(device), targets.to(device)


def _get_target_layers(model):
    # 选择需要分析的层类型：线性层与卷积层
    # 如果模型结构自定义，且要扩展分析范围，可在这里添加更多层类型
    target_types = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    return [(name, layer) for name, layer in model.named_modules() if isinstance(layer, target_types)]


def _captum_topk_indices(model, inputs, targets, top_k_ratio):
    # 针对模型的目标层，用 Captum 计算归因分数并得到 Top-K 索引集合
    layers = _get_target_layers(model)
    if not layers:
        # 没有找到目标层，直接返回空集合
        return set()

    model.eval()
    model.zero_grad()

    attributions = []
    with torch.enable_grad():
        # 需要对输入开启梯度才能进行归因计算
        inputs = inputs.clone().detach().requires_grad_(True)
        for _, layer in layers:
            # 这里使用 LayerConductance：衡量层对最终输出的贡献
            # baselines 设为全零，与“输入从全零逐步变为真实输入”对应
            conductance = LayerConductance(model, layer)
            attr = conductance.attribute(inputs, target=targets, baselines=torch.zeros_like(inputs))
            # 归因取绝对值，避免正负抵消；按 batch 维度求均值形成“层级重要性”
            attr = attr.detach().abs()
            if attr.dim() > 1:
                attr = attr.mean(dim=0)
            # 展平成一维，便于跨层拼接与统一排序
            attributions.append(attr.flatten().cpu())

    if not attributions:
        return set()

    # 拼接所有层的归因向量，形成全局重要性分布
    all_attr = torch.cat(attributions, dim=0)
    total = int(all_attr.numel())
    if total == 0:
        return set()

    if top_k_ratio <= 0:
        raise ValueError("top_k_ratio must be positive")
    # top_k_ratio < 1 视为比例；否则视为绝对数量
    k = int(total * top_k_ratio) if top_k_ratio < 1 else int(top_k_ratio)
    # 保护性裁剪，避免 k 越界或为 0
    k = max(1, min(k, total))
    # 取全局 Top-K 归因位置索引
    _, top_idx = torch.topk(all_attr, k=k, largest=True)
    return set(top_idx.tolist())


def calculate_captum_overlap(model, test_loader, device, top_k, log_fn=print, log_rounds=None):
    # 该函数用于比较“早期轮次模型”与“最终模型”的 Top-K 归因重合度
    # 要求外部提供最终模型与一组早期模型（可以是 dict 或 list）
    final_model = None
    early_models = None
    early_rounds = None
    captum_batch_size = None

    # 兼容 dict 传参或对象属性传参两种方式
    if isinstance(model, dict):
        final_model = model.get("final_model") or model.get("final")
        early_models = model.get("early_models") or model.get("early")
        early_rounds = model.get("early_rounds")
        captum_batch_size = model.get("captum_batch_size")
    else:
        final_model = getattr(model, "final_model", None)
        early_models = getattr(model, "early_models", None)
        early_rounds = getattr(model, "early_rounds", None)
        captum_batch_size = getattr(model, "captum_batch_size", None)

    # 基本校验：必须同时具备最终模型与早期模型集合
    if final_model is None or early_models is None:
        raise ValueError("model must provide final_model and early_models for overlap comparison")

    # 如果 early_models 是 dict，但没显式给 early_rounds，就默认用其 key 顺序
    if isinstance(early_models, dict):
        if early_rounds is None:
            early_rounds = list(early_models.keys())
    else:
        # 如果 early_models 是 list，就用其索引作为 round id
        if early_rounds is None:
            early_rounds = list(range(len(early_models)))

    final_model = final_model.to(device)
    if isinstance(early_models, dict):
        early_items = [(r, early_models[r]) for r in early_rounds]
    else:
        early_items = list(zip(early_rounds, early_models))

    # 只取一个批次做归因，确保比较的一致性
    inputs, targets = _get_captum_batch(test_loader, device, captum_batch_size)
    # 先计算最终模型的 Top-K 归因索引，作为参考集合
    final_topk = _captum_topk_indices(final_model, inputs, targets, top_k)
    if not final_topk:
        raise ValueError("final model did not produce any Captum attributions")

    # 以最终模型 Top-K 的数量作为分母
    k = len(final_topk)
    results = []
    log_rounds_set = set(int(r) for r in log_rounds) if log_rounds is not None else None
    for round_id, early_model in early_items:
        early_model = early_model.to(device)
        # 计算早期模型的 Top-K 归因索引
        early_topk = _captum_topk_indices(early_model, inputs, targets, top_k)
        if not early_topk:
            overlap_ratio = 0.0
        else:
            # 交集 / K，衡量“重要性结构”在早期是否已形成
            overlap_ratio = len(final_topk.intersection(early_topk)) / float(k)
        # 直接打印输出，便于实验日志记录
        if log_rounds_set is None or int(round_id) in log_rounds_set:
            log_fn(f"[Captum] Round {round_id} vs Final: {overlap_ratio * 100:.1f}% Overlap")
        results.append((round_id, overlap_ratio))

    # 返回 (round_id, overlap_ratio) 列表，便于进一步绘图或统计
    return results


