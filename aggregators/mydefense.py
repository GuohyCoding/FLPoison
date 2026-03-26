"""
MyDefense 聚合器：将 DnC 的客户端级粗筛与 LASA 的逐层精筛结合起来。

设计思路：
1. 先在“客户端整体更新向量”层面使用 DnC，筛掉全局上明显离群的客户端；
2. 再仅对 DnC 保留下来的客户端，执行 LASA 风格的逐层异常检测与逐层平均；
3. 最终输出与当前联邦算法语义一致的聚合结果。

这个组合方案的出发点是：
- DnC 擅长识别“整体方向异常”的客户端；
- LASA 擅长识别“某一层局部统计特征异常”的客户端；
- 二者串联后，可以先做客户端级预筛，再做层级精筛。
"""

from aggregators.aggregatorbase import AggregatorBase
from aggregators.aggregator_utils import (
    prepare_grad_updates,
    wrapup_aggregated_grads,
)
import numpy as np
import torch
import warnings
from aggregators import aggregator_registry
from fl.models.model_utils import state2vec_torch, vec2state


@aggregator_registry
class MyDefense(AggregatorBase):
    """
    组合型鲁棒聚合器：
    - 第一阶段：DnC 风格客户端级预筛；
    - 第二阶段：LASA 风格逐层精筛与聚合。

    输入通常为各客户端上传的模型更新或梯度向量；
    输出为一个聚合后的全局更新向量。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 MyDefense，并设置两阶段防御所需超参数。

        参数说明：
        - dnc_subsample_frac: DnC 每轮随机抽取的参数维度比例。
        - dnc_num_iters: DnC 子采样 + 投影筛选的重复轮数。
        - dnc_filter_mode: DnC 客户端筛选模式。
          - "topk": 已知或估计 num_adv 时，直接过滤分数最高的一批客户端；
          - "mad": 未知 num_adv 时，使用 median + c * MAD 自适应过滤极端异常点。
        - dnc_filter_frac: DnC 预计过滤恶意客户端的比例系数。
          在 "topk" 模式下，保留客户端数约为 n - dnc_filter_frac * num_adv。
        - dnc_mad_scale: 在 "mad" 模式下的阈值系数 c，阈值为 median(score) + c * MAD(score)。
        - dnc_min_keep_ratio: DnC 第一阶段至少保留的客户端比例，避免粗筛过严。
        - lasa_norm_bound: LASA 中逐层范数异常检测阈值。
        - lasa_sign_bound: LASA 中逐层符号一致性检测阈值。
        - lasa_sparsity: LASA 稀疏化比例，越大表示保留的参数越少。
        - min_keep_clients: DnC 结束后至少保留的客户端数，避免过筛。
        - fallback_to_dnc_mean: 若 DnC 后没有客户端可用，是否回退到均值聚合。
        """
        super().__init__(args)
        self.default_defense_params = {
            "dnc_subsample_frac": 0.2,
            "dnc_num_iters": 5,
            "dnc_filter_mode": "topk",
            "dnc_filter_frac": 1.0,
            "dnc_mad_scale": 4.0,
            "dnc_min_keep_ratio": 0.6,
            "lasa_norm_bound": 2,
            "lasa_sign_bound": 1,
            "lasa_sparsity": 0.3,
            "min_keep_clients": 3,
            "fallback_to_dnc_mean": True,
        }
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

    def _log_warning(self, message):
        logger = getattr(self.args, "logger", None)
        if logger is not None:
            logger.warning(message)
        else:
            warnings.warn(message)

    def _sanitize_updates(self, updates, stage_name):
        """
        清洗客户端更新中的 nan/inf，并对极端范数做裁剪，降低后续线性代数失败概率。
        """
        if updates.numel() == 0:
            return updates

        sanitized = updates.clone()
        finite_mask = torch.isfinite(sanitized)
        if not torch.all(finite_mask):
            bad_client_mask = ~torch.all(finite_mask, dim=1)
            bad_count = int(bad_client_mask.sum().item())
            self._log_warning(
                f"MyDefense[{stage_name}] detected {bad_count} client updates with nan/inf; "
                "replacing non-finite entries with 0."
            )
            sanitized = torch.nan_to_num(
                sanitized, nan=0.0, posinf=0.0, neginf=0.0
            )

        client_norms = torch.linalg.norm(sanitized, dim=1)
        finite_norms = client_norms[torch.isfinite(client_norms)]
        if finite_norms.numel() == 0:
            self._log_warning(
                f"MyDefense[{stage_name}] found no finite client norms; falling back to all-zero updates."
            )
            return torch.zeros_like(sanitized)

        median_norm = torch.quantile(finite_norms, q=0.5)
        clip_cap = torch.clamp(median_norm * 10.0, min=1e-6)
        clip_mask = client_norms > clip_cap
        if torch.any(clip_mask):
            clip_count = int(clip_mask.sum().item())
            self._log_warning(
                f"MyDefense[{stage_name}] clipped {clip_count} client updates with exploding norms."
            )
            scales = torch.clamp(clip_cap / (client_norms + 1e-12), max=1.0)
            sanitized = sanitized * scales.unsqueeze(1)

        sanitized = torch.nan_to_num(sanitized, nan=0.0, posinf=0.0, neginf=0.0)
        return sanitized

    def _safe_mean_indices(self, gradient_updates, keep_count):
        """
        当 DnC 失效时，优先按较小范数保留一批客户端，作为投票/均值回退。
        """
        num_clients = int(gradient_updates.shape[0])
        keep_count = max(1, min(num_clients, int(keep_count)))
        if num_clients == 0:
            return []

        safe_updates = torch.nan_to_num(
            gradient_updates, nan=0.0, posinf=0.0, neginf=0.0
        )
        client_norms = torch.linalg.norm(safe_updates, dim=1)
        client_norms = torch.nan_to_num(
            client_norms, nan=float("inf"), posinf=float("inf"), neginf=float("inf")
        )
        return torch.topk(client_norms, keep_count, largest=False).indices.detach().cpu().tolist()

    def _resolve_dnc_keep_counts(self, num_clients):
        """
        统一计算 DnC 的软保底保留规模，既支持 topk，也支持 mad 模式回退。
        """
        min_keep_by_ratio = int(np.ceil(float(self.dnc_min_keep_ratio) * num_clients))
        min_keep = max(
            1,
            min(
                num_clients,
                max(int(self.min_keep_clients), min_keep_by_ratio),
            ),
        )

        if self.dnc_filter_mode == "topk":
            keep_target = int(num_clients - self.dnc_filter_frac * self.args.num_adv)
            keep_target = max(1, min(num_clients, keep_target))
            keep_target = max(keep_target, min_keep)
        else:
            keep_target = min_keep

        return keep_target, min_keep

    def _select_dnc_clients_from_score(self, score, keep_target):
        """
        根据配置的 dnc_filter_mode，从 DnC 分数中选择保留客户端。
        """
        num_clients = int(score.numel())
        keep_target = max(1, min(num_clients, int(keep_target)))

        if self.dnc_filter_mode == "topk":
            if keep_target != num_clients:
                return torch.topk(score, keep_target, largest=False).indices
            return torch.arange(num_clients, device=score.device)

        if self.dnc_filter_mode == "mad":
            med = torch.median(score)
            mad = torch.median(torch.abs(score - med))
            mad = torch.clamp(mad, min=1e-12)
            threshold = med + float(self.dnc_mad_scale) * mad
            selected_idx = torch.nonzero(score <= threshold, as_tuple=False).squeeze(-1)

            if selected_idx.numel() < keep_target:
                self._log_warning(
                    "MyDefense DnC mad-mode retained too few clients; backfilling with lowest-score clients."
                )
                selected_idx = torch.topk(score, keep_target, largest=False).indices

            return selected_idx

        raise ValueError(
            f"Unsupported dnc_filter_mode={self.dnc_filter_mode!r}. Expected 'topk' or 'mad'."
        )

    def _compute_principal_direction(self, centered_grads):
        """
        稳健地提取主方向：先尝试当前设备上的 SVD，再回退到 CPU/float64。
        """
        try:
            _, _, vh = torch.linalg.svd(centered_grads, full_matrices=False)
            return vh[0, :].to(device=centered_grads.device, dtype=centered_grads.dtype)
        except RuntimeError as first_error:
            self._log_warning(
                "MyDefense DnC SVD failed on the current device; retrying on CPU with float64."
            )
            try:
                centered_cpu = centered_grads.detach().to(device="cpu", dtype=torch.float64)
                _, _, vh_cpu = torch.linalg.svd(centered_cpu, full_matrices=False)
                return vh_cpu[0, :].to(
                    device=centered_grads.device,
                    dtype=centered_grads.dtype,
                )
            except RuntimeError as second_error:
                self._log_warning(
                    "MyDefense DnC SVD failed again after CPU fallback; switching to norm-based voting fallback."
                )
                raise RuntimeError(
                    f"SVD failed on device and CPU fallback. first={first_error}; second={second_error}"
                ) from second_error

    def aggregate(self, updates, **kwargs):
        """
        执行两阶段聚合流程。

        整体步骤：
        1. 将客户端更新统一转换为梯度向量表示；
        2. 使用 DnC 做客户端级预筛，得到较可信的客户端集合；
        3. 对保留下来的客户端执行 LASA 风格逐层聚合；
        4. 按当前联邦算法语义封装最终聚合结果。

        参数：
        - updates: 客户端上传的更新集合，形状通常为 [num_clients, num_params]。
        - kwargs["last_global_model"]: 上一轮全局模型，用于：
          - FedAvg 场景下将模型向量转换为梯度；
          - 将向量还原成逐层 state_dict；
          - 最终封装聚合结果。
        """
        # 保存上一轮全局模型，供后续向量/层结构转换与结果封装使用。
        self.global_model = kwargs["last_global_model"]

        # 将不同算法语义下的 updates 统一转换为“梯度向量”形式。
        # 例如：
        # - FedAvg: 客户端上传的是模型参数，需要减去全局模型得到伪梯度；
        # - FedSGD/FedOpt: 客户端上传的本身就是梯度，可直接使用。
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model
        )

        # 后续大量使用 torch 张量操作，因此确保输入一定是 float32 Tensor。
        if not torch.is_tensor(gradient_updates):
            gradient_updates = torch.as_tensor(gradient_updates, dtype=torch.float32)

        # 将梯度移动到与模型一致的设备上，尽量保证后续计算在同一 device 完成。
        try:
            device = next(self.global_model.parameters()).device
        except StopIteration:
            device = gradient_updates.device
        gradient_updates = gradient_updates.to(device=device, dtype=torch.float32)
        gradient_updates = self._sanitize_updates(gradient_updates, stage_name="aggregate")

        # 第一阶段：客户端级预筛。
        benign_idx = self._dnc_filter_clients(gradient_updates)

        # 仅保留通过 DnC 预筛的客户端更新，作为第二阶段 LASA 的输入。
        benign_updates = gradient_updates.index_select(
            0, torch.as_tensor(benign_idx, device=device, dtype=torch.long)
        )

        # 理论上 DnC 后不应为空，但为了工程鲁棒性，这里保留兜底逻辑。
        if benign_updates.shape[0] == 0:
            if self.fallback_to_dnc_mean:
                # 兜底策略：直接对全部梯度做普通均值聚合，避免训练中断。
                return wrapup_aggregated_grads(
                    gradient_updates,
                    self.args.algorithm,
                    self.global_model,
                )
            raise ValueError("MyDefense failed to retain any client after DnC filtering.")

        # 第二阶段：对 DnC 保留下来的客户端做 LASA 风格的逐层精筛与聚合。
        aggregated_gradient = self._lasa_aggregate(benign_updates)

        # _lasa_aggregate 返回的是“已经聚合好的单个梯度向量”，
        # 因此这里使用 aggregated=True，告诉封装函数不要再次求均值。
        return wrapup_aggregated_grads(
            aggregated_gradient,
            self.args.algorithm,
            self.global_model,
            aggregated=True,
        )

    def _dnc_filter_clients(self, gradient_updates):
        """
        使用 DnC 风格方法做客户端级预筛，返回保留下来的客户端索引列表。

        DnC 的核心思想：
        1. 随机抽取一部分参数维度；
        2. 对这些子维度上的客户端梯度做中心化；
        3. 用 SVD 找到最主要的异常方向；
        4. 根据各客户端在该方向上的投影大小给出异常分数；
        5. 重复多轮后，保留稳定低分的客户端。

        这里在原始 DnC 之上加入了两个工程改进：
        - 票数回退：如果多轮交集太小，则按入选次数最多的客户端补足；
        - 最少保留数：至少保留 min_keep_clients 个客户端。
        """
        num_clients, num_param = gradient_updates.shape
        if num_clients == 0:
            return []

        # 根据配置模式计算第一阶段粗筛希望保留的客户端数量。
        keep_target, min_keep = self._resolve_dnc_keep_counts(num_clients)

        # candidate_set:
        # 记录“每一轮都被认为比较正常”的客户端交集。
        candidate_set = set(range(num_clients))

        # candidate_votes:
        # 记录每个客户端在 DnC 多轮中被选中的次数，用于交集不足时的回退。
        candidate_votes = torch.zeros(num_clients, device=gradient_updates.device)
        last_selected = list(range(num_clients))
        last_score = None

        for _ in range(self.dnc_num_iters):
            # 随机子采样参数维度，降低高维计算成本，同时提升对定点伪装攻击的鲁棒性。
            sample_size = int(self.dnc_subsample_frac * num_param)
            if sample_size <= 0:
                raise ValueError("dnc_subsample_frac is too small and samples zero dimensions.")

            param_idx = torch.randperm(
                num_param, device=gradient_updates.device
            )[:sample_size]
            sampled_grads = gradient_updates.index_select(1, param_idx)

            # 先减去均值做中心化，去掉整体平移偏移，只保留相对差异。
            mu = torch.mean(sampled_grads, dim=0)
            centered_grads = sampled_grads - mu
            centered_grads = torch.nan_to_num(
                centered_grads, nan=0.0, posinf=0.0, neginf=0.0
            )

            # 若中心化后几乎退化为零矩阵，则说明当前轮无明显主方向，直接保留全部客户端参与投票。
            centered_norm = torch.linalg.norm(centered_grads)
            if (not torch.isfinite(centered_norm)) or centered_norm.item() <= 1e-12:
                selected_idx = torch.arange(num_clients, device=gradient_updates.device)
                candidate_votes[selected_idx] += 1
                last_selected = selected_idx.detach().cpu().tolist()
                candidate_set = candidate_set.intersection(last_selected)
                continue

            # 对中心化矩阵做 SVD，取第一右奇异向量作为“最主要异常方向”。
            try:
                principal_direction = self._compute_principal_direction(centered_grads)
            except RuntimeError:
                fallback_idx = self._safe_mean_indices(gradient_updates, keep_target)
                candidate_votes[
                    torch.as_tensor(
                        fallback_idx,
                        device=gradient_updates.device,
                        dtype=torch.long,
                    )
                ] += 1
                last_selected = fallback_idx
                candidate_set = candidate_set.intersection(fallback_idx)
                continue

            # 客户端在该方向上的投影越大，越可能是异常点。
            # 使用平方是为了只关心投影幅度，不关心正负方向。
            score = torch.matmul(centered_grads, principal_direction) ** 2
            score = torch.nan_to_num(score, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
            last_score = score

            # 根据配置模式筛选当前轮的粗粒度良性客户端。
            selected_idx = self._select_dnc_clients_from_score(score, keep_target)

            # 记录每轮入选次数，供交集不足时进行票数补足。
            candidate_votes[selected_idx] += 1
            last_selected = selected_idx.detach().cpu().tolist()

            # 与历史候选集合取交集，保留“多轮都稳定正常”的客户端。
            candidate_set = candidate_set.intersection(last_selected)

        # 如果交集结果已经足够大，则优先使用交集，保证 DnC 的严格性。
        if len(candidate_set) >= min_keep:
            return sorted(candidate_set)

        # 如果交集太小，则回退为“按入选票数从高到低”补足到最少保留数量。
        vote_order = torch.argsort(candidate_votes, descending=True).detach().cpu().tolist()
        voted_idx = vote_order[:min_keep]
        if voted_idx:
            return sorted(voted_idx)

        # 理论上上一步一般足够；这里再提供一层按最后一轮得分回退的兜底逻辑。
        if last_score is not None:
            score_order = torch.topk(
                last_score, k=min_keep, largest=False
            ).indices.detach().cpu().tolist()
            if score_order:
                return sorted(score_order)

        # 极端情况下，再退回最后一轮筛出的部分客户端。
        return sorted(last_selected[:min_keep])

    def _lasa_aggregate(self, updates):
        """
        对 DnC 预筛后的客户端执行 LASA 风格的逐层精筛聚合。

        主要步骤：
        1. 将每个客户端的梯度向量还原为逐层 state_dict；
        2. 基于全局客户端范数做一次裁剪，抑制过大梯度；
        3. 对每个客户端做稀疏化，仅保留显著参数；
        4. 对每一层分别执行：
           - 范数异常检测
           - 符号一致性检测
        5. 仅对当前层通过筛选的客户端求平均；
        6. 最后将逐层聚合结果重新拼回一个向量。
        """
        device = updates.device
        updates = self._sanitize_updates(updates, stage_name="lasa")
        num_clients = int(updates.shape[0])

        # 将每个客户端的一维梯度向量恢复成与模型结构一致的 state_dict，
        # 这样后面就可以按层逐一处理。
        dict_form_updates = [
            vec2state(updates[i], self.global_model)
            for i in range(num_clients)
        ]

        # 先按客户端整体范数做裁剪：
        # 范数上限取所有客户端范数的中位数，避免单个超大更新主导聚合结果。
        client_norms = torch.linalg.norm(updates, dim=1)
        median_norm = torch.quantile(client_norms, q=0.5)
        clipped_norms = torch.clamp(client_norms, min=0.0, max=median_norm)
        clipped_updates = (
            updates / (client_norms.reshape(-1, 1) + 1e-12)
        ) * clipped_norms.reshape(-1, 1)

        # 将裁剪后的更新也恢复成逐层形式。
        # 注意：后面做逐层筛选时使用的是“稀疏化后的更新”，
        # 而真正参与聚合平均的是“范数裁剪后的更新”。
        dict_form_clipped_updates = [
            vec2state(clipped_updates[i], self.global_model)
            for i in range(num_clients)
        ]

        # 对每个客户端做稀疏化，只保留较大的参数项，以突出关键模式并减弱噪声。
        for i in range(num_clients):
            dict_form_updates[i] = self.sparse_update(dict_form_updates[i])

        aggregated_state = {}
        for key in dict_form_updates[0].keys():
            # 对 BN 的 num_batches_tracked 缓冲区，不做正常聚合，直接置零占位。
            # 这样能保持向量长度与模型结构一致。
            if "num_batches_tracked" in key:
                aggregated_state[key] = torch.zeros_like(dict_form_updates[0][key])
                continue

            # 收集所有客户端当前层的稀疏化更新，并展平为二维矩阵：
            # [num_clients, num_params_in_this_layer]
            flattened_layer_updates = torch.stack(
                [dict_form_updates[i][key].flatten() for i in range(num_clients)],
                dim=0,
            )
            flattened_layer_updates = torch.nan_to_num(
                flattened_layer_updates, nan=0.0, posinf=0.0, neginf=0.0
            )

            # 过滤器 1：层内 L2 范数异常检测。
            # 某个客户端如果在该层的更新范数明显偏离群体中位数，则可能可疑。
            layer_norms = torch.linalg.norm(flattened_layer_updates, dim=1)
            norm_benign_idx = self.mz_score(layer_norms, self.lasa_norm_bound)

            # 过滤器 2：层内符号一致性检测。
            # 这里用该层更新的符号统计特征作为一个低维摘要，检测符号模式是否偏离多数客户端。
            layer_signs = torch.empty(
                num_clients,
                device=device,
                dtype=updates.dtype,
            )
            for i in range(num_clients):
                sign_feat = torch.sign(dict_form_updates[i][key])
                layer_signs[i] = (
                    0.5 * torch.sum(sign_feat)
                    / (torch.sum(torch.abs(sign_feat)) + 1e-12)
                    * (1 - self.lasa_sparsity)
                )
            sign_benign_idx = self.mz_score(layer_signs, self.lasa_sign_bound)

            # 只有同时通过“范数过滤”和“符号过滤”的客户端，
            # 才被认为是这一层的良性客户端。
            benign_idx = sorted(
                set(norm_benign_idx.detach().cpu().tolist()).intersection(
                    sign_benign_idx.detach().cpu().tolist()
                )
            )

            # 如果某一层筛完一个客户端都没有，则回退为 DnC 保留下来的全体客户端，
            # 避免该层完全无法聚合。
            if len(benign_idx) == 0:
                benign_idx = list(range(num_clients))

            # 当前层真正参与平均的是“裁剪后的完整层更新”，
            # 而不是稀疏化后的检测用更新。
            aggregated_state[key] = torch.mean(
                torch.stack(
                    [dict_form_clipped_updates[i][key] for i in benign_idx], dim=0
                ),
                dim=0,
            )
            aggregated_state[key] = torch.nan_to_num(
                aggregated_state[key], nan=0.0, posinf=0.0, neginf=0.0
            )

        # 将逐层聚合后的 state_dict 重新拼成单个向量，作为最终聚合梯度。
        return state2vec_torch(aggregated_state)

    def sparse_update(self, update):
        """
        对单个客户端更新做稀疏化。

        做法：
        - 仅对卷积层权重（4 维）和全连接层权重（2 维）做稀疏化；
        - 统计所有这些层的参数绝对值；
        - 仅保留绝对值最大的 top-k 参数；
        - 其余位置置零。

        这样可以让后续逐层检测更关注“最显著的更新模式”。
        """
        mask = {}
        for key in update.keys():
            # 只处理 4 维卷积核和 2 维线性层权重。
            # 对偏置项、BN 统计量等保持原样。
            if len(update[key].shape) == 4 or len(update[key].shape) == 2:
                if torch.is_tensor(update[key]):
                    mask[key] = torch.ones_like(update[key], dtype=torch.float32)
                else:
                    mask[key] = np.ones_like(update[key], dtype=np.float32)

        # 若稀疏率为 0，则表示不做稀疏化，直接返回原更新。
        if self.lasa_sparsity == 0.0:
            return update

        # 收集需要参与稀疏化的层的绝对值，用于统一确定 top-k 阈值。
        weight_abs = [
            torch.abs(update[key]) if torch.is_tensor(update[key]) else np.abs(update[key])
            for key in update.keys() if key in mask
        ]
        if not weight_abs:
            return update

        if torch.is_tensor(weight_abs[0]):
            all_scores = torch.cat([value.flatten() for value in weight_abs], dim=0)
            num_topk = max(1, int(all_scores.numel() * (1 - self.lasa_sparsity)))
            kth_largest = torch.topk(all_scores, k=num_topk, sorted=False).values.min()
        else:
            all_scores = np.concatenate([value.flatten() for value in weight_abs])
            num_topk = max(1, int(len(all_scores) * (1 - self.lasa_sparsity)))
            kth_largest = np.partition(all_scores, -num_topk)[-num_topk]

        # 将低于阈值的参数置零，仅保留幅值最大的那些位置。
        for key in mask.keys():
            if torch.is_tensor(update[key]):
                mask[key] = torch.where(
                    torch.abs(update[key]) <= kth_largest,
                    torch.zeros_like(mask[key]),
                    mask[key],
                )
                update[key] = update[key] * mask[key]
            else:
                mask[key] = np.where(np.abs(update[key]) <= kth_largest, 0, mask[key])
                update[key].data *= mask[key]

        return update

    def mz_score(self, values, bound):
        """
        使用基于中位数和标准差的标准化偏离分数进行筛选。

        公式：
            score_i = |x_i - median(x)| / (std(x) + eps)

        含义：
        - 分数越大，说明该样本偏离群体中心越明显；
        - 分数小于 bound 的样本视为“正常”。

        返回：
        - 满足 score < bound 的索引集合。
        """
        if torch.is_tensor(values):
            med = torch.median(values)
            std = torch.std(values, unbiased=False)
            scores = torch.abs((values - med) / (std + 1e-12))
            return torch.nonzero(scores < bound, as_tuple=False).squeeze(-1)

        med, std = np.median(values), np.std(values)
        normalized = np.abs((values - med) / (std + 1e-12))
        return np.argwhere(normalized < bound).squeeze(-1)
