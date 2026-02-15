from aggregators.aggregatorbase import AggregatorBase
import numpy as np
import torch
from aggregators import aggregator_registry
from fl.models.model_utils import state2vec_torch, vec2state


@aggregator_registry
class LASA(AggregatorBase):
    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.default_defense_params = {
            "norm_bound": 2,
            "sign_bound": 1,
            "sparsity": 0.3,
        }
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

    def aggregate(self, updates, **kwargs):
        self.global_model = kwargs['last_global_model']
        device = next(self.global_model.parameters()).device

        if torch.is_tensor(updates):
            updates = updates.detach().to(device=device)
        else:
            updates = torch.as_tensor(updates, device=device, dtype=torch.float32)

        num_clients = int(updates.shape[0])

        # Restore each client update to state_dict form for layer-wise processing.
        dict_form_updates = [
            vec2state(updates[i], self.global_model)
            for i in range(num_clients)
        ]

        # Norm clipping based on median client norm.
        client_norms = torch.linalg.norm(updates, dim=1)
        median_norm = torch.quantile(client_norms, q=0.5)
        grads_clipped_norm = torch.clamp(client_norms, min=0, max=median_norm)
        grad_clipped = (updates / (client_norms.reshape(-1, 1) + 1e-12)) * \
            grads_clipped_norm.reshape(-1, 1)

        dict_form_grad_clipped = [
            vec2state(grad_clipped[i], self.global_model)
            for i in range(num_clients)
        ]

        # Client-wise top-k sparsification.
        for i in range(num_clients):
            dict_form_updates[i] = self.sparse_update(dict_form_updates[i])

        key_mean_weight = {}
        for key in dict_form_updates[0].keys():
            if 'num_batches_tracked' in key:
                continue

            key_flattened_updates = torch.stack([
                dict_form_updates[i][key].flatten()
                for i in range(num_clients)
            ], dim=0)

            # Filter-1: norm-based MZ score.
            grad_l2norm = torch.linalg.norm(key_flattened_updates, dim=1)
            s1_benign_idx = self.mz_score(grad_l2norm, self.norm_bound)

            # Filter-2: sign-consistency MZ score.
            layer_signs = torch.empty(
                num_clients,
                device=updates.device,
                dtype=updates.dtype,
            )
            for i in range(num_clients):
                sign_feat = torch.sign(dict_form_updates[i][key])
                layer_signs[i] = (
                    0.5 * torch.sum(sign_feat)
                    / (torch.sum(torch.abs(sign_feat)) + 1e-12)
                    * (1 - self.sparsity)
                )
            s2_benign_idx = self.mz_score(layer_signs, self.sign_bound)

            benign_idx = list(
                set(s1_benign_idx.detach().cpu().tolist()).intersection(
                    s2_benign_idx.detach().cpu().tolist()
                )
            )
            if len(benign_idx) == 0:
                benign_idx = list(range(num_clients))

            key_mean_weight[key] = torch.mean(
                torch.stack([dict_form_grad_clipped[i][key] for i in benign_idx], dim=0),
                dim=0,
            )

        return state2vec_torch(key_mean_weight)

    def sparse_update(self, update):
        mask = {}
        for key in update.keys():
            if len(update[key].shape) == 4 or len(update[key].shape) == 2:
                if torch.is_tensor(update[key]):
                    mask[key] = torch.ones_like(update[key], dtype=torch.float32)
                else:
                    mask[key] = np.ones_like(update[key], dtype=np.float32)

        if self.sparsity == 0.0:
            return update

        weight_abs = [
            torch.abs(update[key]) if torch.is_tensor(update[key]) else np.abs(update[key])
            for key in update.keys() if key in mask
        ]

        if torch.is_tensor(weight_abs[0]):
            all_scores = torch.cat([value.flatten() for value in weight_abs], dim=0)
            num_topk = max(1, int(all_scores.numel() * (1 - self.sparsity)))
            kth_largest = torch.topk(all_scores, k=num_topk, sorted=False).values.min()
        else:
            all_scores = np.concatenate([value.flatten() for value in weight_abs])
            num_topk = int(len(all_scores) * (1 - self.sparsity))
            kth_largest = np.partition(all_scores, -num_topk)[-num_topk]

        for key in mask.keys():
            if torch.is_tensor(update[key]):
                mask[key] = torch.where(
                    torch.abs(update[key]) <= kth_largest,
                    torch.zeros_like(mask[key]),
                    mask[key],
                )
                update[key] = update[key] * mask[key]
            else:
                mask[key] = np.where(
                    np.abs(update[key]) <= kth_largest, 0, mask[key]
                )
                update[key].data *= mask[key]

        return update

    def mz_score(self, values, bound):
        if torch.is_tensor(values):
            med = torch.median(values)
            std = torch.std(values, unbiased=False)
            scores = torch.abs((values - med) / (std + 1e-12))
            return torch.nonzero(scores < bound, as_tuple=False).squeeze(-1)

        med, std = np.median(values), np.std(values)
        for i in range(len(values)):
            values[i] = np.abs((values[i] - med) / std)
        return np.argwhere(values < bound).squeeze(-1)
