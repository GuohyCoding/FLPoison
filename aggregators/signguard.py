from aggregators.aggregatorbase import AggregatorBase
import torch
from aggregators import aggregator_registry
from aggregators.aggregator_utils import prepare_grad_updates, wrapup_aggregated_grads
import random


@aggregator_registry
class SignGuard(AggregatorBase):
    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.default_defense_params = {
            "lower_bound": 0.1,
            "upper_bound": 3.0,
            "selection_fraction": 0.1,
            "clustering": "DBSCAN",
            "random_seed": 0,
        }
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

    def aggregate(self, updates, **kwargs):
        self.global_model = kwargs["last_global_model"]
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model
        )

        device = next(self.global_model.parameters()).device
        if torch.is_tensor(gradient_updates):
            gradient_updates = gradient_updates.detach().to(device=device)
        else:
            gradient_updates = torch.as_tensor(
                gradient_updates, device=device, dtype=torch.float32
            )

        s1_benign_idx, median_norm, client_norms = self.norm_filtering(gradient_updates)
        s2_benign_idx = self.sign_clustering(gradient_updates)

        benign_idx = list(set(s1_benign_idx).intersection(s2_benign_idx))
        if len(benign_idx) == 0:
            benign_idx = s1_benign_idx

        benign_idx_t = torch.as_tensor(benign_idx, device=device, dtype=torch.long)
        selected_norms = client_norms.index_select(0, benign_idx_t)
        grads_clipped_norm = torch.clamp(selected_norms, min=0.0, max=median_norm)

        selected_updates = gradient_updates.index_select(0, benign_idx_t)
        benign_clipped = (
            selected_updates / (selected_norms.reshape(-1, 1) + 1e-12)
        ) * grads_clipped_norm.reshape(-1, 1)

        return wrapup_aggregated_grads(
            benign_clipped, self.args.algorithm, self.global_model
        )

    def norm_filtering(self, gradient_updates):
        client_norms = torch.linalg.norm(gradient_updates, dim=1)
        median_norm = torch.quantile(client_norms, q=0.5)
        benign_mask = (client_norms > self.lower_bound * median_norm) & (
            client_norms < self.upper_bound * median_norm
        )
        benign_idx = torch.nonzero(benign_mask, as_tuple=False).reshape(-1)
        return benign_idx.detach().cpu().tolist(), median_norm, client_norms

    def sign_clustering(self, gradient_updates):
        num_clients = int(gradient_updates.shape[0])
        num_para = int(gradient_updates.shape[1])
        num_selected = max(1, int(self.selection_fraction * num_para))

        random.seed(self.random_seed)
        start_idx = random.randint(0, max(0, num_para - num_selected))
        randomized_weights = gradient_updates[:, start_idx: start_idx + num_selected]

        sign_grads = torch.sign(randomized_weights)
        sign_features = torch.empty(
            (num_clients, 3), device=gradient_updates.device, dtype=gradient_updates.dtype
        )

        def sign_feat(target):
            sign_count = (sign_grads == target).sum(dim=1).to(dtype=gradient_updates.dtype)
            sign_ratio = sign_count / num_selected
            return sign_ratio / (torch.max(sign_ratio) + 1e-8)

        sign_features[:, 0] = sign_feat(1)
        sign_features[:, 1] = sign_feat(0)
        sign_features[:, 2] = sign_feat(-1)

        if self.clustering == "KMeans":
            labels = self._kmeans_labels(sign_features, n_clusters=2, max_iters=50)
        elif self.clustering == "DBSCAN":
            labels = self._dbscan_labels(sign_features, eps=0.05, min_samples=3)
        elif self.clustering == "MeanShift":
            labels = self._meanshift_labels(sign_features)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.clustering}")

        valid = labels >= 0
        if not torch.any(valid):
            return list(range(num_clients))

        unique_labels = torch.unique(labels[valid])
        counts = torch.stack([(labels == l).sum() for l in unique_labels])
        benign_label = unique_labels[torch.argmax(counts)]
        benign_idx = torch.nonzero(labels == benign_label, as_tuple=False).reshape(-1)
        return benign_idx.detach().cpu().tolist()

    def _kmeans_labels(self, x, n_clusters=2, max_iters=50):
        n = x.shape[0]
        if n <= n_clusters:
            return torch.arange(n, device=x.device, dtype=torch.long)

        g = torch.Generator(device=x.device)
        g.manual_seed(int(self.random_seed))
        perm = torch.randperm(n, generator=g, device=x.device)
        centers = x[perm[:n_clusters]].clone()

        labels = torch.zeros(n, device=x.device, dtype=torch.long)
        for _ in range(max_iters):
            dists = torch.cdist(x, centers)
            new_labels = torch.argmin(dists, dim=1)
            if torch.equal(new_labels, labels):
                break
            labels = new_labels
            for k in range(n_clusters):
                mask = labels == k
                if torch.any(mask):
                    centers[k] = x[mask].mean(dim=0)
        return labels

    def _dbscan_labels(self, x, eps=0.05, min_samples=3):
        n = x.shape[0]
        d = torch.cdist(x, x)
        neighbors = d <= eps
        core = neighbors.sum(dim=1) >= min_samples

        labels = torch.full((n,), -1, device=x.device, dtype=torch.long)
        visited = torch.zeros(n, device=x.device, dtype=torch.bool)
        cid = 0

        for i in range(n):
            if visited[i] or not core[i]:
                continue
            queue = [i]
            labels[i] = cid
            visited[i] = True
            while queue:
                p = queue.pop()
                nbrs = torch.nonzero(neighbors[p], as_tuple=False).reshape(-1)
                for q in nbrs.tolist():
                    if labels[q] == -1:
                        labels[q] = cid
                    if not visited[q] and core[q]:
                        visited[q] = True
                        queue.append(q)
            cid += 1
        return labels

    def _meanshift_labels(self, x, max_iters=20):
        n = x.shape[0]
        if n <= 1:
            return torch.zeros(n, device=x.device, dtype=torch.long)

        d = torch.cdist(x, x)
        bandwidth = torch.quantile(d, q=0.5)
        bandwidth = torch.clamp(bandwidth, min=1e-3)

        points = x.clone()
        for _ in range(max_iters):
            d = torch.cdist(points, points)
            kernel = (d <= bandwidth).to(x.dtype)
            denom = kernel.sum(dim=1, keepdim=True) + 1e-8
            new_points = torch.matmul(kernel, points) / denom
            if torch.max(torch.linalg.norm(new_points - points, dim=1)) < 1e-4:
                points = new_points
                break
            points = new_points

        labels = torch.full((n,), -1, device=x.device, dtype=torch.long)
        centers = []
        for i in range(n):
            assigned = False
            for cidx, c in enumerate(centers):
                if torch.linalg.norm(points[i] - c) <= bandwidth * 0.5:
                    labels[i] = cidx
                    assigned = True
                    break
            if not assigned:
                centers.append(points[i])
                labels[i] = len(centers) - 1
        return labels
