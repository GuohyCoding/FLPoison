import numpy as np
import torch
import warnings
from sklearn.random_projection import SparseRandomProjection
from aggregators.aggregator_utils import prepare_grad_updates, wrapup_aggregated_grads
from aggregators.aggregatorbase import AggregatorBase
from aggregators import aggregator_registry
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth

@aggregator_registry
class SimpleClustering(AggregatorBase):
    """
    Simple majority clustering based on gradient updates.
    """
    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.default_defense_params = {
            "clustering": "DBSCAN",
            "max_cluster_dim": 256,
        }
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

    def _log_warning(self, message):
        logger = getattr(self.args, "logger", None)
        if logger is not None:
            logger.warning(message)
        else:
            warnings.warn(message)

    def aggregate(self, updates, **kwargs):
        # load global model at last epoch
        self.global_model = kwargs['last_global_model']
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model)

        if not torch.is_tensor(gradient_updates):
            gradient_updates = torch.as_tensor(gradient_updates)
        if gradient_updates.ndim != 2:
            raise ValueError(
                f"SimpleClustering expects 2D client updates, got shape {tuple(gradient_updates.shape)}"
            )
        if gradient_updates.shape[0] == 0:
            return wrapup_aggregated_grads(
                gradient_updates, self.args.algorithm, self.global_model
            )

        finite_row_mask = torch.isfinite(gradient_updates).all(dim=1)
        finite_indices = torch.where(finite_row_mask)[0]
        if finite_indices.numel() < gradient_updates.shape[0]:
            bad_count = int((~finite_row_mask).sum().item())
            self._log_warning(
                f"SimpleClustering detected {bad_count} client updates with NaN/Inf; "
                "excluding them from clustering and aggregation."
            )

        if finite_indices.numel() == 0:
            self._log_warning(
                "SimpleClustering found no finite client updates; falling back to sanitized mean aggregation."
            )
            sanitized_updates = torch.nan_to_num(
                gradient_updates, nan=0.0, posinf=0.0, neginf=0.0
            )
            return wrapup_aggregated_grads(
                sanitized_updates, self.args.algorithm, self.global_model
            )

        valid_updates = gradient_updates.index_select(0, finite_indices)

        # sklearn clustering expects CPU numpy arrays
        cluster_input = valid_updates.detach().cpu().numpy()
        if cluster_input.dtype != np.float32 and cluster_input.dtype != np.float64:
            cluster_input = cluster_input.astype(np.float32, copy=False)

        # avoid huge memory use in sklearn pairwise routines
        if cluster_input.ndim == 2 and cluster_input.shape[1] > self.max_cluster_dim:
            projector = SparseRandomProjection(
                n_components=self.max_cluster_dim, random_state=0
            )
            cluster_input = projector.fit_transform(cluster_input)

        if self.clustering == "MeanShift":
            bandwidth = estimate_bandwidth(
                cluster_input, quantile=0.5, n_samples=50)
            grad_cluster = MeanShift(bandwidth=bandwidth,
                                     bin_seeding=True, cluster_all=False)
        elif self.clustering == "DBSCAN":
            grad_cluster = DBSCAN(eps=0.05, min_samples=3)

        grad_cluster.fit(cluster_input)
        labels = grad_cluster.labels_
        n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
        # select the cluster with the majority of benign clients
        if n_cluster <= 0:
            benign_local_idx = list(range(len(labels)))
        else:
            benign_label = np.argmax([np.sum(labels == i)
                                     for i in range(n_cluster)])
            benign_local_idx = np.where(labels == benign_label)[0].tolist()

        if not benign_local_idx:
            self._log_warning(
                "SimpleClustering selected an empty benign cluster; falling back to all finite client updates."
            )
            benign_local_idx = list(range(valid_updates.shape[0]))

        benign_idx = finite_indices[benign_local_idx]
        return wrapup_aggregated_grads(
            gradient_updates[benign_idx], self.args.algorithm, self.global_model
        )
