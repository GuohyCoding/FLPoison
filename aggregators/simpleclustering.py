import numpy as np
import torch
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

    def aggregate(self, updates, **kwargs):
        # load global model at last epoch
        self.global_model = kwargs['last_global_model']
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model)

        # sklearn clustering expects CPU numpy arrays
        if torch.is_tensor(gradient_updates):
            cluster_input = gradient_updates.detach().cpu().numpy()
        else:
            cluster_input = np.asarray(gradient_updates)
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
            benign_idx = list(range(len(labels)))
        else:
            benign_label = np.argmax([np.sum(labels == i)
                                     for i in range(n_cluster)])
            benign_idx = np.where(labels == benign_label)[0].tolist()

        return wrapup_aggregated_grads(gradient_updates[benign_idx], self.args.algorithm, self.global_model)
