"""MyTest attacker module for recording FL global direction similarity data.

This module follows the project's attacker format. It registers ``MyTest`` as
an omniscient model-poisoning attacker, but it does not poison training: its
``omniscient`` method returns each attacker client's original benign update.

The recorded quantity is the *cumulative* global direction from the initial
model, ``W_t - W_0`` (the trajectory the global model has travelled up to round
t), rather than a single-round delta. Cumulative displacement cancels per-round
client/sampling noise, so the early-round direction aligns far better with the
converged trajectory ``W_T - W_0`` (used as the target). This is the standard
early-bird / lottery-ticket style measure of "do the early rounds already
reveal the converged principal direction and important parameters".

Instead of embedding compressed direction vectors into the training text log,
this attacker writes them to a dedicated ``*.mytest.pt`` artifact next to the
run log. Only the rounds that the figures actually need are kept: the early
rounds ``1..early_rounds`` and a rolling window of the last ``final_window``
rounds. ``plot_mytest.py`` (and ``run_mytest.py``) read these artifacts
directly, so the training log stays clean and lightweight.
"""

from collections import deque
from pathlib import Path

import torch

from attackers import attacker_registry
from attackers.pbases.mpbase import MPBase
from fl.client import Client
from global_utils import actor


def _parse_top_n_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    if isinstance(value, str):
        return [int(item) for item in value.replace(",", " ").split()]
    return [int(value)]


def _parse_top_ratios(value):
    if value is None:
        return [0.05, 0.10]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    if isinstance(value, str):
        return [float(item) for item in value.replace(",", " ").split()]
    return [float(value)]


def _artifact_path(output):
    """Derive the ``*.mytest.pt`` artifact path from the run's text log path."""
    path = Path(str(output))
    return path.with_name(f"{path.stem}.mytest.pt")


@attacker_registry
@actor("attacker", "model_poisoning", "omniscient")
class MyTest(MPBase, Client):
    """A no-op omniscient attacker that records global direction test data."""

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {
            "early_rounds": 10,
            "final_window": 5,
            "top_ratios": [0.05, 0.10],
            "top_n_list": None,
            "dense_rounds": 50,   # record every round up to this round number
            "record_stride": 10,  # after dense_rounds, record every Nth round
        }
        self.update_and_set_attr()
        self.top_ratios = _parse_top_ratios(self.top_ratios)
        self.top_n_list = _parse_top_n_list(self.top_n_list)
        self.early_rounds = int(self.early_rounds)
        self.final_window = int(self.final_window)
        self.dense_rounds = int(self.dense_rounds)
        self.record_stride = max(1, int(self.record_stride))
        # Initial global model W_0, captured once; cumulative directions are
        # measured relative to it (W_t - W_0).
        self._init_global_weights_vec = None
        self._logged_rounds = set()
        # Direction vectors kept for plotting: the early rounds plus a rolling
        # window of the most recent ``final_window`` rounds.
        self._early = {}
        self._final = deque(maxlen=max(1, self.final_window))
        # Sampled history: dense for early rounds, strided afterwards.
        self._all_rounds = {}
        self._artifact_path = _artifact_path(getattr(self.args, "output", "mytest_run"))

    def omniscient(self, clients):
        """Record the cumulative global direction and keep updates benign.

        Parameters
        ----------
        clients:
            Current FL clients. The method returns the current attacker updates
            unchanged so the configured aggregation receives the same values it
            would have received from benign clients.
        """
        current_global = torch.as_tensor(
            self.global_weights_vec, device="cpu", dtype=torch.float32
        ).flatten()
        current_epoch = int(getattr(self, "global_epoch", 0))

        # Capture W_0 on the very first observation (the initial global model).
        if self._init_global_weights_vec is None:
            self._init_global_weights_vec = current_global.clone()

        # At epoch e the client has incremented global_epoch to e+1, and
        # current_global is M_e = W_e (result of e aggregations). Round e then
        # records the cumulative direction W_e - W_0. Round 0 is skipped (zero).
        completed_round = current_epoch - 1
        if completed_round >= 1 and completed_round not in self._logged_rounds:
            direction = current_global - self._init_global_weights_vec
            self._record_direction(completed_round, direction)
            self._logged_rounds.add(completed_round)

        return self._collect_attacker_updates(clients)

    def _collect_attacker_updates(self, clients):
        attackers = [client for client in clients if client.category == "attacker"]
        updates = []
        for client in attackers:
            update = client.update
            if torch.is_tensor(update):
                updates.append(update.detach().reshape(-1).clone())
            else:
                updates.append(torch.as_tensor(update).reshape(-1).clone())
        if not updates:
            return None
        return torch.stack(updates, dim=0)

    def _record_direction(self, completed_round, direction):
        direction = direction.detach().to(device="cpu", dtype=torch.float32).flatten()

        if completed_round <= self.early_rounds:
            self._early[int(completed_round)] = direction.clone()
        # The rolling window always tracks the most recent rounds, so the final
        # window is correct even if training stops before ``epochs`` rounds.
        self._final.append((int(completed_round), direction.clone()))
        # Sampled history: every round up to dense_rounds, then every record_stride rounds.
        if completed_round <= self.dense_rounds or completed_round % self.record_stride == 0:
            self._all_rounds[int(completed_round)] = direction.clone()

        logger = getattr(self.args, "logger", None)
        if logger is not None:
            logger.info(
                f"MYTEST round={completed_round} "
                f"direction_l2={float(torch.linalg.norm(direction).item()):.6f} "
                f"artifact={self._artifact_path}"
            )

    def finalize(self):
        """Write the artifact to disk once at the end of training."""
        self._save_artifact()

    def _save_artifact(self):
        payload = {
            "tag": "mytest_direction",
            "meta": {
                "direction_mode": "cumulative",  # vectors are W_t - W_0
                "dataset": getattr(self.args, "dataset", ""),
                "model_name": getattr(self.args, "model", ""),
                "total_rounds": int(getattr(self.args, "epochs", 0)),
                "early_rounds": int(self.early_rounds),
                "final_window": int(self.final_window),
                "top_ratios": list(self.top_ratios),
                "top_n_list": list(self.top_n_list),
                "num_parameters": int(self._last_num_parameters()),
                "seed": int(getattr(self.args, "seed", -1)),
            },
            "early": {int(r): vec for r, vec in self._early.items()},
            "final": [(int(r), vec) for r, vec in self._final],
            "all": {int(r): vec for r, vec in self._all_rounds.items()},
        }
        self._artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self._artifact_path)

    def _last_num_parameters(self):
        if self._final:
            return self._final[-1][1].numel()
        if self._early:
            return next(iter(self._early.values())).numel()
        return 0


# Example usage:
# python main.py -config configs/FedSGD_MNIST_config.yaml -att MyTest -num_adv 1 -def Mean
# Optional attack params:
# -attack_params "{'early_rounds': 10, 'final_window': 5, 'top_ratios': [0.05, 0.10]}"
#
# Preferred workflow (one command, multi-dataset comparison figures):
# python run_mytest.py -datasets MNIST FashionMNIST -alg FedSGD --early_rounds 10 --final_window 5
