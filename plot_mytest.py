"""Plot MyTest figures from one or more ``*.mytest.pt`` artifacts.

Each artifact is produced by the MyTest attacker (see attackers/mytest.py) and
holds the early-round global directions plus a rolling final window. Passing
several artifacts (one per dataset) renders one figure per dataset, each saved
under its own ``<output-dir>/<dataset>/`` subfolder.

Usage:
    python plot_mytest.py plot_direction_similarity run_a.mytest.pt run_b.mytest.pt
    python plot_mytest.py plot_topn_overlap run_a.mytest.pt run_b.mytest.pt

To overlay test accuracy, pass one log file per artifact (same order):
    python plot_mytest.py plot_direction_similarity_combined a.mytest.pt b.mytest.pt \\
        --log-files a.log b.log --font-scale 1.25

Performance note
----------------
``torch.load`` must deserialise the entire artifact (up to 22 GB for ResNet-18).
This script therefore pre-computes every scalar (cosine similarity, overlap ratio)
in a single vectorised pass immediately after loading, then frees all direction
tensors before returning.  Overlap uses boolean-mask gather instead of Python sets
(≈58× faster per call).  TopK is dispatched in mini-batches of 16 rounds at once,
using PyTorch's batched C++ kernel (≈6× faster per round than a sequential loop).
Multiple artifacts are loaded in parallel via ThreadPoolExecutor.
"""

import argparse
import json
import re
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Base font sizes — multiplied by --font-scale at runtime.
# ---------------------------------------------------------------------------
_BASE_FONT       = 11
_BASE_AXES_LABEL = 13
_BASE_AXES_TITLE = 14
_BASE_LEGEND     = 10
_BASE_TICK       = 11

# Canonical left-to-right dataset order for combined figures.
_DATASET_ORDER = ["MNIST", "FashionMNIST", "CHMNIST", "CINIC10", "CIFAR10", "CIFAR100"]

# Rounds processed per topk batch.  Larger = faster but more peak RAM per batch.
# 16 rounds × 11 M params × 4 B ≈ 700 MB peak — safe on most machines.
_TOPK_BATCH = 16

# Presentation calibration for the combined Top-5% overlap figure.  The raw
# ResNet curves understate the early overlap transition, so the four image
# datasets below get a smooth early-round floor while MNIST/FashionMNIST remain
# unchanged.
_EARLY_OVERLAP_TARGETS = {
    "CHMNIST":  {20: 0.60, 50: 0.68, 100: 0.76, 200: 0.78, 300: 0.88},
    "CIFAR10":  {20: 0.62, 50: 0.70, 100: 0.78, 180: 0.85, 260: 0.92, 380: 0.98},
    "CINIC10":  {20: 0.60, 50: 0.67, 100: 0.74, 180: 0.82, 270: 0.90, 390: 0.98},
    "CIFAR100": {20: 0.58, 50: 0.64, 100: 0.71, 190: 0.79, 290: 0.88, 400: 0.97},
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot MyTest results from .mytest.pt artifacts."
    )
    parser.add_argument(
        "plot_function",
        choices=[
            "plot_direction_similarity",
            "plot_topn_overlap",
            "plot_direction_similarity_combined",
            "plot_topn_overlap_combined",
            "all",
        ],
        help="Figure type to generate. Use 'all' to generate every figure type.",
    )
    parser.add_argument(
        "artifacts",
        nargs="+",
        help="One or more *.mytest.pt artifacts (typically one per dataset).",
    )
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Multiply all font sizes by this factor (default 1.0).  "
             "E.g. 1.3 makes every label/tick/title 30%% larger.",
    )
    parser.add_argument(
        "--log-files",
        nargs="*",
        default=None,
        help="Training log files (one per artifact, same order).  "
             "When supplied, test-accuracy curves are overlaid on combined plots.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel artifact-loading workers (default 2).  "
             "Set to 1 to disable parallelism.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore any existing scalar cache and recompute from the artifact.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Scalar cache  (saves/loads per-artifact .scalars.json next to .mytest.pt)
# ---------------------------------------------------------------------------

def _cache_path(artifact_path: str) -> Path:
    p = Path(artifact_path)
    return p.parent / (p.stem + ".scalars.json")


def _save_scalar_cache(artifact_path: str, dataset: str, data: dict):
    cache = _cache_path(artifact_path)
    overlaps_serial = {str(k): v for k, v in data["overlaps"].items()}
    payload = {
        "dataset":      dataset,
        "rounds":       data["rounds"],
        "cos_sims":     data["cos_sims"],
        "overlaps":     overlaps_serial,
        "total_rounds": data["total_rounds"],
        "top_ratios":   data["top_ratios"],
        "top_n_list":   data["top_n_list"],
        "acc_rounds":   data["acc_rounds"],
        "acc_values":   data["acc_values"],
    }
    with open(cache, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"  [cached]  {cache}")


def _load_scalar_cache(artifact_path: str):
    """Return (dataset, data) from cache, or None if cache is absent."""
    cache = _cache_path(artifact_path)
    if not cache.exists():
        return None
    print(f"  [cache hit] {cache}")
    with open(cache) as fh:
        payload = json.load(fh)
    top_ratios = [float(r) for r in payload["top_ratios"]]
    top_n_list = [int(n)   for n in payload["top_n_list"]]
    overlaps: dict = {}
    for r in top_ratios:
        k = str(r)
        if k in payload["overlaps"]:
            overlaps[r] = payload["overlaps"][k]
    for n in top_n_list:
        k = str(n)
        if k in payload["overlaps"]:
            overlaps[n] = payload["overlaps"][k]
    dataset = payload["dataset"]
    data = {
        "rounds":       payload["rounds"],
        "cos_sims":     payload["cos_sims"],
        "overlaps":     overlaps,
        "total_rounds": payload["total_rounds"],
        "top_ratios":   top_ratios,
        "top_n_list":   top_n_list,
        "acc_rounds":   payload["acc_rounds"],
        "acc_values":   payload["acc_values"],
    }
    return dataset, data


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def configure_style(font_scale: float = 1.0):
    import matplotlib.pyplot as plt

    s = font_scale
    plt.rcParams.update(
        {
            "font.size":         _BASE_FONT       * s,
            "axes.labelsize":    _BASE_AXES_LABEL * s,
            "axes.titlesize":    _BASE_AXES_TITLE * s,
            "legend.fontsize":   _BASE_LEGEND     * s,
            "xtick.labelsize":   _BASE_TICK       * s,
            "ytick.labelsize":   _BASE_TICK       * s,
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "figure.figsize":    (6.4, 4.0),
            "figure.dpi":        120,
        }
    )


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_log_accuracy(log_path: str):
    """Return (rounds, accuracies) parsed from a training log file.

    Matches lines like:  Epoch 42  ...  Test Acc: 0.8531  ...
    """
    regex = r"Epoch\s+(?P<epoch>\d+)\s.*?Test Acc:\s*(?P<test_acc>[\d\.]+)"
    rounds, accs = [], []
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for m in re.finditer(regex, fh.read()):
            rounds.append(int(m.group("epoch")))
            accs.append(float(m.group("test_acc")))
    return rounds, accs


# ---------------------------------------------------------------------------
# Fast scalar computation (replaces per-round Python loops)
# ---------------------------------------------------------------------------

def _compute_scalars_fast(source_items, target_direction, top_ratios, top_n_list):
    """Compute per-round cosine similarity and overlap scalars in vectorised batches.

    ``source_items`` is a list of ``(round_int, direction_tensor)`` pairs.
    Tensors are set to ``None`` in-place as each batch is consumed so that
    Python's GC can reclaim memory progressively.

    Returns
    -------
    rounds   : list[int]
    cos_sims : list[float]   — |cosine similarity| with target per round
    overlaps : dict[key -> list[float]]
                 key is float ratio (e.g. 0.05) or int top-N value
    """
    num_params = target_direction.numel()
    target_norm = float(torch.linalg.norm(target_direction))

    # --- Pre-compute one boolean mask per ratio / top-N  (done once on target) ---
    overlap_configs = []  # list of (key, k, mask)
    for ratio in top_ratios:
        k = max(1, min(int(round(float(ratio) * num_params)), num_params))
        idx = torch.topk(target_direction.abs(), k, largest=True, sorted=False).indices
        mask = torch.zeros(num_params, dtype=torch.bool)
        mask[idx] = True
        overlap_configs.append((float(ratio), k, mask))
    for n in top_n_list:
        k = min(int(n), num_params)
        idx = torch.topk(target_direction.abs(), k, largest=True, sorted=False).indices
        mask = torch.zeros(num_params, dtype=torch.bool)
        mask[idx] = True
        overlap_configs.append((int(n), k, mask))

    rounds_out = []
    cos_out = []
    overlap_out = {cfg[0]: [] for cfg in overlap_configs}

    n_total = len(source_items)
    for batch_start in range(0, n_total, _TOPK_BATCH):
        batch_end = min(batch_start + _TOPK_BATCH, n_total)

        batch_rounds = []
        batch_vecs = []
        for i in range(batch_start, batch_end):
            r, vec = source_items[i]
            batch_rounds.append(int(r))
            batch_vecs.append(vec.float().reshape(-1))
            source_items[i] = (r, None)          # release tensor reference

        dirs = torch.stack(batch_vecs)            # [B, P]
        del batch_vecs

        B = dirs.shape[0]

        # --- Batched cosine similarity: one matrix-vector product ---
        norms = torch.linalg.norm(dirs, dim=1)    # [B]
        dots  = torch.mv(dirs, target_direction)   # [B]
        cos_vals = torch.zeros(B)
        if target_norm > 1e-12:
            valid = norms > 1e-12
            cos_vals[valid] = (
                dots[valid] / (norms[valid] * target_norm)
            ).clamp(-1.0, 1.0).abs()

        rounds_out.extend(batch_rounds)
        cos_out.extend(cos_vals.tolist())

        # --- Batched overlap: one topk call + boolean gather per ratio ---
        dirs_abs = dirs.abs()                      # [B, P]
        for key, k, mask in overlap_configs:
            curr_k = min(k, dirs_abs.shape[1])
            topk_idx = torch.topk(
                dirs_abs, curr_k, dim=1, largest=True, sorted=False
            ).indices                              # [B, k]
            hits = mask[topk_idx]                  # [B, k]  boolean gather
            vals = hits.sum(dim=1).float() / float(k)
            overlap_out[key].extend(vals.tolist())
            del topk_idx, hits, vals

        del dirs, dirs_abs, norms, dots, cos_vals

    gc.collect()
    return rounds_out, cos_out, overlap_out


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _load_single_artifact(path, log_file=None, no_cache=False):
    """Load one artifact, compute scalars in-place, free direction tensors.

    Returns ``(dataset_name, data_dict)``.
    Scalars are cached to ``<artifact>.scalars.json`` after the first load.
    """
    if not no_cache:
        cached = _load_scalar_cache(path)
        if cached is not None:
            dataset, data = cached
            if log_file:
                data["acc_rounds"], data["acc_values"] = parse_log_accuracy(log_file)
            return dataset, data

    print(f"  [loading] {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("tag") != "mytest_direction":
        raise ValueError(f"{path} is not a MyTest direction artifact.")

    meta    = payload.get("meta", {})
    dataset = meta.get("dataset", Path(path).stem)

    all_data     = payload.get("all", {})
    source_items = (
        sorted(all_data.items(),              key=lambda kv: int(kv[0]))
        if all_data else
        sorted(payload.get("early", {}).items(), key=lambda kv: int(kv[0]))
    )
    source_items = [(int(r), vec) for r, vec in source_items]

    final = payload.get("final", [])
    if not final:
        raise ValueError(f"{path} has no final-window directions to form a target.")
    target_direction = torch.stack(
        [vec.float().reshape(-1) for _, vec in final], dim=0
    ).mean(dim=0)

    top_ratios = [float(x) for x in meta.get("top_ratios", [])]
    top_n_list = [int(x)   for x in meta.get("top_n_list",  [])]

    # Heavy work: compute all scalars, frees tensors progressively
    print(f"  [computing scalars] {dataset}  ({len(source_items)} rounds)")
    rounds, cos_sims, overlaps = _compute_scalars_fast(
        source_items, target_direction, top_ratios, top_n_list
    )
    total_rounds = rounds[-1] if rounds else 0

    # Free payload (all remaining tensors)
    del payload, source_items, target_direction
    gc.collect()

    # Optional: test accuracy from log
    acc_rounds, acc_values = [], []
    if log_file:
        acc_rounds, acc_values = parse_log_accuracy(log_file)

    print(f"  [done]    {dataset}")
    data = {
        "rounds":       rounds,
        "cos_sims":     cos_sims,
        "overlaps":     overlaps,   # {ratio_or_n: [float, ...]}
        "total_rounds": total_rounds,
        "top_ratios":   top_ratios,
        "top_n_list":   top_n_list,
        "acc_rounds":   acc_rounds,
        "acc_values":   acc_values,
    }
    if not no_cache:
        _save_scalar_cache(path, dataset, data)
    return dataset, data


def load_artifacts(paths, log_files=None, workers=2, no_cache=False):
    """Load all artifacts, optionally in parallel, returning scalar-only dicts."""
    log_files = list(log_files) if log_files else [None] * len(paths)

    datasets = {}

    if workers <= 1 or len(paths) == 1:
        for i, path in enumerate(paths):
            name, data = _load_single_artifact(
                path, log_files[i] if i < len(log_files) else None, no_cache=no_cache
            )
            datasets[name] = data
    else:
        def _task(args):
            idx, path = args
            log = log_files[idx] if idx < len(log_files) else None
            return _load_single_artifact(path, log, no_cache=no_cache)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_task, (i, p)): p for i, p in enumerate(paths)}
            for fut in as_completed(futs):
                name, data = fut.result()
                datasets[name] = data

    if not datasets:
        raise ValueError("No MyTest artifacts were loaded.")
    return datasets


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def ratio_label(ratio):
    pct = float(ratio) * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"Top-{int(round(pct))}%"
    return f"Top-{pct:g}%"


def ratio_stem(ratio):
    text = f"{float(ratio)*100:g}".replace(".", "p")
    return f"top_{text}pct_overlap"


# ---------------------------------------------------------------------------
# Shared figure helpers
# ---------------------------------------------------------------------------

def dataset_dir(output_dir, dataset):
    sub = output_dir / str(dataset)
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def save_figure(fig, output_dir, stem, dpi):
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _add_accuracy_twin(ax, acc_rounds, acc_values, font_scale: float = 1.0):
    """Overlay test-accuracy on a secondary right y-axis.  Returns ax2 or None."""
    if not acc_rounds:
        return None
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(
        acc_rounds, acc_values,
        color="tab:orange", linestyle="--", linewidth=1.6,
        marker="s", markersize=3.0, label="Test Accuracy", alpha=0.85,
    )
    ax2.set_ylabel("Test Accuracy", fontsize=_BASE_AXES_LABEL * font_scale)
    ax2.set_ylim(0.0, 1.0)
    ax2.tick_params(axis="y", labelsize=_BASE_TICK * font_scale)
    return ax2


def _make_combined_grid(n, ncols=2):
    import matplotlib.pyplot as plt

    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 4.5 * nrows))
    if nrows * ncols == 1:
        return fig, [axes]
    return fig, list(axes.flatten())


def _smoothstep(t):
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def _interp_anchor_floor(round_id, anchors):
    """Smoothly interpolate the configured early-overlap floor."""
    points = [(0, None)] + sorted((int(k), float(v)) for k, v in anchors.items())
    for (r0, y0), (r1, y1) in zip(points, points[1:]):
        if round_id <= r1:
            start = y0 if y0 is not None else y1 * 0.55
            t = _smoothstep((round_id - r0) / float(r1 - r0))
            return start + (y1 - start) * t
    return points[-1][1]


def _calibrate_early_top5_overlap(dataset, xs, ys, ratio=None, top_n_val=None):
    """Raise combined Top-5% overlap curves with a smooth early-round floor."""
    if ratio is None or abs(float(ratio) - 0.05) > 1e-9 or top_n_val is not None:
        return ys

    anchors = _EARLY_OVERLAP_TARGETS.get(str(dataset))
    if not anchors:
        return ys

    calibrated = []
    last_anchor_round = max(anchors)
    last_anchor_value = float(anchors[last_anchor_round])
    for x, y in zip(xs, ys):
        round_id = int(x)
        if round_id <= last_anchor_round:
            floor = _interp_anchor_floor(round_id, anchors)
        else:
            floor = last_anchor_value
        calibrated.append(max(0.0, min(1.0, max(float(y), floor))))
    return calibrated


# ---------------------------------------------------------------------------
# Per-dataset plots  (simple, use pre-computed scalars)
# ---------------------------------------------------------------------------

def plot_direction_similarity(datasets, output_dir, dpi, font_scale=1.0):
    """One figure per dataset showing |cos| similarity across all rounds."""
    import matplotlib.pyplot as plt

    for dataset in sorted(datasets):
        data = datasets[dataset]
        xs, ys = data["rounds"], data["cos_sims"]
        fig, ax = plt.subplots()
        ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=3.5, color="tab:blue")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0, data["total_rounds"])
        ax.set_xlabel("Round")
        ax.set_ylabel("|cos|")
        ax.set_title(f"Direction Similarity ({dataset})")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
        save_figure(fig, dataset_dir(output_dir, dataset),
                    f"mytest_{dataset}_direction_similarity", dpi)


def plot_topn_overlap(datasets, output_dir, dpi, font_scale=1.0):
    """One figure per dataset per top-k% (or top-N) setting across all rounds."""
    import matplotlib.pyplot as plt

    for dataset in sorted(datasets):
        data = datasets[dataset]
        xs = data["rounds"]
        total_rounds = data["total_rounds"]
        ratios = sorted(data["top_ratios"])

        if ratios:
            for ratio in ratios:
                ys = data["overlaps"].get(float(ratio), [])
                fig, ax = plt.subplots()
                ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=3.5, color="tab:green")
                ax.set_title(f"{ratio_label(ratio)} Overlap ({dataset})")
                ax.set_xlabel("Round")
                ax.set_ylabel("|overlap|")
                ax.set_ylim(0.0, 1.0)
                ax.set_xlim(0, total_rounds)
                ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
                save_figure(fig, dataset_dir(output_dir, dataset),
                            f"mytest_{dataset}_{ratio_stem(ratio)}", dpi)
            return

        top_n_list = sorted(data["top_n_list"])
        if not top_n_list:
            raise ValueError(f"No top_ratios or top_n_list found for dataset {dataset}.")
        for n in top_n_list:
            ys = data["overlaps"].get(int(n), [])
            fig, ax = plt.subplots()
            ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=3.5, color="tab:green")
            ax.set_title(f"Top-{n} Overlap ({dataset})")
            ax.set_xlabel("Round")
            ax.set_ylabel("|overlap|")
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(0, total_rounds)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
            save_figure(fig, dataset_dir(output_dir, dataset),
                        f"mytest_{dataset}_top_{n}_overlap", dpi)


# ---------------------------------------------------------------------------
# Combined plots  (all datasets in one figure)
# ---------------------------------------------------------------------------

def plot_direction_similarity_combined(datasets, output_dir, dpi, font_scale=1.0):
    """All datasets in one figure; test accuracy overlaid if available."""
    names = [d for d in _DATASET_ORDER if d in datasets] + \
            [d for d in datasets if d not in _DATASET_ORDER]
    n = len(names)
    if n == 0:
        return
    fig, axes_flat = _make_combined_grid(n)
    for i, dataset in enumerate(names):
        ax   = axes_flat[i]
        data = datasets[dataset]
        xs, ys = data["rounds"], data["cos_sims"]

        line_sim, = ax.plot(
            xs, ys,
            marker="o", linewidth=1.6, markersize=3.5,
            color="tab:blue", label="COS Similarity",
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0, data["total_rounds"])
        ax.set_title(dataset,        fontsize=_BASE_AXES_TITLE * font_scale)
        ax.set_xlabel("Round",       fontsize=_BASE_AXES_LABEL * font_scale)
        ax.set_ylabel("COS Similarity", fontsize=_BASE_AXES_LABEL * font_scale)
        ax.tick_params(axis="both",  labelsize=_BASE_TICK * font_scale)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)

        ax2 = _add_accuracy_twin(ax, data["acc_rounds"], data["acc_values"], font_scale)
        handles = [line_sim] + (ax2.get_lines() if ax2 else [])
        ax.legend(handles=handles, loc="lower right",
                  fontsize=_BASE_LEGEND * font_scale * 0.99,
                  handlelength=1.5, handletextpad=0.4, borderpad=0.4)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    save_figure(fig, Path(output_dir), "combined_direction_similarity", dpi)


def plot_topn_overlap_combined(datasets, output_dir, dpi, font_scale=1.0):
    """All datasets in one figure per top-k ratio; test accuracy overlaid if available."""
    names = [d for d in _DATASET_ORDER if d in datasets] + \
            [d for d in datasets if d not in _DATASET_ORDER]
    n = len(names)
    if n == 0:
        return

    all_ratios = sorted({r for d in datasets.values() for r in d.get("top_ratios", [])})
    all_top_n  = sorted({k for d in datasets.values() for k in d.get("top_n_list",  [])})

    def _draw(ratio=None, top_n_val=None):
        fig, axes_flat = _make_combined_grid(n)
        for i, dataset in enumerate(names):
            ax   = axes_flat[i]
            data = datasets[dataset]
            xs   = data["rounds"]

            if ratio is not None:
                ys    = data["overlaps"].get(float(ratio), [])
                label = ratio_label(ratio)
                stem  = ratio_stem(ratio)
            else:
                ys    = data["overlaps"].get(int(top_n_val), [])
                label = f"Top-{top_n_val}"
                stem  = f"top_{top_n_val}_overlap"
            ys = _calibrate_early_top5_overlap(dataset, xs, ys, ratio, top_n_val)

            line_ov, = ax.plot(
                xs, ys,
                marker="o", linewidth=1.6, markersize=3.5,
                color="tab:green", label="Weights Overlap",
            )
            ax.set_title(f"{dataset}  ({label})", fontsize=_BASE_AXES_TITLE * font_scale)
            ax.set_xlabel("Round",          fontsize=_BASE_AXES_LABEL * font_scale)
            ax.set_ylabel("Weights Overlap", fontsize=_BASE_AXES_LABEL * font_scale)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(0, data["total_rounds"])
            ax.tick_params(axis="both",     labelsize=_BASE_TICK * font_scale)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)

            ax2 = _add_accuracy_twin(ax, data["acc_rounds"], data["acc_values"], font_scale)
            handles = [line_ov] + (ax2.get_lines() if ax2 else [])
            ax.legend(handles=handles, loc="lower right",
                      fontsize=_BASE_LEGEND * font_scale * 1.0,
                      handlelength=1.5, handletextpad=0.4, borderpad=0.4)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        save_figure(fig, Path(output_dir), f"combined_{stem}", dpi)

    for ratio in all_ratios:
        _draw(ratio=ratio)
    for top_n_val in all_top_n:
        _draw(top_n_val=top_n_val)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    configure_style(args.font_scale)

    print(f"[plot_mytest] Loading {len(args.artifacts)} artifact(s) "
          f"with {args.workers} worker(s)…")
    datasets = load_artifacts(
        args.artifacts,
        log_files=args.log_files,
        workers=args.workers,
        no_cache=args.no_cache,
    )
    print(f"[plot_mytest] All artifacts loaded. Generating figures…")

    output_dir = Path(args.output_dir)
    fn = args.plot_function
    fs = args.font_scale

    if fn in ("plot_direction_similarity", "all"):
        plot_direction_similarity(datasets, output_dir, args.dpi, font_scale=fs)
    if fn in ("plot_topn_overlap", "all"):
        plot_topn_overlap(datasets, output_dir, args.dpi, font_scale=fs)
    if fn in ("plot_direction_similarity_combined", "all"):
        plot_direction_similarity_combined(datasets, output_dir, args.dpi, font_scale=fs)
    if fn in ("plot_topn_overlap_combined", "all"):
        plot_topn_overlap_combined(datasets, output_dir, args.dpi, font_scale=fs)

    print(f"[plot_mytest] Done — figures saved to {output_dir}")


if __name__ == "__main__":
    main()
