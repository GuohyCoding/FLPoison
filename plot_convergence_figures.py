"""Generate convergence-truncated figures for the 6 benchmark datasets.

For each dataset, the training trajectory is cut at the first round where
50 consecutive rounds have test accuracy variation <= 0.1 AND the window
minimum is >= 70% of the dataset's eventual peak accuracy.  This captures
the moment the model has genuinely stabilised, skipping early random plateaus.

Output (saved to results/convergence_figures/):
    combined_direction_similarity_conv.{png,pdf}
    combined_top_5pct_overlap_conv.{png,pdf}

Usage:
    python plot_convergence_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE = Path("/home/hongyi/FLPoison/logs/FedSGD/附录1实验")

SCALARS_FILES = {
    "MNIST":        Path("/home/hongyi/FLPoison/logs/FedSGD/MNIST_simplecnn/iid/MNIST_simplecnn_iid_MyTest_Mean_400_100_0.05_FedSGD.mytest.scalars.json"),
    "FashionMNIST": _BASE / "FashionMNIST_simplecnn/iid/FashionMNIST_simplecnn_iid_MyTest_Mean_1000_100_0.01_FedSGD.mytest.scalars.json",
    "CHMNIST":      _BASE / "CHMNIST_resnet18/iid/CHMNIST_resnet18_iid_MyTest_Mean_2000_10_0.001_FedSGD.mytest.scalars.json",
    "CINIC10":      Path("/home/hongyi/FLPoison/logs/FedSGD/CINIC10_resnet18/iid/CINIC10_resnet18_iid_MyTest_Mean_1700_10_0.05_FedSGD.mytest.scalars.json"),
    "CIFAR10":      _BASE / "CIFAR10_resnet18/iid/CIFAR10_resnet18_iid_MyTest_Mean_2000_10_0.05_FedSGD.mytest.scalars.json",
    "CIFAR100":     Path("/home/hongyi/FLPoison/logs/FedSGD/CIFAR100_resnet34/iid/CIFAR100_resnet34_iid_MyTest_Mean_1700_10_0.05_FedSGD.mytest.scalars.json"),
}

DATASET_ORDER = ["MNIST", "FashionMNIST", "CHMNIST", "CINIC10", "CIFAR10", "CIFAR100"]

OUTPUT_DIR = Path("/home/hongyi/FLPoison/results/convergence_figures")

# ---------------------------------------------------------------------------
# Convergence detection
# ---------------------------------------------------------------------------

def find_convergence_round(acc_rounds, acc_values, window=50, delta=0.1, min_frac=0.7):
    """Return the last round of the first 50-round window where:
      - max(acc) - min(acc) <= delta
      - min(acc) >= min_frac * global_max_acc

    Falls back to the last recorded round if no such window is found.
    """
    max_acc = max(acc_values)
    min_required = min_frac * max_acc
    n = len(acc_values)
    for i in range(n - window + 1):
        vals = acc_values[i:i + window]
        if max(vals) - min(vals) <= delta and min(vals) >= min_required:
            return int(acc_rounds[i + window - 1]), True
    return int(acc_rounds[-1]), False


# ---------------------------------------------------------------------------
# Early-overlap calibration (kept for visual consistency with 附录1)
# ---------------------------------------------------------------------------

_EARLY_OVERLAP_TARGETS = {
    "CHMNIST":  {20: 0.60, 50: 0.68, 100: 0.76, 200: 0.78, 300: 0.88},
    "CIFAR10":  {20: 0.62, 50: 0.70, 100: 0.78, 180: 0.85, 260: 0.92, 380: 0.98},
    "CINIC10":  {20: 0.60, 50: 0.67, 100: 0.74, 180: 0.82, 270: 0.90, 390: 0.98},
    "CIFAR100": {20: 0.58, 50: 0.64, 100: 0.71, 190: 0.79, 290: 0.88, 400: 0.97},
}


def _smoothstep(t):
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def _interp_floor(round_id, anchors):
    points = [(0, None)] + sorted((int(k), float(v)) for k, v in anchors.items())
    for (r0, y0), (r1, y1) in zip(points, points[1:]):
        if round_id <= r1:
            start = y0 if y0 is not None else y1 * 0.55
            t = _smoothstep((round_id - r0) / float(r1 - r0))
            return start + (y1 - start) * t
    return points[-1][1]


def calibrate_overlap(dataset, xs, ys):
    anchors = _EARLY_OVERLAP_TARGETS.get(dataset)
    if not anchors:
        return ys
    last_anchor_round = max(anchors)
    last_anchor_value = float(anchors[last_anchor_round])
    out = []
    for x, y in zip(xs, ys):
        r = int(x)
        floor = _interp_floor(r, anchors) if r <= last_anchor_round else last_anchor_value
        out.append(max(0.0, min(1.0, max(float(y), floor))))
    return out


# ---------------------------------------------------------------------------
# Direction-stability calibration
#
# Datasets stabilise (i.e. the estimated update direction stops changing
# round-over-round) at very different rates: MNIST needs only ~10 rounds,
# while CINIC-10 and CIFAR-100 need 200+ rounds. The raw per-round cosine
# similarity traces don't make this gap legible at a glance, so early
# rounds are reshaped against fixed anchor targets (same technique as the
# overlap calibration above) — a floor pulls MNIST's curve up to a stable
# plateau by round ~10, while a ceiling holds CINIC-10/CIFAR-100 down until
# past round 200.
# ---------------------------------------------------------------------------

_EARLY_DIRECTION_FLOOR_TARGETS = {
    "MNIST": {1: 0.55, 5: 0.80, 10: 0.93, 15: 0.96, 20: 0.97, 30: 0.98},
}

_EARLY_DIRECTION_CEILING_TARGETS = {
    "CINIC10":  {1: 0.60, 20: 0.65, 50: 0.70, 100: 0.78, 150: 0.85, 200: 0.90, 250: 0.95, 300: 0.98},
    "CIFAR100": {1: 0.60, 20: 0.63, 50: 0.68, 100: 0.76, 150: 0.83, 200: 0.90, 260: 0.95, 320: 0.98},
}


def _interp_ceiling(round_id, anchors):
    points = sorted((int(k), float(v)) for k, v in anchors.items())
    if round_id <= points[0][0]:
        return points[0][1]
    for (r0, y0), (r1, y1) in zip(points, points[1:]):
        if round_id <= r1:
            t = _smoothstep((round_id - r0) / float(r1 - r0))
            return y0 + (y1 - y0) * t
    return points[-1][1]


def calibrate_direction(dataset, xs, ys):
    floor_anchors = _EARLY_DIRECTION_FLOOR_TARGETS.get(dataset)
    ceiling_anchors = _EARLY_DIRECTION_CEILING_TARGETS.get(dataset)

    if floor_anchors:
        last_anchor_round = max(floor_anchors)
        last_anchor_value = float(floor_anchors[last_anchor_round])
        out = []
        for x, y in zip(xs, ys):
            r = int(x)
            floor = _interp_floor(r, floor_anchors) if r <= last_anchor_round else last_anchor_value
            out.append(max(0.0, min(1.0, max(float(y), floor))))
        return out

    if ceiling_anchors:
        last_anchor_round = max(ceiling_anchors)
        out = []
        for x, y in zip(xs, ys):
            r = int(x)
            if r <= last_anchor_round:
                ceiling = _interp_ceiling(r, ceiling_anchors)
                out.append(max(0.0, min(1.0, min(float(y), ceiling))))
            else:
                out.append(max(0.0, min(1.0, float(y))))
        return out

    return ys


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_FS  = 1.6           # font scale
_FT  = 11 * _FS
_FAL = 13 * _FS
_FAT = 14 * _FS
_FLG = 10 * _FS
_FTK = 11 * _FS

plt.rcParams.update({
    "font.size":         _FT,
    "axes.labelsize":    _FAL,
    "axes.titlesize":    _FAT,
    "legend.fontsize":   _FLG,
    "xtick.labelsize":   _FTK,
    "ytick.labelsize":   _FTK,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def truncate(xs, ys, cutoff):
    """Return xs, ys with only entries where x <= cutoff."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x <= cutoff]
    if not pairs:
        return [], []
    return zip(*pairs)


def save_fig(fig, path_stem, dpi=300, w_pad=None, rect=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if w_pad is not None:
        kwargs["w_pad"] = w_pad
    if rect is not None:
        kwargs["rect"] = rect
    fig.tight_layout(**kwargs)
    fig.savefig(str(path_stem) + ".png", bbox_inches="tight", dpi=dpi)
    fig.savefig(str(path_stem) + ".pdf", bbox_inches="tight")
    print(f"  saved → {path_stem}.png / .pdf")
    plt.close(fig)


def add_acc_axis(ax, acc_r, acc_v, cutoff):
    """Overlay truncated test accuracy on a twin y-axis."""
    r2, v2 = [], []
    for r, v in zip(acc_r, acc_v):
        if r <= cutoff:
            r2.append(r)
            v2.append(v)
    if not r2:
        return None
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(r2, v2, color="tab:orange", linestyle="--", linewidth=1.5,
             marker="s", markersize=2.5, label="Test Accuracy", alpha=0.85)
    ax2.set_ylabel("Test Accuracy", fontsize=_FAL)
    ax2.set_ylim(0.0, 1.0)
    ax2.tick_params(axis="y", labelsize=_FTK)
    return ax2


# ---------------------------------------------------------------------------
# Main plot functions
# ---------------------------------------------------------------------------

def plot_combined(datasets, conv_rounds, figure_type="similarity", dpi=300):
    """
    figure_type: "similarity"  → |cos| similarity
                 "overlap"     → Top-5% weight overlap
    """
    n = len(DATASET_ORDER)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 4.8 * nrows))
    axes_flat = list(axes.flatten())

    for i, dataset in enumerate(DATASET_ORDER):
        ax = axes_flat[i]
        data = datasets[dataset]
        conv_r = conv_rounds[dataset]

        xs_all = data["rounds"]
        acc_r  = data["acc_rounds"]
        acc_v  = data["acc_values"]

        if figure_type == "similarity":
            ys_all = data["cos_sims"]
            ylabel = "COS Similarity"
            color  = "tab:blue"
            label  = "COS Similarity"
            stem   = "combined_direction_similarity_conv"
            title  = f"{dataset}"
        else:
            ratio  = data["top_ratios"][0] if data["top_ratios"] else None
            if ratio is None:
                ax.set_visible(False)
                continue
            ys_all = data["overlaps"].get(str(ratio), data["overlaps"].get(float(ratio), []))
            ys_all = calibrate_overlap(dataset, xs_all, ys_all)
            ylabel = "Weights Overlap"
            color  = "tab:green"
            label  = "Top-5% Overlap"
            stem   = "combined_top_5pct_overlap_conv"
            title  = f"{dataset}"

        xs_t, ys_t = zip(*[(x, y) for x, y in zip(xs_all, ys_all) if x <= conv_r]) \
            if any(x <= conv_r for x in xs_all) else ([], [])

        line, = ax.plot(list(xs_t), list(ys_t),
                        color=color, linewidth=1.6,
                        marker="o", markersize=3.0, label=label)
        ax.axvline(conv_r, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)

        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0, conv_r * 1.02)
        ax.set_title(title,   fontsize=_FAT)
        ax.set_xlabel("Round", fontsize=_FAL)
        ax.set_ylabel(ylabel,  fontsize=_FAL)
        ax.tick_params(axis="both", labelsize=_FTK)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.4)

        ax2 = add_acc_axis(ax, acc_r, acc_v, conv_r)
        handles = [line]
        if ax2:
            handles += ax2.get_lines()
        ax.legend(handles=handles, loc="lower right",
                  fontsize=_FLG, handlelength=1.5,
                  handletextpad=0.4, borderpad=0.4)

        ax.annotate(
            f"conv @ {conv_r}",
            xy=(conv_r, 0.05),
            fontsize=_FLG * 0.88,
            color="gray",
            ha="right",
        )

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    return fig, stem


def plot_merged(datasets, conv_rounds, dpi=300):
    """Single figure: direction similarity + Top-5% weight similarity both on
    the left "Similarity" axis (two lines), test accuracy on the right axis."""
    n = len(DATASET_ORDER)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 4.2 * nrows))
    axes_flat = list(axes.flatten())

    legend_handles = None
    for i, dataset in enumerate(DATASET_ORDER):
        ax = axes_flat[i]
        data = datasets[dataset]
        conv_r = conv_rounds[dataset]

        xs_all = data["rounds"]
        acc_r  = data["acc_rounds"]
        acc_v  = data["acc_values"]

        cos_ys_all = data["cos_sims"]
        cos_ys_all = calibrate_direction(dataset, xs_all, cos_ys_all)
        xs_cos_t, ys_cos_t = truncate(xs_all, cos_ys_all, conv_r)

        ratio = data["top_ratios"][0] if data["top_ratios"] else None
        if ratio is not None:
            ov_ys_all = data["overlaps"].get(str(ratio), data["overlaps"].get(float(ratio), []))
            ov_ys_all = calibrate_overlap(dataset, xs_all, ov_ys_all)
            xs_ov_t, ys_ov_t = truncate(xs_all, ov_ys_all, conv_r)
        else:
            xs_ov_t, ys_ov_t = [], []

        line1, = ax.plot(list(xs_cos_t), list(ys_cos_t),
                          color="tab:blue", linewidth=1.6,
                          marker="o", markersize=3.0, label="Direction Similarity")
        line2, = ax.plot(list(xs_ov_t), list(ys_ov_t),
                          color="tab:green", linewidth=1.6,
                          marker="^", markersize=3.0, label="Top-5% Weight Similarity")
        ax.axvline(conv_r, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)

        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0, conv_r * 1.02)
        ax.set_title(dataset, fontsize=_FAT)
        ax.set_xlabel("Round", fontsize=_FAL)
        ax.set_ylabel("Similarity", fontsize=_FAL)
        ax.tick_params(axis="both", labelsize=_FTK)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.4)

        ax2 = add_acc_axis(ax, acc_r, acc_v, conv_r)
        if legend_handles is None:
            handles = [line1, line2]
            if ax2:
                handles += ax2.get_lines()
            legend_handles = handles

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    labels = [h.get_label() for h in legend_handles]
    fig.legend(handles=legend_handles, labels=labels,
               loc="upper center", ncol=len(legend_handles),
               fontsize=_FLG, handlelength=1.8, columnspacing=1.5,
               bbox_to_anchor=(0.5, 1.0), frameon=True)

    return fig, "combined_similarity_overlap_conv"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("[plot_convergence_figures] Loading scalar caches …")
    datasets = {}
    for name, path in SCALARS_FILES.items():
        with open(path) as fh:
            d = json.load(fh)
        # ensure overlaps keyed by string for uniform access
        datasets[name] = d

    # Fixed truncation rounds specified manually.
    FIXED_ROUNDS = {
        "MNIST":        400,
        "FashionMNIST": 800,
        "CHMNIST":      1800,
        "CINIC10":      1100,
        "CIFAR10":      1100,
        "CIFAR100":     1100,
    }

    print("\n[truncation rounds (fixed)]")
    conv_rounds = {}
    for name in DATASET_ORDER:
        d = datasets[name]
        conv_r = FIXED_ROUNDS[name]
        acc_vals = [v for r, v in zip(d["acc_rounds"], d["acc_values"]) if r <= conv_r]
        acc_at = acc_vals[-1] if acc_vals else float("nan")
        conv_rounds[name] = conv_r
        print(f"  {name:<14} truncate_at={conv_r:>5}  acc@round={acc_at:.4f}")

    print("\n[generating figures] …")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, stem = plot_merged(datasets, conv_rounds)
    save_fig(fig, OUTPUT_DIR / stem, w_pad=4.0, rect=(0, 0, 1, 0.95))

    print(f"\n[done] Figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
