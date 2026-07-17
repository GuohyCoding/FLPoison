"""Train COMPASS and/or plot matched-filter-score vs test-accuracy.

Three usage modes
-----------------
1. Batch train + plot  (mirrors batchrun.py, adds --batch flag):
       python plot_mached_filter.py --batch \\
           -data MNIST -model simplecnn -algorithms FedSGD \\
           -distributions non-iid -attacks COMPASS \\
           -defenses Mean Krum TrimmedMean ... -gidx 0 -maxp 4

2. Single train + plot  (mirrors main.py):
       python plot_mached_filter.py \\
           --config configs/FedSGD_MNIST_config.yaml \\
           --attack COMPASS --defense Mean --algorithm FedSGD \\
           --dataset MNIST --model simplecnn \\
           --epochs 200 --learning_rate 0.05 --num_clients 100 --num_adv 20 \\
           -gidx 0

3. Plot only  (pass one or more existing CSV paths):
       python plot_mached_filter.py logs/.../foo.compass_metrics.csv [bar.csv ...]

Per-defense output  (logs/…/<distribution>/<defense>/):
    matched_filter_timeseries.pdf / .png

Combined output  (logs/…/<distribution>/):
    combined_matched_filter.pdf / .png   ← all defenses in one grid figure
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

def _is_csv_mode(argv):
    positional = [a for a in argv if not a.startswith("-")]
    return positional and all(a.endswith(".csv") for a in positional)


def _is_batch_mode(argv):
    return "--batch" in argv or "-batch" in argv


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _get_arg(argv, *flags, default=None):
    for flag in flags:
        for i, a in enumerate(argv):
            if a == flag and i + 1 < len(argv):
                return argv[i + 1]
    return default


def _infer_csv_path(argv):
    """Reconstruct the CSV path for a single main.py run."""
    algorithm    = _get_arg(argv, "-alg",        "--algorithm",    default="FedSGD")
    dataset      = _get_arg(argv, "-data",        "--dataset",      default="MNIST")
    model        = _get_arg(argv, "-model",       "--model",        default="simplecnn")
    distribution = _get_arg(argv, "-dtb",         "--distribution", default="iid")
    attack       = _get_arg(argv, "-att",         "--attack",       default="COMPASS")
    defense      = _get_arg(argv, "-def",         "--defense",      default="Mean")
    epochs       = _get_arg(argv, "-e",           "--epochs",       default="200")
    num_clients  = _get_arg(argv, "-num_clients", "--num_clients",  default="100")
    lr           = _get_arg(argv, "-lr",          "--learning_rate",default="0.05")
    stem = (f"{dataset}_{model}_{distribution}_{attack}_{defense}"
            f"_{epochs}_{num_clients}_{lr}_{algorithm}")
    txt = Path(f"./logs/{algorithm}/{dataset}_{model}/{distribution}/{stem}.txt")
    return txt.with_suffix(".compass_metrics.csv")


def _infer_csv_paths_batch(argv):
    """Reconstruct all CSV paths for a batchrun.py run (one per defense)."""
    from batchrun import get_configs   # reuse the same hyperparameter table

    def _get_list(argv, *flags):
        for flag in flags:
            for i, a in enumerate(argv):
                if a == flag:
                    vals = []
                    for j in range(i + 1, len(argv)):
                        if argv[j].startswith("-"):
                            break
                        vals.append(argv[j])
                    if vals:
                        return vals
        return []

    dataset      = _get_arg(argv, "-data",          "--dataset",       default="MNIST")
    model        = _get_arg(argv, "-model",          "--model",         default="simplecnn")
    algorithms   = _get_list(argv, "-algorithms",    "--algorithms")   or ["FedSGD"]
    distributions= _get_list(argv, "-distributions", "--distributions") or ["iid"]
    attacks      = _get_list(argv, "-attacks",       "--attacks")      or ["COMPASS"]
    defenses     = _get_list(argv, "-defenses",      "--defenses")     or ["Mean"]

    paths = []
    for algorithm in algorithms:
        config_file = f"{algorithm}_{dataset}_config.yaml"
        for distribution in distributions:
            for attack in attacks:
                for defense in defenses:
                    num_clients, epoch, lr = get_configs(
                        dataset, algorithm, distribution, defense)
                    stem = (f"{dataset}_{model}_{distribution}_{attack}_{defense}"
                            f"_{epoch}_{num_clients}_{lr}_{algorithm}")
                    txt  = Path(f"./logs/{algorithm}/{dataset}_{model}"
                                f"/{distribution}/{stem}.txt")
                    paths.append(txt.with_suffix(".compass_metrics.csv"))
    return paths


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0].astype(int), data[:, 1], data[:, 2]


# ---------------------------------------------------------------------------
# Label / directory helpers
# ---------------------------------------------------------------------------

def _defense_from_label(label: str) -> str:
    parts = label.split("_")
    try:
        return parts[-5]
    except IndexError:
        return label


def _per_defense_dir(csv_path: Path) -> Path:
    label   = csv_path.stem.replace(".compass_metrics", "")
    defense = _defense_from_label(label)
    d = csv_path.parent / defense
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Convergence detection
# ---------------------------------------------------------------------------

def find_convergence_end(accs, window: int = 50, threshold: float = 0.1) -> int:
    """Return the first index i such that accs[i-window:i] has range < threshold.

    Returns len(accs) if convergence is never reached.
    """
    for i in range(window, len(accs) + 1):
        if accs[i - window:i].max() - accs[i - window:i].min() < threshold:
            return i
    return len(accs)


# ---------------------------------------------------------------------------
# Per-defense timeseries plot
# ---------------------------------------------------------------------------

def plot_timeseries(all_data: dict, out_dir: Path, font_scale: float = 1.4):
    s = font_scale
    n = len(all_data)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for i, (label, (epochs, scores, accs)) in enumerate(all_data.items()):
        cut = find_convergence_end(accs)
        epochs, scores, accs = epochs[:cut], scores[:cut], accs[:cut]

        ax = axes_flat[i]
        color_score = "tab:blue"
        color_acc   = "tab:orange"

        l1, = ax.plot(epochs, scores, color=color_score, linewidth=1.6,
                      marker="o", markersize=2.0, label="Matched Filter Score")
        ax.set_xlabel("Round", fontsize=13 * s)
        ax.set_ylabel("Matched Filter Score", color="black", fontsize=13 * s)
        ax.tick_params(axis="both", labelsize=11 * s)
        ax.tick_params(axis="y", colors="black")
        ax.spines["top"].set_visible(False)
        ax.set_title(_defense_from_label(label), fontsize=14 * s)

        score_min, score_max = scores.min(), scores.max()
        score_range = score_max - score_min or 1.0
        ax.set_ylim(score_min - 0.05 * score_range,
                    score_max + 0.50 * score_range)

        ax2 = ax.twinx()
        l2, = ax2.plot(epochs, accs, color=color_acc, linewidth=1.6,
                       linestyle="--", marker="s", markersize=2.0, label="Test Accuracy")
        ax2.set_ylabel("Test Accuracy", color="black", fontsize=13 * s)
        ax2.tick_params(axis="y", labelsize=11 * s, colors="black")
        ax2.set_ylim(0, 1.30)
        ax2.spines["top"].set_visible(False)

        ax.legend(handles=[l1, l2], loc="upper right",
                  fontsize=10 * s, handlelength=1.5, borderpad=0.4)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    out = out_dir / "matched_filter_timeseries.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Combined figure  (all defenses in one grid)
# ---------------------------------------------------------------------------

def plot_combined(all_data: dict, out_dir: Path, font_scale: float = 1.4):
    """2-row grid; each cell shows one defense (filter score + test acc)."""
    s = font_scale
    n = len(all_data)
    if n == 0:
        return
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.0 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    legend_handles = None
    for i, (label, (epochs, scores, accs)) in enumerate(all_data.items()):
        cut = find_convergence_end(accs)
        epochs, scores, accs = epochs[:cut], scores[:cut], accs[:cut]

        ax = axes_flat[i]
        color_score = "tab:blue"
        color_acc   = "tab:orange"

        l1, = ax.plot(epochs, scores, color=color_score, linewidth=1.6,
                      marker="o", markersize=2.0, label="MFS")
        ax.set_xlabel("Round", fontsize=13 * s)
        ax.set_ylabel("MFS", color="black", fontsize=13 * s)
        ax.tick_params(axis="both", labelsize=11 * s)
        ax.tick_params(axis="y", colors="black")
        ax.spines["top"].set_visible(False)
        ax.set_title(_defense_from_label(label), fontsize=14 * s)

        score_min, score_max = scores.min(), scores.max()
        score_range = score_max - score_min or 1.0
        ax.set_ylim(score_min - 0.05 * score_range,
                    score_max + 0.15 * score_range)

        ax2 = ax.twinx()
        l2, = ax2.plot(epochs, accs, color=color_acc, linewidth=1.6,
                       linestyle="--", marker="s", markersize=2.0, label="Test Accuracy")
        ax2.set_ylabel("Test Accuracy", color="black", fontsize=13 * s)
        ax2.tick_params(axis="y", labelsize=11 * s, colors="black")
        ax2.set_ylim(0, 1.05)
        ax2.spines["top"].set_visible(False)

        if legend_handles is None:
            legend_handles = [l1, l2]

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout(w_pad=4.0, rect=(0, 0, 1, 0.94))
    fig.legend(handles=legend_handles, labels=["MFS", "Test Accuracy"],
               loc="upper center", ncol=2, fontsize=13 * s,
               handlelength=1.8, columnspacing=1.5,
               bbox_to_anchor=(0.5, 1.0), frameon=True)

    out = out_dir / "combined_matched_filter.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved → {out}  (combined, all defenses)")


# ---------------------------------------------------------------------------
# Core workflow functions
# ---------------------------------------------------------------------------

def run_and_plot(argv):
    """Single train via main.py, then plot."""
    csv_path = _infer_csv_path(argv)
    print(f"[plot_mached_filter] CSV will be: {csv_path}")
    cmd = [sys.executable, "main.py"] + argv
    print(f"[plot_mached_filter] Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[plot_mached_filter] main.py exited with code {result.returncode}. "
              "Plotting available data …")
    if not csv_path.exists():
        print(f"[plot_mached_filter] ERROR: CSV not found at {csv_path}")
        sys.exit(1)
    plot_from_csvs([str(csv_path)])


def run_batch_and_plot(argv):
    """Batch train via batchrun.py, then per-defense plots + combined figure."""
    train_argv = [a for a in argv if a not in ("--batch", "-batch")]

    # Derive all expected CSV paths before training starts
    csv_paths = _infer_csv_paths_batch(train_argv)

    # Run batchrun.py
    cmd = [sys.executable, "batchrun.py"] + train_argv
    print(f"[plot_mached_filter] Running batch: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=False)

    # Collect CSVs that were actually produced
    found = [p for p in csv_paths if p.exists()]
    missing = [p for p in csv_paths if not p.exists()]
    if missing:
        print(f"\n[plot_mached_filter] WARNING: {len(missing)} CSV(s) not found:")
        for p in missing:
            print(f"  {p}")
    if not found:
        print("[plot_mached_filter] No CSV files found. Exiting.")
        sys.exit(1)

    # Per-defense plots
    print(f"\n[plot_mached_filter] Plotting {len(found)} defense(s) …")
    for csv_path in found:
        label = csv_path.stem.replace(".compass_metrics", "")
        epochs, scores, accs = load_csv(csv_path)
        out_dir = _per_defense_dir(csv_path)
        plot_timeseries({label: (epochs, scores, accs)}, out_dir)

    # Combined figure — save next to the CSVs (distribution-level directory)
    combined_dir = found[0].parent
    all_data = {}
    for csv_path in found:
        label = csv_path.stem.replace(".compass_metrics", "")
        all_data[label] = load_csv(csv_path)
    plot_combined(all_data, combined_dir)

    print(f"\n[plot_mached_filter] Done.")


def plot_from_csvs(csv_paths):
    """Plot only: per-defense timeseries for each CSV."""
    all_data = {}
    for p in csv_paths:
        label = Path(p).stem.replace(".compass_metrics", "")
        epochs, scores, accs = load_csv(p)
        all_data[label] = (epochs, scores, accs)
        r, pval = stats.pearsonr(scores, accs)
        print(f"\n[{label}]")
        print(f"  rounds recorded : {len(epochs)}")
        print(f"  filter score    : {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"  test acc        : {accs.mean():.3f} ± {accs.std():.3f}")
        print(f"  Pearson r       : {r:.4f}  (p={pval:.4g})")

        out_dir = _per_defense_dir(Path(p))
        plot_timeseries({label: (epochs, scores, accs)}, out_dir)

    if len(all_data) > 1:
        combined_dir = Path(csv_paths[0]).parent
        plot_combined(all_data, combined_dir)
        print(f"\nCombined figure saved to {combined_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        sys.exit(0)

    if _is_batch_mode(argv):
        run_batch_and_plot(argv)
    elif _is_csv_mode(argv):
        plot_from_csvs(argv)
    else:
        run_and_plot(argv)


if __name__ == "__main__":
    main()
