"""One-command runner for the MyTest direction-similarity experiment.

This orchestrator runs the MyTest attacker across one or more datasets,
optionally in parallel (one subprocess per dataset), then automatically
renders the cross-dataset comparison figures.

Examples:
    # Sequential (original behaviour):
    python run_mytest.py -datasets MNIST FashionMNIST -alg FedSGD \
        --epochs 30 --early_rounds 10 --final_window 5 -gidx 0

    # Parallel — 6 datasets, 3 GPUs, 2 datasets per GPU:
    python run_mytest.py -datasets MNIST FashionMNIST CHMNIST CIFAR10 CINIC10 CIFAR100 \
        -alg FedSGD -e 500 --early_rounds 10 --final_window 5 -gidx 0 1 2

    # Parallel — 6 datasets on a single GPU (memory-permitting):
    python run_mytest.py -datasets MNIST FashionMNIST CHMNIST CIFAR10 CINIC10 CIFAR100 \
        -alg FedSGD -e 500 -gidx 0 --workers 6

    # Explicit config files instead of -datasets/-alg:
    python run_mytest.py -configs configs/FedSGD_MNIST_config.yaml \
        configs/FedSGD_FashionMNIST_config.yaml --early_rounds 10
"""

import argparse
import concurrent.futures
import importlib
import logging
import multiprocessing
from pathlib import Path

from attackers.mytest import _artifact_path
from global_args import read_yaml, single_preprocess
from plot_mytest import (
    configure_style,
    load_artifacts,
    plot_direction_similarity,
    plot_topn_overlap,
    plot_direction_similarity_combined,
    plot_topn_overlap_combined,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MyTest across datasets and plot direction/Top-N similarity."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-configs", "--configs", nargs="+",
        help="One or more YAML config files (one run each).",
    )
    source.add_argument(
        "-datasets", "--datasets", nargs="+",
        help="Dataset names; combined with -alg to build configs/{alg}_{dataset}_config.yaml.",
    )
    parser.add_argument("-alg", "--algorithm", default="FedSGD",
                        help="Algorithm prefix for -datasets config lookup.")

    # Shared MyTest / training parameters.
    parser.add_argument("--early_rounds", type=int, default=10)
    parser.add_argument("--final_window", type=int, default=5)
    parser.add_argument("--top_ratios", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument("--dense_rounds", type=int, default=50,
                        help="Record every round up to this round (default 50).")
    parser.add_argument("--record_stride", type=int, default=10,
                        help="After dense_rounds, record every Nth round (default 10).")
    parser.add_argument("-e", "--epochs", type=int, default=None)
    parser.add_argument("-seed", "--seed", type=int, default=None)
    parser.add_argument("-dtb", "--distribution", default=None)
    parser.add_argument("-lr", "--learning_rate", type=float, default=None)
    parser.add_argument("-num_clients", "--num_clients", type=int, default=None)
    parser.add_argument("-bs", "--batch_size", type=int, default=None)
    parser.add_argument("-def", "--defense", default="Mean")
    parser.add_argument("-num_adv", "--num_adv", type=float, default=1)
    parser.add_argument("-gidx", "--gpu_idx", type=int, nargs="+", default=[0])

    parser.add_argument(
        "--workers", type=int, default=None,
        help=(
            "Number of datasets to run in parallel. "
            "Defaults to len(gpu_idx) (one worker per GPU). "
            "Set to 1 to disable parallelism."
        ),
    )
    parser.add_argument("--output-dir", default="results/figures",
                        help="Where to save the comparison figures.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--font-scale", type=float, default=1.8,
        help="Multiply all font sizes by this factor (default 2.0). "
             "E.g. 2.0 makes every label/tick/title 100%% larger — "
             "recommended for paper figures that are scaled down on the page.",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if a dataset's artifact already exists.")
    parser.add_argument(
        "--artifacts-dir", nargs="*", default=[],
        metavar="DIR",
        help="Extra directories to search for existing *.mytest.pt artifacts "
             "(current directory is always searched automatically).",
    )
    return parser.parse_args()


def resolve_configs(args):
    if args.configs:
        return [Path(c) for c in args.configs]
    return [
        Path(f"configs/{args.algorithm}_{dataset}_config.yaml")
        for dataset in args.datasets
    ]


def cleanup_logger(logger_name):
    logger = logging.getLogger(logger_name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def build_run_args(config_path, args):
    """Load a config and force the MyTest / benign-recording setup onto it."""
    run_args = read_yaml(str(config_path))
    run_args.attack = "MyTest"
    run_args.attack_params = {
        "early_rounds": args.early_rounds,
        "final_window": args.final_window,
        "top_ratios": list(args.top_ratios),
        "top_n_list": None,
        "dense_rounds": args.dense_rounds,
        "record_stride": args.record_stride,
    }
    run_args.defense = args.defense
    run_args.defense_params = None
    run_args.num_adv = args.num_adv
    run_args.gpu_idx = list(args.gpu_idx)
    if args.epochs is not None:
        run_args.epochs = args.epochs
    if args.seed is not None:
        run_args.seed = args.seed
    if args.distribution is not None:
        run_args.distribution = args.distribution
    if args.learning_rate is not None:
        run_args.learning_rate = args.learning_rate
    if args.num_clients is not None:
        run_args.num_clients = args.num_clients
    if args.batch_size is not None:
        run_args.batch_size = args.batch_size
    single_preprocess(run_args)
    return run_args


def find_artifact(run_args, extra_dirs=()):
    """Return the artifact Path if it exists in any known location.

    Search order:
      1. The canonical path derived from run_args.output  (e.g. logs/…/*.mytest.pt)
      2. The current working directory
      3. Any extra directories supplied via --artifacts-dir

    Returns the first match, or the canonical path (even if absent) so the
    caller can create it there.
    """
    canonical = _artifact_path(run_args.output)
    if canonical.exists():
        return canonical
    filename = canonical.name
    for directory in (Path("."), *[Path(d) for d in extra_dirs]):
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return canonical  # does not exist yet; will be created here


# ---------------------------------------------------------------------------
# Subprocess worker — must be a top-level function so multiprocessing.spawn
# can pickle and import it in the child process.
# ---------------------------------------------------------------------------

def _worker_run(config_path_str: str, args_dict: dict, assigned_gpu: int) -> str:
    """Run one dataset's FL training in a fresh subprocess.

    Parameters
    ----------
    config_path_str:
        Absolute or relative path to the YAML config file.
    args_dict:
        Plain-dict copy of the argparse Namespace (all picklable primitives).
        ``gpu_idx`` is ignored — ``assigned_gpu`` takes precedence.
    assigned_gpu:
        The GPU index this worker should use.

    Returns
    -------
    str
        Absolute path of the written ``*.mytest.pt`` artifact.
    """
    import importlib
    import logging
    import argparse
    from pathlib import Path
    from global_args import read_yaml, single_preprocess
    from attackers.mytest import _artifact_path

    # Reconstruct a minimal args namespace inside the child.
    args = argparse.Namespace(**args_dict)
    args.gpu_idx = [assigned_gpu]

    # Build run_args exactly as build_run_args() does, but with the child's gpu.
    run_args = read_yaml(config_path_str)
    run_args.attack = "MyTest"
    run_args.attack_params = {
        "early_rounds": args.early_rounds,
        "final_window": args.final_window,
        "top_ratios": list(args.top_ratios),
        "top_n_list": None,
        "dense_rounds": args.dense_rounds,
        "record_stride": args.record_stride,
    }
    run_args.defense = args.defense
    run_args.defense_params = None
    run_args.num_adv = args.num_adv
    run_args.gpu_idx = [assigned_gpu]
    if args.epochs is not None:
        run_args.epochs = args.epochs
    if args.seed is not None:
        run_args.seed = args.seed
    if args.distribution is not None:
        run_args.distribution = args.distribution
    if args.learning_rate is not None:
        run_args.learning_rate = args.learning_rate
    if args.num_clients is not None:
        run_args.num_clients = args.num_clients
    if args.batch_size is not None:
        run_args.batch_size = args.batch_size
    single_preprocess(run_args)

    dataset = getattr(run_args, "dataset", config_path_str)

    experiment_main = importlib.import_module("main")
    original_plot = experiment_main.plot_accuracy
    experiment_main.plot_accuracy = lambda *_a, **_k: None

    def _cleanup_logger():
        logger = logging.getLogger("main")
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()

    _cleanup_logger()
    try:
        experiment_main.fl_run(run_args)
    finally:
        experiment_main.plot_accuracy = original_plot
        _cleanup_logger()

    canonical = _artifact_path(run_args.output)
    if not canonical.exists():
        raise RuntimeError(
            f"Run for {dataset} finished but no artifact was written at {canonical}."
        )
    return str(canonical.resolve())


# ---------------------------------------------------------------------------
# Sequential fallback (workers=1 or single dataset)
# ---------------------------------------------------------------------------

def run_single(config_path, args):
    """Run one dataset in-process and return its artifact path."""
    run_args = build_run_args(config_path, args)
    artifact = find_artifact(run_args, extra_dirs=args.artifacts_dir)

    if artifact.exists() and not args.force:
        print(f"[run_mytest] SKIP  {run_args.dataset}: artifact found at {artifact}")
        return artifact

    print(f"[run_mytest] RUN   {run_args.dataset} -> {_artifact_path(run_args.output)}")
    experiment_main = importlib.import_module("main")
    original_plot_accuracy = experiment_main.plot_accuracy
    experiment_main.plot_accuracy = lambda *_a, **_k: None
    cleanup_logger("main")
    try:
        experiment_main.fl_run(run_args)
    finally:
        experiment_main.plot_accuracy = original_plot_accuracy
        cleanup_logger("main")

    canonical = _artifact_path(run_args.output)
    if not canonical.exists():
        raise RuntimeError(
            f"Run for {run_args.dataset} finished but no artifact was written at {canonical}."
        )
    return canonical


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    configs = resolve_configs(args)

    # ── Pre-flight: show which datasets will be skipped and which will run ──
    print("[run_mytest] ── Pre-flight check ──────────────────────────────")
    to_run = []       # config paths that actually need training
    artifacts = {}    # config_path -> resolved artifact Path (skipped or done)

    for config_path in configs:
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        run_args = build_run_args(config_path, args)
        artifact = find_artifact(run_args, extra_dirs=args.artifacts_dir)
        if artifact.exists() and not args.force:
            print(f"  [SKIP (artifact exists)      ] {run_args.dataset:15s}  {artifact}")
            artifacts[config_path] = artifact
        else:
            print(f"  [RUN                         ] {run_args.dataset:15s}  {_artifact_path(run_args.output)}")
            to_run.append(config_path)
    print("[run_mytest] ─────────────────────────────────────────────────────")

    # ── Run training (parallel or sequential) ──────────────────────────────
    n_workers = args.workers if args.workers is not None else len(args.gpu_idx)
    n_workers = max(1, min(n_workers, len(to_run))) if to_run else 1

    # Assign GPUs round-robin across the datasets that need running.
    gpu_assignments = [args.gpu_idx[i % len(args.gpu_idx)] for i in range(len(to_run))]

    if not to_run:
        print("[run_mytest] All datasets already have artifacts — skipping training.")

    elif n_workers == 1:
        # Sequential path: runs in the same process (simpler, easier to debug).
        for config_path in to_run:
            artifact = run_single(config_path, args)
            artifacts[config_path] = artifact

    else:
        # Parallel path: one subprocess per dataset via multiprocessing.spawn.
        # spawn is required for CUDA — fork leaves CUDA in an undefined state.
        print(
            f"[run_mytest] Launching {len(to_run)} run(s) with "
            f"{n_workers} parallel worker(s)"
        )
        for cfg, gpu in zip(to_run, gpu_assignments):
            # Extract dataset name for display (best-effort).
            try:
                _ra = build_run_args(cfg, args)
                ds_name = getattr(_ra, "dataset", cfg.stem)
            except Exception:
                ds_name = cfg.stem
            print(f"  GPU {gpu}  <-  {ds_name}")

        # args_dict: only picklable primitives so spawn can transfer it.
        args_dict = {k: v for k, v in vars(args).items()}

        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx
        ) as executor:
            future_to_cfg = {
                executor.submit(_worker_run, str(cfg), args_dict, gpu): cfg
                for cfg, gpu in zip(to_run, gpu_assignments)
            }
            for future in concurrent.futures.as_completed(future_to_cfg):
                cfg = future_to_cfg[future]
                try:
                    result_path = future.result()
                    artifacts[cfg] = Path(result_path)
                    print(f"[run_mytest] DONE  {cfg.stem}  ->  {result_path}")
                except Exception as exc:
                    print(f"[run_mytest] FAILED {cfg}: {exc}")
                    raise

    # ── Plot ───────────────────────────────────────────────────────────────
    ordered_artifacts = [artifacts[cfg] for cfg in configs]

    # Derive log paths automatically: foo.mytest.pt -> foo.txt (if it exists).
    log_files = []
    for art in ordered_artifacts:
        art = Path(art)
        log = art.parent / (art.name[: -len(".mytest.pt")] + ".txt")
        log_files.append(str(log) if log.exists() else None)

    fs = args.font_scale
    configure_style(fs)
    datasets = load_artifacts([str(a) for a in ordered_artifacts], log_files=log_files, no_cache=False)
    output_dir = Path(args.output_dir)

    # Per-dataset figures (each saved under output_dir/<dataset>/)
    plot_direction_similarity(datasets, output_dir, args.dpi, font_scale=fs)
    plot_topn_overlap(datasets, output_dir, args.dpi, font_scale=fs)

    # Combined figures (all datasets in one figure, saved under output_dir/)
    plot_direction_similarity_combined(datasets, output_dir, args.dpi, font_scale=fs)
    plot_topn_overlap_combined(datasets, output_dir, args.dpi, font_scale=fs)

    print(
        f"[run_mytest] Per-dataset figures saved under {output_dir}/<dataset>/\n"
        f"[run_mytest] Combined figures saved under {output_dir}/\n"
        f"[run_mytest] Datasets: {sorted(datasets)}"
    )


if __name__ == "__main__":
    main()
