"""
批量调度 main_hyper.py 的脚本。

主要流程与 batchrun.py 保持一致:
    1. 根据命令行参数枚举实验组合并生成命令。
    2. 为每个组合构建超参搜索的基础输出路径，并检查是否已有搜索产物。
    3. 使用多进程并发运行 main_hyper.py，将异常输出单独记录。
"""

import argparse
import os
import shlex
import subprocess
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path


SUPPORTED_HYPER_ATTACKS = ["COMPASS"]


def should_skip_hyper_run(base_output: str, defense: str) -> bool:
    """
    判断某个超参搜索组合是否已经跑过。

    main_hyper.py 会把基础输出路径自动移动到 defense 子目录下，并生成:
        - __hypersearch_index__.jsonl
        - __hypersearch_summary__*.txt
        - __hyper_t*.txt
    这些文件任意一个存在，都说明该组合已经启动过搜索，批量脚本直接跳过。
    """
    base_path = Path(base_output)
    final_base = base_path.parent / defense / base_path.name
    search_index = final_base.with_name(f"{final_base.stem}__hypersearch_index__.jsonl")

    if search_index.exists():
        return True
    if list(final_base.parent.glob(f"{final_base.stem}__hypersearch_summary__*.txt")):
        return True
    if list(final_base.parent.glob(f"{final_base.stem}__hyper_t*.txt")):
        return True
    return False


def run_command(command: str, base_output: str, defense: str) -> None:
    """
    执行单个超参搜索命令，并将异常输出写入 err_logs。
    """
    if should_skip_hyper_run(base_output, defense):
        print(f"Hyper-search outputs for {base_output} ({defense}) already exist, skip")
        return

    print(f"Running command: {command}")
    os.makedirs(os.path.dirname(base_output), exist_ok=True)

    error_anchor = str(Path(base_output).parent / defense / Path(base_output).name)
    tmp = error_anchor.replace("logs", "err_logs")
    out_error_file = f"{tmp[:-4]}.err"

    process = subprocess.Popen(
        command,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pid = process.pid
    print(f"Started command with PID: {pid}")

    stdout, stderr = process.communicate()
    if process.returncode == 0:
        print(f"Command {command} finished successfully with PID: {pid}")
        return

    print(f"Command {command} failed with PID: {pid}")
    print(f"Error: {stderr}")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(out_error_file, "w", encoding="utf-8") as out_error_log:
        out_error_log.write(stdout)
        out_error_log.write(stderr)


def get_configs(dataset: str, algorithm: str, distribution: str, defense: str):
    """
    复用 batchrun.py 的实验配置选择逻辑。
    """
    params = {
        "MNIST": {
            "FedSGD": {"epoch": 200, "lr": 0.05},
            "FedOpt": {"epoch": 100, "lr": 0.01},
        },
        "FashionMNIST": {
            "FedSGD": {"epoch": 1000, "lr": 0.05},
            "FedOpt": {"epoch": 100, "lr": 0.01},
        },
        "CIFAR10": {
            "FedSGD": {
                "epoch": 1000,
                "lr": 0.05,
                "non-iid": {
                    "defenses": ["Krum", "MultiKrum", "Bucketing", "Bulyan", "SignGuard", "DnC", "FLAME"],
                    "lr": 0.002,
                },
            },
            "FedOpt": {
                "epoch": 200,
                "lr": 0.02,
                "non-iid": {
                    "defenses": ["Krum", "Bucketing"],
                    "lr": 0.002,
                },
            },
        },
        "CINIC10": {
            "FedSGD": {"epoch": 200, "lr": 0.05},
        },
        "CIFAR100": {
            "FedSGD": {
                "epoch": 1000,
                "lr": 0.05,
                "non-iid": {
                    "defenses": ["Krum", "MultiKrum", "Bucketing", "Bulyan", "SignGuard", "DnC", "FLAME"],
                    "lr": 0.002,
                },
            },
        },
        "CIFAR20": {
            "FedSGD": {
                "epoch": 1000,
                "lr": 0.05,
                "non-iid": {
                    "defenses": ["Krum", "MultiKrum", "Bucketing", "Bulyan", "SignGuard", "DnC", "FLAME"],
                    "lr": 0.002,
                },
            },
        },
        "CIFAR50": {
            "FedSGD": {
                "epoch": 1000,
                "lr": 0.05,
                "non-iid": {
                    "defenses": ["Krum", "MultiKrum", "Bucketing", "Bulyan", "SignGuard", "DnC", "FLAME"],
                    "lr": 0.002,
                },
            },
        },
        "TinyImageNet": {
            "FedSGD": {"epoch": 150, "lr": 0.05},
        },
        "CHMNIST": {
            "FedSGD": {"epoch": 400, "lr": 0.05},
        },
        "5GNIDD": {
            "FedSGD": {"epoch": 500, "lr": 0.05},
        },
    }

    dataset_params = params.get(dataset, {})
    if dataset in ["CIFAR10", "CIFAR100", "CIFAR20", "CIFAR50"]:
        num_clients = 20
    elif dataset == "5GNIDD":
        num_clients = 100
    else:
        num_clients = 50
    algo_params = dataset_params.get(algorithm, {})

    if isinstance(algo_params, dict):
        epoch = algo_params["epoch"]
        lr = algo_params["lr"]

        if distribution == "non-iid" and "non-iid" in algo_params:
            non_iid_params = algo_params["non-iid"]
            if defense in non_iid_params.get("defenses", []):
                lr = non_iid_params.get("lr", lr)

        return num_clients, epoch, lr

    raise ValueError(f"Invalid configuration for {dataset} with {algorithm}")


def get_all_attacks_defenses():
    """
    从默认配置文件中提取攻击与防御名称。

    为避免批量脚本调度当前尚未实现超参搜索插件的攻击，这里会自动过滤到
    SUPPORTED_HYPER_ATTACKS 范围内。
    """
    try:
        from global_args import read_yaml

        args = vars(read_yaml("./configs/FedSGD_MNIST_config.yaml"))
        attacks = [
            attack_item["attack"]
            for attack_item in args["attacks"]
            if attack_item["attack"] in SUPPORTED_HYPER_ATTACKS
        ]
        defenses = [defense_item["defense"] for defense_item in args["defenses"]]
        return attacks, defenses
    except Exception as exc:
        print(f"Warning: failed to load default attacks/defenses from config: {exc}")
        return SUPPORTED_HYPER_ATTACKS, [
            "Mean",
            "Krum",
            "MultiKrum",
            "TrimmedMean",
            "Median",
            "Bulyan",
            "RFA",
            "FLTrust",
            "CenteredClipping",
            "DnC",
            "Bucketing",
            "SignGuard",
            "Auror",
            "FoolsGold",
            "NormClipping",
            "CRFL",
            "DeepSight",
            "FLAME",
        ]


def build_command(
    config_file: str,
    dataset: str,
    model: str,
    epoch: int,
    attack: str,
    defense: str,
    distribution: str,
    algorithm: str,
    learning_rate: float,
    gpu_idx: int,
    base_output: str,
    args: argparse.Namespace,
) -> str:
    """
    构造单个 main_hyper.py 命令。
    """
    command_parts = [
        "python",
        "-u",
        "main_hyper.py",
        f"-config=./configs/{config_file}",
        "-data",
        dataset,
        "-model",
        model,
        "-e",
        str(epoch),
        "-att",
        attack,
        "-def",
        defense,
        "-dtb",
        distribution,
        "-alg",
        algorithm,
        "-lr",
        str(learning_rate),
        "-gidx",
        str(gpu_idx),
        "-o",
        base_output,
        "--max_search_trials",
        str(args.max_search_trials),
        "--search_patience",
        str(args.search_patience),
        "--success_acc_threshold",
        str(args.success_acc_threshold),
        "--success_consecutive_rounds",
        str(args.success_consecutive_rounds),
        "--success_eval_start_round",
        str(args.success_eval_start_round),
        "--search_step_decay",
        str(args.search_step_decay),
    ]

    if args.attack_params is not None:
        command_parts.extend(["--attack_params", args.attack_params])
    if args.defense_params is not None:
        command_parts.extend(["--defense_params", args.defense_params])

    return " ".join(shlex.quote(part) for part in command_parts)


def main(args: argparse.Namespace) -> None:
    distributions = args.distributions
    algorithms = args.algorithms
    attacks = args.attacks
    defenses = args.defenses
    dataset = args.dataset
    model = args.model
    gpu_idx = args.gpu_idx
    max_processes = args.max_processes
    datasets_models = [(dataset, model)]
    folder_name = "FLPoison"

    current_dir = os.getcwd()
    if folder_name in current_dir:
        workdir = current_dir
    elif os.path.isdir(os.path.join(current_dir, folder_name)):
        workdir = os.path.join(current_dir, folder_name)
    else:
        print(
            f"Error: The current directory '{current_dir}' is not in {folder_name} and does not contain an {folder_name} folder."
        )
        sys.exit(1)

    pool = Pool(processes=max_processes)
    tasks = []
    for algorithm in algorithms:
        for dataset_name, model_name in datasets_models:
            config_file = f"{algorithm}_{dataset_name}_config.yaml"
            for distribution in distributions:
                for attack in attacks:
                    if attack not in SUPPORTED_HYPER_ATTACKS:
                        print(f"Attack {attack} is not supported by main_hyper.py yet, skip")
                        continue
                    for defense in defenses:
                        num_clients, epoch, learning_rate = get_configs(
                            dataset_name, algorithm, distribution, defense
                        )
                        base_output = (
                            f"{workdir}/logs/{algorithm}/{dataset_name}_{model_name}/{distribution}/"
                            f"{dataset_name}_{model_name}_{distribution}_{attack}_{defense}_"
                            f"{epoch}_{num_clients}_{learning_rate}_{algorithm}.txt"
                        )
                        command = build_command(
                            config_file=config_file,
                            dataset=dataset_name,
                            model=model_name,
                            epoch=epoch,
                            attack=attack,
                            defense=defense,
                            distribution=distribution,
                            algorithm=algorithm,
                            learning_rate=learning_rate,
                            gpu_idx=gpu_idx,
                            base_output=base_output,
                            args=args,
                        )
                        tasks.append((command, base_output, defense))

    pool.starmap(partial(run_command), tasks)
    pool.close()
    pool.join()


if __name__ == "__main__":
    default_attacks, default_defenses = get_all_attacks_defenses()

    parser = argparse.ArgumentParser(
        description="Batch runner for main_hyper.py hyper-parameter searches."
    )
    parser.add_argument(
        "-distributions",
        "--distributions",
        nargs="+",
        default=["iid", "non-iid"],
        help="List of distributions to use.",
    )
    parser.add_argument(
        "-algorithms",
        "--algorithms",
        nargs="+",
        default=["FedSGD", "FedOpt"],
        help="List of algorithm types to use.",
    )
    parser.add_argument(
        "-data",
        "--dataset",
        type=str,
        default="MNIST",
        help="Dataset to use.",
    )
    parser.add_argument(
        "-model",
        "--model",
        type=str,
        default="simplecnn",
        help="Model to use.",
    )
    parser.add_argument(
        "-gidx",
        "--gpu_idx",
        type=int,
        default=0,
        help="GPU index to use.",
    )
    parser.add_argument(
        "-maxp",
        "--max_processes",
        type=int,
        default=4,
        help="Max number of parallel processes.",
    )
    parser.add_argument(
        "-attacks",
        "--attacks",
        nargs="+",
        default=default_attacks or SUPPORTED_HYPER_ATTACKS,
        help="List of attacks to use for hyper-search.",
    )
    parser.add_argument(
        "-defenses",
        "--defenses",
        nargs="+",
        default=default_defenses,
        help="List of defenses to use.",
    )
    parser.add_argument(
        "--max_search_trials",
        type=int,
        default=0,
        help="Maximum number of hyper-search trials; <= 0 means searching until success.",
    )
    parser.add_argument(
        "--search_patience",
        type=int,
        default=2,
        help="How many local-search rounds without improvement before shrinking step size.",
    )
    parser.add_argument(
        "--success_acc_threshold",
        type=float,
        default=0.18,
        help="Attack success threshold on test accuracy.",
    )
    parser.add_argument(
        "--success_consecutive_rounds",
        type=int,
        default=20,
        help="When all rounds finish, require the final N rounds to stay below threshold.",
    )
    parser.add_argument(
        "--success_eval_start_round",
        type=int,
        default=50,
        help="Deprecated compatibility option passed through to main_hyper.py.",
    )
    parser.add_argument(
        "--search_step_decay",
        type=float,
        default=0.5,
        help="Step decay factor after patience is exhausted.",
    )
    parser.add_argument(
        "--attack_params",
        type=str,
        default=None,
        help="Optional attack_params override string passed to main_hyper.py.",
    )
    parser.add_argument(
        "--defense_params",
        type=str,
        default=None,
        help="Optional defense_params override string passed to main_hyper.py.",
    )

    main(parser.parse_args())
