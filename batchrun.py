"""
批量调度脚本，用于组合不同数据集、算法、攻击与防御配置并并行启动联邦学习实验。

主要流程:
    1. 根据命令行参数枚举实验组合并生成命令。
    2. 为每个组合构建日志路径，若已存在对应日志则跳过。
    3. 使用多进程并发运行实验，并将异常输出单独记录。
"""

import argparse
import subprocess
import os
from multiprocessing import Pool
from functools import partial
import sys
from global_args import read_yaml


def run_command(command, file_name):
    """
    执行单个实验命令，并将标准输出写入日志、异常写入独立文件。

    参数:
        command (str): 待执行的 shell 命令。
        file_name (str): 对应的标准输出日志文件路径。
    返回:
        None: 结果通过文件与终端日志体现。
    异常:
        subprocess.CalledProcessError: 若需要对失败命令额外处理可在调用端捕获。
    复杂度:
        时间复杂度取决于命令本身；函数开销为 O(1)。

    费曼学习法:
        (A) 该函数检查日志是否已存在，若未运行则执行命令并保存输出。
        (B) 想象在实验室里操作仪器：先看记录本是否已有结果，没有才启动实验并把读数记录下来。
        (C) 步骤拆解:
            - 判断 `file_name` 是否已存在，存在则直接跳过避免重复实验。
            - 打印命令信息并确保日志目录存在，为输出做好准备。
            - 通过 `subprocess.Popen` 运行命令并获取标准输出/错误。
            - 若命令成功完成，打印成功信息；否则生成错误日志并写入 stdout/stderr，方便排查。
        (D) 示例:
            >>> run_command("echo hello", "logs/demo.txt")
        (E) 边界与测试:
            - 若命令需要交互输入将导致阻塞，应在测试时改用纯输出命令。
            - 建议测试: 1) 使用成功命令验证日志行为; 2) 使用故意失败的命令检查错误日志写法。
        (F) 背景参考:
            - 概念: Shell 命令调度与日志管理。
            - 参考书籍: 《Python Cookbook》中关于 subprocess 的章节。
    """
    # 若日志已存在，视为实验完成，直接跳过重复运行
    if os.path.exists(f"{file_name}"):
        print(f"File {file_name} exists, skip")
        return

    print(f"Running command: {command}")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # 将 logs 路径替换为 err_logs，用于存储错误输出
    tmp = file_name.replace("logs", "err_logs")
    out_error_file = f"{tmp[:-4]}.err"

    process = subprocess.Popen(
        command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pid = process.pid
    print(f"Started command with PID: {pid}")

    stdout, stderr = process.communicate()
    if process.returncode == 0:
        print(f"Command {command} finished successfully with PID: {pid}")
    else:
        print(f"Command {command} failed with PID: {pid}")
        print(f"Error: {stderr}")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(out_error_file, 'w') as out_error_log:
            # 同时记录标准输出与错误输出，便于复现与诊断
            out_error_log.write(stdout)
            out_error_log.write(stderr)


def get_configs(dataset, algorithm, distribution, defense):
    """
    根据数据集与算法选择实验配置，包括客户端数量、训练轮次与学习率。

    参数:
        dataset (str): 数据集名称，例如 MNIST、CIFAR10。
        algorithm (str): 联邦优化算法，如 FedSGD、FedOpt。
        distribution (str): 数据分布策略，常见有 iid、non-iid 等。
        defense (str): 防御名称，用于在特定场景下覆盖默认学习率。
    返回:
        Tuple[int, int, float]: (客户端数量, 训练轮次 epoch, 学习率 lr)。
    异常:
        ValueError: 当不存在对应配置组合时抛出。
    复杂度:
        查询操作为 O(1)。

    费曼学习法:
        (A) 函数依据预设表查出指定组合的训练参数。
        (B) 类比查阅训练手册：不同队伍（数据集）和战术（算法）对应不同练习计划。
        (C) 步骤拆解:
            - 从字典中取出指定数据集对应的算法配置。
            - 若成功获取，读取默认 epoch 与学习率。
            - 若分布为 non-iid，检查是否存在针对某些防御的额外学习率覆盖。
            - 返回客户端数量、训练轮次与学习率。
        (D) 示例:
            >>> get_configs("MNIST", "FedSGD", "iid", "Mean")
            (50, 300, 0.01)
        (E) 边界与测试:
            - 若数据集或算法名称拼写错误将触发 ValueError，应在单元测试覆盖。
            - 建议测试: 1) 逐个常见组合检查返回值; 2) 模拟 non-iid 覆盖场景确保学习率调整生效。
        (F) 背景参考:
            - 概念: 联邦学习超参数调度。
            - 参考书籍: 《Federated Learning》实验设置章节。
    """
    params = {
        "MNIST": {
            "FedSGD": {"epoch": 4000, "lr": 0.01},
            "FedOpt": {"epoch": 100, "lr": 0.01}
        },
        "CIFAR10": {
            "FedSGD": {
                "epoch": 300, "lr": 0.05,
                "non-iid": {
                    "defenses": ["Krum", "MultiKrum", "Bucketing", "Bulyan", "SignGuard", "DnC", "FLAME"],
                    "lr": 0.002
                }
            },
            "FedOpt": {
                "epoch": 300, "lr": 0.02,
                "non-iid": {
                    "defenses": ["Krum", "Bucketing"],
                    "lr": 0.002
                }
            }
        },
        "TinyImageNet": {
            "FedSGD": {"epoch": 150, "lr": 0.05}
        },
        "CHMNIST": {
            "FedSGD": {"epoch": 150, "lr": 0.001}
        },
    }

    dataset_params = params.get(dataset, {})
    num_clients = 20 if dataset == "CIFAR10" else 50
    algo_params = dataset_params.get(algorithm, {})

    if isinstance(algo_params, dict):
        epoch = algo_params["epoch"]
        lr = algo_params["lr"]

        # 针对 non-iid 场景的学习率覆盖
        if distribution == "non-iid" and "non-iid" in algo_params:
            non_iid_params = algo_params["non-iid"]
            if defense in non_iid_params.get("defenses", []):
                lr = non_iid_params.get("lr", lr)

        return num_clients, epoch, lr

    raise ValueError(f"Invalid configuration for {dataset} with {algorithm}")


def main(args):
    """
    解析命令行参数，生成所有实验组合并并行执行。

    参数:
        args (argparse.Namespace): 命令行解析结果，包含算法、攻击、防御等配置。
    返回:
        None: 实验执行过程通过日志体现。
    异常:
        SystemExit: 若找不到项目目录则退出。
    复杂度:
        构建任务列表为 O(|组合数|)，并行执行时间取决于每个命令耗时。

    费曼学习法:
        (A) 函数枚举所有参数组合并调用 `run_command` 并发执行。
        (B) 像实验室经理按清单安排所有实验，并把任务分配给多名助理。
        (C) 步骤拆解:
            - 读取外部传入的参数（数据集、模型、攻击、防御等）。
            - 确认当前路径是否位于项目目录或其父目录，确保日志路径有效。
            - 使用多进程池，根据组合生成命令与日志文件名，并收集到任务列表。
            - 通过 `pool.starmap` 并行执行所有任务，最后关闭进程池。
        (D) 示例:
            >>> ns = argparse.Namespace(dataset="MNIST", model="lenet", attacks=["NoAttack"], defenses=["Mean"], distributions=["iid"], algorithms=["FedSGD"], gpu_idx=0, max_processes=2)
            >>> main(ns)
        (E) 边界与测试:
            - 若组合数量极大，任务构建可能消耗较多内存，可在测试中限制列表规模。
            - 建议测试: 1) 使用少量组合运行 smoke test; 2) 模拟日志已存在的情况验证跳过逻辑。
        (F) 背景参考:
            - 概念: 并行批量实验调度。
            - 参考书籍: 《High Performance Python》关于多进程章节。
    """
    distributions = ['iid', 'non-iid', 'class-imbalanced_iid']
    algorithms = ['FedSGD', 'FedAvg', 'FedOpt']
    folder_name = 'FLPoison'
    gpu_idx = 1
    MAX_PROCESSES = 6

    # 覆盖默认值，使用命令行传入的参数
    dataset = args.dataset
    model = args.model
    attacks = args.attacks
    defenses = args.defenses
    distributions = args.distributions
    algorithms = args.algorithms
    gpu_idx = args.gpu_idx
    MAX_PROCESSES = args.max_processes
    datasets_models = [(dataset, model)]

    # 检查当前工作目录是否位于或包含项目文件夹
    current_dir = os.getcwd()
    if folder_name in current_dir:
        dir = current_dir
    elif os.path.isdir(os.path.join(current_dir, folder_name)):
        dir = os.path.join(current_dir, folder_name)
    else:
        print(
            f"Error: The current directory '{current_dir}' is not in {folder_name} and does not contain an {folder_name} folder.")
        sys.exit(1)

    # 创建进程池，准备并发执行任务
    pool = Pool(processes=MAX_PROCESSES)
    tasks = []
    for algorithm in algorithms:
        for dataset, model in datasets_models:
            config_file = f"{algorithm}_{dataset}_config.yaml"
            for distribution in distributions:
                for attack in attacks:
                    for defense in defenses:
                        num_clients, epoch, learning_rate = get_configs(
                            dataset, algorithm, distribution, defense)

                        command = f'python -u main.py -config=./configs/{config_file} -data {dataset} -model {model} -e {epoch} -att {attack} -def {defense} -dtb {distribution} -alg {algorithm} -lr {learning_rate} -gidx {gpu_idx}'
                        file_name = f'{dir}/logs/{algorithm}/{dataset}_{model}/{distribution}/{dataset}_{model}_{distribution}_{attack}_{defense}_{epoch}_{num_clients}_{learning_rate}_{algorithm}.txt'

                        # 收集任务，稍后统一并行运行
                        tasks.append((command, file_name))

    # 并行执行所有命令
    pool.starmap(partial(run_command), tasks)

    # 关闭并等待进程池结束
    pool.close()
    pool.join()


def get_all_attacks_defenses():
    """
    从默认配置文件中提取所有攻击与防御名称，便于快速查询。

    返回:
        Tuple[List[str], List[str]]: (攻击列表, 防御列表)。
    异常:
        FileNotFoundError: 当配置文件不存在时由 read_yaml 抛出。
    复杂度:
        解析 YAML 的时间复杂度随配置大小线性增长。

    费曼学习法:
        (A) 函数读取配置文件列出所有攻击与防御选项。
        (B) 类比翻阅实验目录，找到目前可用的招式清单。
        (C) 步骤拆解:
            - 调用 `read_yaml` 解析默认配置。
            - 遍历配置中的 attacks 与 defenses 项提取名称。
            - 返回两个列表，供调用方使用。
        (D) 示例:
            >>> attacks, defenses = get_all_attacks_defenses()
        (E) 边界与测试:
            - 若配置格式变动（字段名称改变），解析会失败，需要在测试中监控。
            - 建议测试: 1) 使用真实配置文件执行一次; 2) 构造最小 YAML 文件验证字段解析。
        (F) 背景参考:
            - 概念: YAML 配置读取。
            - 参考书籍: 《Python YAML Cookbook》。
    """
    args = vars(read_yaml('./configs/FedSGD_MNIST_config.yaml'))
    attacks = [attack_i['attack'] for attack_i in args['attacks']]
    defenses = [defense_j['defense'] for defense_j in args['defenses']]
    return attacks, defenses


def test():
    """
    打印默认配置中的攻击与防御列表，便于快速检查解析结果。

    返回:
        None: 结果直接输出到终端。
    复杂度:
        取决于 `get_all_attacks_defenses` 的复杂度。

    费曼学习法:
        (A) 函数调用 `get_all_attacks_defenses` 并打印结果。
        (B) 好比实验前先读出清单，确认所有工具齐全。
        (C) 步骤拆解:
            - 调用辅助函数获取攻击与防御名称。
            - 以人类可读方式打印到屏幕。
        (D) 示例:
            >>> test()
        (E) 边界与测试:
            - 若配置文件缺失，此函数将抛出异常，需在测试中准备样例配置。
            - 建议测试: 1) 使用真实环境执行; 2) 捕获异常检查错误提示。
        (F) 背景参考:
            - 概念: 调试辅助函数设计。
    """
    attacks, defenses = get_all_attacks_defenses()
    print(f"attacks = {attacks}", end="\n\n")
    print(f"defenses = {defenses}")


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Run distributed training with attacks and defenses.")

    # Adding arguments for distributions, algorithms, and gpu_idx
    parser.add_argument('-distributions', '--distributions', nargs='+', default=['iid', 'non-iid'],
                        help="List of distributions to use. Default is ['iid'].")
    parser.add_argument('-algorithms', '--algorithms', nargs='+', default=[
                        'FedSGD', 'FedOpt'], help="List of algorithm types to use. Default is ['FedSGD'].")
    # data
    parser.add_argument('-data', '--dataset', type=str, default='MNIST',
                        help="Dataset to use. Default is MNIST.")
    parser.add_argument('-model', '--model', type=str, default='lenet',
                        help="Model to use. Default is lenet.")

    parser.add_argument('-gidx', '--gpu_idx', type=int, default=1,
                        help="GPU index to use. Default is 1.")
    parser.add_argument('-maxp', '--max_processes', type=int, default=6,
                        help="Max number of process parallel. Default is 6.")
    # attacks
    parser.add_argument('-attacks', '--attacks', nargs='+', default=['NoAttack', 'Gaussian', 'SignFlipping', 'IPM', 'ALIE', 'FangAttack', 'MinMax',
                        'MinSum',  'Mimic', 'LabelFlipping', 'BadNets', 'ModelReplacement', 'DBA', 'AlterMin', 'EdgeCase', 'Neurotoxin'], help="List of attacks to use.")
    parser.add_argument('-defenses', '--defenses', nargs='+', default=['Mean', 'Krum', 'MultiKrum', 'TrimmedMean', 'Median', 'Bulyan', 'RFA', 'FLTrust',
                        'CenteredClipping', 'DnC', 'Bucketing', 'SignGuard', 'Auror', 'FoolsGold', 'NormClipping', 'CRFL', 'DeepSight', 'FLAME'], help="List of defenses to use.")
    # Parse the arguments
    args = parser.parse_args()
    # Call the main function with parsed args
    main(args)


# __AI_ANNOTATION_SUMMARY__
# - run_command: 检查日志并执行实验命令，记录成功或错误输出。
# - get_configs: 根据数据集与算法组合查表返回客户端数、轮次与学习率。
# - main: 枚举所有参数组合并使用多进程并发运行实验。
# - get_all_attacks_defenses: 从默认配置文件中提取攻击与防御名称列表。
# - test: 调试辅助函数，打印所有攻击与防御配置。
