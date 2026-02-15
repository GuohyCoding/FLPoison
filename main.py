"""
联邦学习实验入口脚本，负责组织数据加载、客户端初始化、训练循环、攻击触发与结果记录。

核心流程:
    1. 解析命令行与配置文件参数，完成实验预处理；
    2. 构建服务器与客户端对象，执行联邦训练并同步模型；
    3. 支持 omniscient 攻击与计时统计，并记录日志/绘制曲线。
"""

import gc
import logging
import time
from pathlib import Path
import numpy as np
import torch

from fl import coordinator
from global_args import benchmark_preprocess, read_args, override_args, single_preprocess
from global_utils import avg_value, print_filtered_args, setup_logger, setup_seed
from datapreprocessor.data_utils import load_data, split_dataset
from fl.server import Server
from plot_utils import plot_accuracy


def fl_run(args):
    """
    执行联邦学习主流程，包括数据装载、训练循环与评估记录。

    参数:
        args (SimpleNamespace): 实验配置对象，包含模型、数据、攻击等设置。
    返回:
        None: 函数通过日志与文件输出结果。
    异常:
        ValueError: 当 omniscient 攻击未生成合法更新时由子函数抛出。
    复杂度:
        时间复杂度 O(epochs × num_clients × local_cost)，空间复杂度 O(|model|)。

    费曼学习法:
        (A) 函数按照既定流程完成一次联邦学习实验。
        (B) 类比一场跨校接力赛：先集合队员、训练、收集成绩，再评估与记录。
        (C) 步骤拆解:
            - 初始化日志与随机种子，确保实验可追踪且结果可重复。
            - 加载数据并根据分布策略分配给各客户端。
            - 创建客户端与服务器实例，并设置联邦优化算法。
            - 进入全局训练循环：广播模型、客户端本地训练、执行可能的全知攻击、
              服务器聚合、评估并记录指标。
            - 训练完成后输出计时信息与可视化结果。
        (D) 示例:
            >>> fl_run(args)  # args 由 preprocess 阶段准备
        (E) 边界与测试:
            - 若 `args.epochs` 为 0，将不会执行训练循环，应确认是否符合预期。
            - 建议测试: 1) 使用极小数据集运行 smoke test； 2) 模拟无攻击与有攻击场景验证日志输出。
        (F) 背景参考:
            - 概念: 联邦学习训练管线。
            - 参考书籍: 《Federated Learning》实验框架章节。
    """
    # Step 0: prepare logging and randomness so that the entire FL run is reproducible.
    # 步骤0：初始化日志与随机种子，保证实验过程、输出与随机性完全可追踪。
    args.logger = setup_logger(
        __name__, f'{args.output}', level=logging.INFO)
    print_filtered_args(args, args.logger)
    start_time = time.time()
    args.logger.info(
        f"Started on {time.asctime(time.localtime(start_time))}")
    # fix randomness
    # 设置统一随机种子；涉及 Torch/Numpy/随机扰动的步骤都会因此复现。
    setup_seed(args.seed)

    # 1. Load the raw dataset and partition it according to the specified heterogeneity pattern.
    # 第1步：加载原始数据集，并根据 iid / non-iid / pat 等分布策略切分给各客户端。
    train_dataset, test_dataset = load_data(args)
    client_indices, test_dataset = split_dataset(
        args, train_dataset, test_dataset)
    args.logger.info("Data partitioned")

    # 2. Instantiate every client/server with its own data slice so local training is independent.
    # 第2步：创建客户端与服务器对象，并为每个客户端绑定其专属的数据索引，实现本地独立训练。
    clients = coordinator.init_clients(
        args, client_indices, train_dataset, test_dataset)
    the_server = Server(args, clients, test_dataset, train_dataset)

    # 3. Configure the FL optimizer/aggregator so both sides know how to update weights.
    # 第3步：根据配置设定联邦优化/聚合算法，确保客户端与服务器使用同一套更新逻辑（如 FedAvg、Median 等）。
    coordinator.set_fl_algorithm(args, the_server, clients)
    args.logger.info("Clients and server are initialized")
    args.logger.info("Starting Training...")
    prev_aggregated_update = None
    low_acc_streak = 0
    for global_epoch in range(args.epochs):
        epoch_msg = f"Epoch {global_epoch:<3}"
        # print(f"Global epoch {global_epoch} begin")
        # server dispatches numpy version global weights 1d vector to clients
        global_weights_vec = the_server.global_weights_vec

        # clients' local training: broadcast, fit locally, store statistics for logging.
        # 客户端本地训练：先拉取新模型，再独立迭代若干 local epochs，并记录训练指标以便聚合与日志输出。
        avg_train_acc, avg_train_loss = [], []
        for client in clients:
            # pull the latest global model before each local update
            # 拉取最新全局模型，确保所有客户端从同一权重出发。
            client.load_global_model(global_weights_vec)
            train_acc, train_loss = client.local_training()
            # serialize updates (weights / gradients) so the server can aggregate them later
            # 序列化本地更新（权重向量/梯度），待会传回服务器参与聚合。
            client.fetch_updates()
            avg_train_acc.append(train_acc)
            avg_train_loss.append(train_loss)

        avg_train_loss = avg_value(avg_train_loss)
        avg_train_acc = avg_value(avg_train_acc)
        if avg_train_acc < 0.18:
            low_acc_streak += 1
        else:
            low_acc_streak = 0
        epoch_msg += f"  Train Acc: {avg_train_acc:.4f}  Train loss: {avg_train_loss:.4f}  "

        # perform post-training attacks, for omniscient model poisoning attack, pass all clients
        omniscient_attack(clients)

        # server collects weights from clients
        the_server.collect_updates(global_epoch)
        the_server.aggregation()  # run the configured robust mean/aggregator

        # # XXX：测试当前轮和上一轮的方向；
        # cos_similarity = float("nan")
        # global_update_l2 = float("nan")
        # if the_server.aggregated_update is None:
        #     current_update = None
        # elif torch.is_tensor(the_server.aggregated_update):
        #     current_update = the_server.aggregated_update.detach().cpu().numpy()
        # else:
        #     current_update = np.asarray(the_server.aggregated_update)
        # if prev_aggregated_update is not None and current_update is not None:
        #     prev_norm = np.linalg.norm(prev_aggregated_update)
        #     curr_norm = np.linalg.norm(current_update)
        #     if prev_norm > 0 and curr_norm > 0:
        #         cos_similarity = float(
        #             np.dot(prev_aggregated_update, current_update) / (prev_norm * curr_norm))
        #     global_update_l2 = float(curr_norm)
        # elif current_update is not None:
        #     global_update_l2 = float(np.linalg.norm(current_update))
        # prev_aggregated_update = np.copy(
        #     current_update) if current_update is not None else None
        # epoch_msg += f"Cos: {cos_similarity:.3f}  L2: {global_update_l2:.4f}  "

        # 服务器端执行指定聚合规则（如 FedAvg、Robust Aggregator）来融合更新。
        the_server.update_global()  # push the aggregated model back to the global buffer
        # 将聚合完成的全局模型写回缓冲区，供下一轮广播。
        # evalute the attack success rate (ASR) when a backdoor attack is launched
        # 若存在后门攻击，此处额外统计 ASR/主任务准确率等指标，用于衡量防御效果。
        test_stats = coordinator.evaluate(
            the_server, test_dataset, args, global_epoch)

        # print the training and testing results of the current global_epoch
        # 输出当前全局轮的训练/测试统计信息，便于追踪收敛与攻击成效。
        epoch_msg += "\t".join(
            [f"{key}: {value:.4f}" for key, value in test_stats.items()])
        
        if low_acc_streak >= 50:
            epoch_msg += "\nAttack succeeded."
            args.logger.info(epoch_msg)
            break

        args.logger.info(epoch_msg)
        # clear memory to reduce GPU/CPU pressure in long experiments
        # 清理 Python/GPU 缓存，避免长时间实验导致显存/内存膨胀。
        gc.collect()


    # XXX：输出各客户端本地更新方向的平均余弦相似度（已去除首轮）
    # for client in clients:
    #     num = len(client.local_cos_history)
    #     avg_cos = sum(client.local_cos_history[1:]) / (num - 1) if num > 1 else float("nan")
    #     print(
    #         f"Client {client.worker_id} local cos avg: {avg_cos:.6f}"
    #     )

    if args.record_time:
        # 可选：记录每个客户端与服务器端在通信/训练阶段的耗时，便于性能评估。
        report_time(clients, the_server)

    plot_accuracy(args.output)  # generate and save accuracy/ASR curves for later inspection
    # 绘制准确率/攻击成功率随全局轮数的曲线并保存，方便后续分析。

    end_time = time.time()
    time_difference = end_time - start_time
    minutes, seconds = int(
        time_difference // 60), int(time_difference % 60)
    args.logger.info(
        f"Training finished on {time.asctime(time.localtime(end_time))} using {minutes} minutes and {seconds} seconds in total.")


def report_time(clients, the_server):
    """
    汇总客户端与服务器的计时结果，输出到各自日志。

    参数:
        clients (List[Client]): 所有客户端对象，需具备 time_recorder 属性。
        the_server (Server): 服务器实例，同样具备 time_recorder。
    返回:
        None
    复杂度:
        时间复杂度 O(num_clients)。

    费曼学习法:
        (A) 函数调用每个节点的计时器，打印平均耗时信息。
        (B) 类比比赛结束后，主持人逐个播报选手与裁判的耗时统计。
        (C) 步骤拆解:
            - 遍历客户端列表，依次调用 `time_recorder.report`。
            - 最后对服务器执行同样的报告。
        (D) 示例:
            >>> report_time(clients, server)
        (E) 边界与测试:
            - 若节点未开启计时器，`time_recorder` 可能不存在，需要测试中确保属性准备就绪。
            - 建议测试: 1) 开启记录后调用并检查输出文件； 2) 对未开启计时的节点确认不会崩溃。
        (F) 背景参考:
            - 概念: 性能分析结果汇总。
            - 参考书籍: 《Performance Analysis in Distributed Systems》。
    """
    [c.time_recorder.report(f"Client {idx}") for idx, c in enumerate(clients)]
    the_server.time_recorder.report("Server")


def omniscient_attack(clients):
    """
    触发全知攻击，协调恶意客户端根据global信息篡改更新向量。

    参数:
        clients (List[Client]): 所有客户端对象，攻击者需实现 `omniscient` 方法。
    返回:
        None
    异常:
        ValueError: 当攻击者未生成更新向量时抛出。
    复杂度:
        时间复杂度 O(num_attackers × |update|)。

    费曼学习法:
        (A) 函数让全知攻击者共享信息并注入恶意更新。
        (B) 类比作弊团队交换答案，再分别填入试卷。
        (C) 步骤拆解:
            - 筛选出具有 `omniscient` 属性的攻击者。
            - 若不存在直接返回；否则由第一个攻击者生成恶意更新。
            - 判断返回是单个更新还是批量，按情况分发给各攻击者。
        (D) 示例:
            >>> omniscient_attack(clients)
        (E) 边界与测试:
            - 若攻击者返回 None，应抛出异常提醒测试此逻辑。
            - 建议测试: 1) 多个攻击者共享相同更新; 2) 单更新与多更新分发是否正确。
        (F) 背景参考:
            - 概念: 全知模型投毒攻击（omniscient attack）。
            - 参考书籍: 《Adversarial Machine Learning in Federated Systems》。
    """
    # Filter out all omniscient attackers from the client list
    omniscient_attackers = [
        client for client in clients
        if client.category == "attacker" and "omniscient" in client.attributes
    ]

    # If no omniscient attackers exist, exit early
    if not omniscient_attackers:
        return
    # Generate malicious updates using the first attacker's logic
    malicious_updates = omniscient_attackers[0].omniscient(clients)
    if malicious_updates is None:
        raise ValueError("No updates generated by the omniscient attacker")

    # Check if the malicious update is a single vector or a batch of updates
    is_single_update = len(
        malicious_updates.shape) == 1 or malicious_updates.shape[0] == 1

    if is_single_update:
        # If a single update is provided, all attackers perform their own attack
        omniscient_attackers[0].update = malicious_updates
        for client in omniscient_attackers[1:]:
            client.update = client.omniscient(clients)
    else:
        # If multiple updates are provided, assign each update to an attacker
        # An attack method aiming to provide the same updates for all attackers can return repeated updates.
        for client, update in zip(omniscient_attackers, malicious_updates):
            client.update = update


def main(args, cli_args):
    """
    总控函数：根据命令行参数选择单次实验或基准测试流程。

    参数:
        args (SimpleNamespace): 配置文件载入的基础参数。
        cli_args (argparse.Namespace): 命令行解析结果。
    返回:
        None
    复杂度:
        时间复杂度取决于 `fl_run` 开销。

    费曼学习法:
        (A) 函数根据是否启用 benchmark 决定预处理方案，并启动训练。
        (B) 类比比赛裁判先确认赛事模式：单场对决或循环赛，然后发令。
        (C) 步骤拆解:
            - 若 benchmark 开关为真，执行基准模式预处理并直接训练。
            - 否则先覆盖命令行参数、执行单次预处理，再启动训练。
        (D) 示例:
            >>> main(args, cli_args)
        (E) 边界与测试:
            - 需保证 `args` 在 benchmark 场景下包含完整攻防列表。
            - 建议测试: 1) benchmark=True 路径; 2) 单次实验路径。
        (F) 背景参考:
            - 概念: 实验调度与参数管理。
            - 参考书籍: 《Experiment Management for Machine Learning》。
    """
    # if Benchmarks is True, run all combinations of attacks and defenses
    if cli_args.benchmark:
        benchmark_preprocess(args)
        
        # XXX: 若任务已完成则跳过
        png_path = Path(args.output).with_suffix(".png")
        if png_path.exists():
            print(f"PNG {png_path.name} exists, skip")
            return

        fl_run(args)
    else:
        override_args(args, cli_args)
        single_preprocess(args)
        png_path = Path(args.output).with_suffix(".png")
        if png_path.exists():
            print(f"PNG {png_path.name} exists, skip")
            return
        fl_run(args)


if __name__ == "__main__":
    args, cli_args = read_args()
    main(args, cli_args)


# __AI_ANNOTATION_SUMMARY__
# - fl_run: 协调整个联邦学习训练流程，涵盖数据准备、训练、攻击与评估。
# - report_time: 汇总客户端与服务器的计时统计输出。
# - omniscient_attack: 对具全知能力的攻击者执行恶意更新分发。
# - main: 根据 benchmark 开关选择预处理路径并启动训练。
