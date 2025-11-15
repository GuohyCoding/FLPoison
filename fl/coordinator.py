"""
协调器工具函数，负责构建客户端集合、配置联邦算法并执行全局评估。

该模块对外提供以下功能:
    - init_clients: 根据攻击设定实例化客户端或攻击者。
    - set_fl_algorithm: 为服务器与客户端同步联邦优化算法。
    - evaluate: 在测试集上执行全局与攻击相关指标评估。
"""

from collections import OrderedDict
from .client import Client
from attackers import get_attacker_handler
from datapreprocessor.data_utils import subset_by_idx
from attackers import data_poisoning_attacks, hybrid_attacks


def init_clients(args, client_indices, train_dataset, test_dataset):
    """
    根据配置与索引划分初始化所有联邦客户端或攻击者实例。

    参数:
        args (argparse.Namespace): 全局运行配置，包含客户端数量、攻击类型等。
        client_indices (List[List[int]]): 每个客户端对应的样本索引集合。
        train_dataset (Dataset): 完整训练数据集，将按索引切分。
        test_dataset (Dataset): 共享测试数据集引用，供客户端评估使用。
    返回:
        List[Client]: 已实例化的客户端对象列表，顺序对应 worker_id。
    异常:
        AssertionError: 当配置了攻击但攻击者数量为 0 时抛出。
    复杂度:
        时间复杂度 O(num_clients × |subset|)，空间复杂度 O(num_clients)。

    费曼学习法:
        (A) 该函数按配置生成所有参与训练的客户端或攻击者。
        (B) 类比在比赛开始前，把运动员按队伍分组并发放训练材料。
        (C) 步骤拆解:
            - 初始化空列表，用来存放每个创建的客户端实例。
            - 遍历所有 worker_id，根据攻击配置决定使用普通客户端还是攻击者类。
            - 调用 `subset_by_idx` 将训练集切分为当前客户端的本地子集。
            - 使用选定的类实例化客户端，传入本地数据与测试集引用。
            - 将实例追加到列表中，保持与 worker_id 对应的顺序。
        (D) 示例:
            >>> clients = init_clients(args, client_indices, train_ds, test_ds)
            >>> len(clients) == args.num_clients
            True
        (E) 边界与测试:
            - 若 `client_indices` 长度与 `args.num_clients` 不匹配，将在索引访问时抛错。
            - 若攻击开启但 `args.num_adv` 为 0，会触发 AssertionError。
            - 建议测试: 1) 使用少量数据验证各客户端数据量是否正确; 2) 模拟攻击场景检查前 `num_adv` 个客户端是否为攻击者类型。
        (F) 背景参考:
            - 概念: 客户端采样与数据划分。
            - 参考书籍: 《Federated Learning》关于数据分片的章节。
    """
    clients = []
    for worker_id in range(args.num_clients):
        # for attacker, if the attack type is not model poisoning attack, use the default client class. For data poisoning attacks, it's already handled in the client class.
        # for benign clients, use the default client class
        if args.attack == "NoAttack":
            """
            For NoAttack scenario, use client class, and ignore args.num_adv
            """
            client_obj = Client
        else:
            if args.num_adv == 0:
                raise AssertionError(
                    "Attack {args.attack} specified, but attackers set to 0.")
            client_obj = Client if worker_id >= args.num_adv else get_attacker_handler(
                args.attack)
        # 根据分配的样本索引提取客户端本地数据子集
        local_dataset = subset_by_idx(
            args, train_dataset, client_indices[worker_id])
        # 实例化客户端或攻击者，传入本地数据与测试集引用
        tmp_client = client_obj(args, worker_id,
                                local_dataset, test_dataset)
        clients.append(tmp_client)
    return clients


def set_fl_algorithm(args, the_server, clients):
    """
    为服务器与客户端设定联邦优化算法，确保各方使用一致策略。

    参数:
        args (argparse.Namespace): 全局配置，包含 `algorithm` 指定项。
        the_server (Server): 联邦服务器实例，负责聚合全局模型。
        clients (List[Client]): 客户端对象列表。
    返回:
        None: 设置过程直接作用于服务器与客户端对象。
    异常:
        ValueError: 当既无显式算法参数也无服务器默认值时抛出。
    复杂度:
        时间复杂度 O(num_clients)，空间复杂度 O(1)。

    费曼学习法:
        (A) 函数让服务器和所有客户端采用同一种联邦训练算法。
        (B) 好比裁判发布正式比赛规则，所有选手都得按同一个标准执行。
        (C) 步骤拆解:
            - 优先检查 `args.algorithm` 是否给出算法名称。
            - 若未指定，则尝试读取服务器自带的默认算法类型。
            - 若两者都缺失，则抛出异常提示配置不完整。
            - 调用服务器的 `set_algorithm` 方法完成配置。
            - 逐一调用客户端的 `set_algorithm`，保持策略一致。
        (D) 示例:
            >>> set_fl_algorithm(args, server, clients)
            >>> server.algorithm == args.algorithm
            True
        (E) 边界与测试:
            - 若服务器未实现 `set_algorithm`，此函数会触发 AttributeError。
            - 若 `clients` 列表为空，循环将跳过但不会报错，需确认这是预期行为。
            - 建议测试: 1) 在缺省 `args.algorithm` 下依靠服务器默认值; 2) 刻意清除两者验证 ValueError 是否被抛出。
        (F) 背景参考:
            - 概念: 联邦优化算法选择（FedAvg、FedOpt 等）。
            - 参考书籍: 《Federated Learning》对常见算法有综述。
    """
    if args.algorithm:
        alg_type = args.algorithm
    elif hasattr(the_server, 'algorithm'):
        args.algorithm = the_server.algorithm
    else:
        raise ValueError(
            "No specified algorithm or default algorithm type of the server. Please specify an algorithm type, with `--algorithm`")

    the_server.set_algorithm(alg_type)
    for client in clients:
        client.set_algorithm(alg_type)


def evaluate(the_server, test_dataset, args, global_epoch):
    """
    在测试数据集上评估全局模型性能及攻击成功率（若适用）。

    参数:
        the_server (Server): 当前持有全局模型的服务器实例。
        test_dataset (Dataset): 用于评估的测试数据集。
        args (argparse.Namespace): 配置对象，决定是否计算 ASR 等指标。
        global_epoch (int): 当前全局通信轮次计数（本函数未直接使用）。
    返回:
        OrderedDict: 包含准确率、损失以及在攻击场景下的 ASR 指标。
    异常:
        ValueError: 若测试数据构建数据加载器失败，底层可能抛出。
    复杂度:
        时间复杂度 O(|test_dataset|)，空间复杂度 O(1)。

    费曼学习法:
        (A) 函数评估当前全局模型的干净性能与后门攻击成功率。
        (B) 类比期末考试既要考常规题，也要测试是否被作弊题目迷惑。
        (C) 步骤拆解:
            - 初始化待记录的指标键值，并在存在类不平衡时加入尾类准确率。
            - 调用服务器的 `get_dataloader` 构建测试加载器。
            - 使用服务器的 `test` 方法获得干净测试结果并填入字典。
            - 若攻击为数据投毒或混合类型，则调用首个攻击客户端的 `client_test`
              以 `poison_epochs=True` 模式计算攻击成功率与损失。
            - 返回整理好的指标字典，供上层记录或展示。
        (D) 示例:
            >>> metrics = evaluate(server, test_ds, args, global_epoch=10)
            >>> metrics["Test Acc"]
            0.87
        (E) 边界与测试:
            - `args.distribution` 中若未包含 'imbalanced' 字样，将不计算尾类准确率。
            - 若 `args.attack` 不在指定列表中，ASR 指标保持默认值。
            - 建议测试: 1) 在无攻击场景下验证输出键是否正确; 2) 在模拟投毒场景下确认 ASR 被计算。
        (F) 背景参考:
            - 概念: 攻击成功率（Attack Success Rate, ASR）评估。
            - 参考书籍: 《Adversarial Machine Learning》对后门评估方法有介绍。
    """
    test_keys = ["Test Acc", "Test loss", "ASR", "ASR loss"]
    results = OrderedDict()

    # normal evaluation
    imbalanced_flag = True if 'imbalanced' in args.distribution else False
    if imbalanced_flag:
        test_keys.insert(1, 'Tail Acc')

    test_loader = the_server.get_dataloader(test_dataset, train_flag=False)
    clean_test = the_server.test(
        the_server.global_model, test_loader, imbalanced=imbalanced_flag)

    for idx in range(len(clean_test)):
        results[test_keys[idx]] = clean_test[idx]

    if args.attack in data_poisoning_attacks + hybrid_attacks:
        # index [0, f] is poisoning attacker
        results['ASR'], results['ASR loss'] = the_server.clients[0].client_test(
            the_server.global_model, test_dataset, poison_epochs=True)
    return results


# __AI_ANNOTATION_SUMMARY__
# - init_clients: 根据攻击配置实例化全部客户端并划分本地数据子集。
# - set_fl_algorithm: 同步服务器与客户端使用的联邦优化算法。
# - evaluate: 评估全局模型表现并在适用场景下计算攻击成功率。
