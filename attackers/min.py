# -*- coding: utf-8 -*-

import numpy as np
from global_utils import actor
from attackers.pbases.mpbase import MPBase
from attackers import attacker_registry
from fl.client import Client


class MinBase(MPBase, Client):
    """Min 系列模型投毒攻击基类，依据梯度距离度量构造隐蔽的恶意更新。

    该族攻击源自 NDSS 2021 论文《Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning》，
    通过约束恶意更新与良性更新的距离，不易被鲁棒聚合策略侦测。

    属性:
        default_attack_params (dict): 默认攻击参数，包含 `gamma_init` 与 `stop_threshold`。
        algorithm (str): 默认假设的聚合算法，这里为 `FedSGD`。
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        """初始化 Min 攻击基类，设置扰动初值与搜索阈值。

        概述:
            调用 `Client` 构造器完成基础上下文，写入二分搜索起点 `gamma_init` 以及终止阈值 `stop_threshold`。

        参数:
            args (argparse.Namespace): 运行配置，需包含 `num_adv`、`epochs` 等字段。
            worker_id (int): 当前攻击客户端编号。
            train_dataset (Dataset): 本地训练数据集。
            test_dataset (Dataset): 本地测试数据集。

        返回:
            None。

        异常:
            AttributeError: 缺失必要字段时由父类抛出。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)。

        费曼学习法:
            (A) 函数设置 Min 攻击的搜索起点与收敛阈值。
            (B) 类比在调试天线时先给出初始信号强度和停止条件。
            (C) 步骤拆解:
                1. 调用 `Client.__init__` 获取联邦上下文。
                2. 定义默认参数（初始 γ 与停止阈值）。
                3. 通过 `update_and_set_attr` 合并外部配置。
                4. 记录默认聚合算法为 `FedSGD`。
            (D) 示例:
                >>> attacker = MinBase(args, worker_id=0, train_dataset=train, test_dataset=test)
                >>> attacker.gamma_init
                10.0
            (E) 边界条件与测试建议: 确保 `gamma_init` 为正；测试外部覆写是否成功。
            (F) 背景参考: NDSS 2021 MinMax/MinSum 模型投毒。
        """
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {
            'gamma_init': 10.0,
            'stop_threshold': 1e-5,
        }
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

    def omniscient(self, clients):
        """在全知场景下运行 Min 攻击，生成恶意更新向量。

        概述:
            根据当前子类名称（MinSum 或 MinMax）选择度量函数，
            通过 `Min` 函数求解最优恶意更新，并为全部攻击者复制。

        参数:
            clients (List[Client]): 本轮参与的客户端列表。

        返回:
            numpy.ndarray: 形状 `(num_adv, d)` 的攻击向量矩阵。

        异常:
            ValueError: 当良性客户端不足或度量计算失败时由 `Min` 抛出。

        复杂度:
            时间复杂度 O(m * d * log(γ范围))，m 为良性客户端数；空间复杂度 O(d)。

        费曼学习法:
            (A) 函数调用具体的 Min 攻击求解器并复制结果给所有攻击者。
            (B) 类比找到一条最隐蔽的潜行路线后，让所有同伙照搬路线行动。
            (C) 步骤拆解:
                1. 读取当前类名，区分 MinMax 或 MinSum。
                2. 调用 `Min` 函数，输入度量类型与搜索参数。
                3. 使用 `np.tile` 将单个恶意向量复制给全部攻击者。
            (D) 示例:
                >>> attack_vecs = attacker.omniscient(clients)
                >>> attack_vecs.shape
                (attacker.args.num_adv, len(attacker.update))
            (E) 边界条件与测试建议: 保证至少有一个良性客户端；测试输出行是否一致。
            (F) 背景参考: Min 攻击求解流程、鲁棒聚合对抗分析。
        """
        attack_vec = Min(
            clients,
            self.__class__.__name__,
            'unit_vec',
            self.gamma_init,
            self.stop_threshold,
        )
        # repeat attack vector for all attackers
        return np.tile(attack_vec, (self.args.num_adv, 1))


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class MinMax(MinBase):
    """MinMax 攻击实现：约束恶意更新的最大距离不超过良性之间的最大距离。"""
    pass


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class MinSum(MinBase):
    """MinSum 攻击实现：约束恶意更新与良性之间距离和不超过良性之间的距离和。"""
    pass


def get_metrics(metric_type):
    """根据攻击类型返回对应的距离度量函数。

    概述:
        MinMax 使用最大范数距离，MinSum 使用平方范数和。

    参数:
        metric_type (str): 攻击类型，取值 `'MinMax'` 或 `'MinSum'`。

    返回:
        Callable[[np.ndarray], float]: 距离度量函数。

    异常:
        KeyError: 当 `metric_type` 不支持时建议调用方抛错（此处默认返回 `None`）。

    复杂度:
        时间复杂度 O(n)，n 为输入向量数量；空间复杂度 O(1)。

    费曼学习法:
        (A) 函数选择一种“距离尺子”来衡量恶意更新离良性更新有多远。
        (B) 类比在裁缝店根据需求选择最长边尺或面积尺。
        (C) 步骤拆解:
            1. 判断类型是否为 MinMax，返回最大 L2 范数。
            2. 若为 MinSum，返回平方范数之和。
            3. 其他情况返回 `None`（调用者需检查）。
        (D) 示例:
            >>> metric = get_metrics('MinMax')
            >>> metric(np.array([[1, 0], [0, 1]]))
            1.0
        (E) 边界条件与测试建议: 对非法类型给出明确提示；测试两种类型是否返回预期函数。
        (F) 背景参考: 向量范数、鲁棒聚合中的距离度量。
    """
    if metric_type == 'MinMax':
        def metric(x): return np.linalg.norm(x, axis=1).max()
    elif metric_type == 'MinSum':
        def metric(x): return np.square(np.linalg.norm(x, axis=1)).sum()
    else:
        metric = None
    return metric


def Min(clients, type, dev_type, gamma_init, stop_threshold):
    """求解 Min 攻击的恶意更新向量。

    概述:
        计算良性更新均值与偏差方向，通过二分搜索 γ（λ）参数，
        使恶意更新在选定度量下不超过良性更新之间的最大（或总）距离。

    参数:
        clients (List[Client]): 当前参与训练的客户端。
        type (str): 攻击类型（MinMax 或 MinSum）。
        dev_type (str): 偏差方向类型，支持 `'unit_vec'`、`'sign'`、`'std'`。
        gamma_init (float): γ 的初始值，也是搜索上界。
        stop_threshold (float): 二分搜索终止阈值。

    返回:
        numpy.ndarray: 最终的恶意更新向量。

    异常:
        ValueError: 若不存在良性客户端或 `dev_type` 无法识别。

    复杂度:
        时间复杂度 O(m * d * log(γ范围))，m 为良性客户端数；空间复杂度 O(d)。

    费曼学习法:
        (A) 该函数像调试天线一样，逐步调整 γ，让恶意信号既强又不露馅。
        (B) 类比将假钞混入真币堆里，通过逐步缩放假钞图案确保机器检验不过界。
        (C) 步骤拆解:
            1. 获取对应类型的距离度量函数。
            2. 收集良性更新并计算均值（代表主方向）。
            3. 根据 `dev_type` 选择扰动方向（单位向量 / 符号 / 标准差）。
            4. 计算良性更新之间的上界距离（或距离和）。
            5. 以二分搜索调整 γ：若恶意更新仍在上界内则增大 γ，否则缩小。
            6. 收敛后输出最优恶意更新。
        (D) 示例:
            >>> mal = Min(clients, 'MinMax', 'unit_vec', gamma_init=10.0, stop_threshold=1e-5)
        (E) 边界条件与测试建议: 确认存在至少一个良性客户端；测试不同 `dev_type` 输出是否合理。
        (F) 背景参考: 二分搜索、向量范数约束、Min 攻击原理。
    """
    metric = get_metrics(type)
    if metric is None:
        raise ValueError(f"Unsupported metric type: {type}")

    # 收集所有良性客户端的更新，并计算均值作为基线方向。
    benign_update = np.array(
        [i.update for i in clients if i.category == "benign"])
    if benign_update.size == 0:
        raise ValueError("No benign clients available for Min attack.")
    benign_mean = np.mean(benign_update, axis=0)

    # 根据偏差类型选择扰动方向；默认使用单位向量。
    if dev_type == 'unit_vec':
        deviation = benign_mean / np.linalg.norm(benign_mean)
    elif dev_type == 'sign':
        deviation = np.sign(benign_mean)
    elif dev_type == 'std':
        deviation = np.std(benign_update, axis=0)
    else:
        raise ValueError(f"Unsupported deviation type: {dev_type}")

    lamda, step, lamda_succ = gamma_init, gamma_init / 2, 0

    # 计算良性更新之间的度量上界，确保恶意更新隐藏在该范围内。
    upper_bound = np.max([
        metric(benign_update - benign_update[i])
        for i in range(len(benign_update))
    ])

    # 二分搜索 γ，使恶意更新既尽量偏离又不超出上界。
    while np.abs(lamda_succ - lamda) > stop_threshold:
        # 使用当前 γ 对均值进行扰动，得到候选恶意更新。
        mal_update = benign_mean - lamda * deviation
        # 评估该恶意更新与良性更新的距离。
        mal_metric_value = metric(benign_update - mal_update)

        if mal_metric_value <= upper_bound:
            lamda_succ = lamda
            lamda += step
        else:
            lamda -= step
        step /= 2

    # 输出最终恶意更新。
    mal_update = benign_mean - lamda_succ * deviation
    return mal_update


# __AI_ANNOTATION_SUMMARY__
# 类 MinBase: Min 攻击基类，封装默认参数与全知式求解。
# 方法 __init__ (MinBase): 初始化 γ 与停止阈值。
# 方法 omniscient (MinBase): 调用 Min 求解器并复制恶意更新。
# 类 MinMax: 基于最大距离约束的 Mimic 攻击子类。
# 类 MinSum: 基于距离和约束的 Mimic 攻击子类。
# 函数 get_metrics: 返回 Min 攻击所需的距离度量函数。
# 函数 Min: 二分搜索 γ 生成满足约束的恶意更新向量。
