"""
服务器端核心逻辑，负责聚合客户端更新、维护全局模型并协调联邦训练。

主要职责:
    - 初始化全局模型与聚合器；
    - 收集客户端更新并执行聚合；
    - 调用联邦算法完成全局参数更新。
"""

import numpy as np
import torch
from aggregators import get_aggregator
from fl.algorithms import get_algorithm_handler
from fl.models import get_model
from fl.models.model_utils import model2vec_torch
from fl.worker import Worker
from global_utils import TimingRecorder


class Server(Worker):
    """
    联邦学习服务器端基类，协调客户端更新并维护全局模型状态。

    属性:
        clients (List[Client]): 当前参与训练的客户端列表。
        global_model (torch.nn.Module): 持有的全局模型副本。
        global_weights_vec (torch.Tensor): 扁平化的全局模型参数，用于聚合与广播。
        aggregator (AggregatorBase): 决定如何聚合客户端更新的策略对象。
        aggregated_update (torch.Tensor): 最近一次聚合后的更新向量。
        test_dataset (Dataset): 用于评估的测试数据集引用。
        train_dataset (Dataset): 可选训练数据集，部分防御（如 FLTrust）需要。
        time_recorder (TimingRecorder): 选用的计时器，用于性能分析。
    """

    def __init__(self, args, clients, test_dataset, train_dataset):
        """
        初始化服务器端状态，包括全局模型、聚合器与计时器。

        参数:
            args (argparse.Namespace): 全局配置，包含防御策略、输出目录等。
            clients (List[Client]): 参与联邦训练的客户端实例列表。
            test_dataset (Dataset): 用于中心化评估的测试数据。
            train_dataset (Dataset): 服务器可访问的训练数据，防御策略可能使用。
        返回:
            None: 构造函数仅执行状态初始化。
        异常:
            RuntimeError: 在模型或聚合器初始化失败时由底层库抛出。
        复杂度:
            时间复杂度 O(|model| + num_clients)，空间复杂度 O(|model|)。

        费曼学习法:
            (A) 函数为服务器准备好协调训练所需的模型、聚合器与工具。
            (B) 好比总教练搭建训练总部：先准备总队模型、收集数据工具，再安排计时员。
            (C) 步骤拆解:
                - 调用父类构造函数，继承日志记录与公共方法。
                - 保存客户端、测试集与可能用于防御的训练数据引用。
                - 通过工厂方法 `get_model` 构建全局模型，保持与客户端一致。
                - 使用 `model2vec` 将模型参数展平，便于数值聚合。
                - 初始化聚合结果缓存 `aggregated_update`，尺寸匹配全局参数。
                - 根据防御策略获取聚合器实例，支持鲁棒或定制聚合。
                - 若开启时间记录，为 `aggregation` 方法加上计时装饰器。
            (D) 示例:
                >>> server = Server(args, clients, test_ds, train_ds)
                >>> server.global_model is not None
                True
            (E) 边界与测试:
                - 若 `clients` 列表为空，后续聚合可能产生空更新，需在测试中覆盖。
                - 若 `args.defense` 未匹配到聚合器，会触发 KeyError 或 RuntimeError。
                - 建议测试: 1) 使用 Mock 客户端验证初始化流程; 2) 模拟启用计时功能，确认装饰器生效。
            (F) 背景参考:
                - 概念: 联邦学习服务器与聚合策略。
                - 参考书籍: 《Federated Learning》对服务器职责有系统描述。
        """
        server_id = -1  # server worker_id = -1
        super().__init__(args, worker_id=server_id)
        # 持有当前参与训练的客户端列表，后续用于收集更新
        self.clients = clients
        # 测试数据通常仅服务器可见，用于全局模型评估
        self.test_dataset = test_dataset
        # 存储训练数据引用，部分防御（如 FLTrust）需使用可信数据
        self.train_dataset = train_dataset  # it's only used in the defense FLTrust

        # 初始化全局模型，使其结构与客户端保持一致
        self.global_model = get_model(args)
        # 将模型参数展平，方便执行向量化聚合
        self.global_weights_vec = model2vec_torch(self.global_model)

        # 初始化聚合后更新的缓存向量，初始为全零
        self.aggregated_update = torch.zeros_like(self.global_weights_vec)

        # 根据防御配置选择聚合器，实现鲁棒或定制化的聚合策略（工厂方法）
        self.aggregator = get_aggregator(
            self.args.defense)(self.args, train_dataset=self.train_dataset)

        if self.args.record_time:
            # 开启计时器，为关键聚合流程记录运行时间
            self.time_recorder = TimingRecorder(self.worker_id,
                                                self.args.output)
            self.aggregation = self.time_recorder.timing_decorator(
                self.aggregation)

    def set_algorithm(self, algorithm):
        """
        绑定服务器端使用的联邦优化算法处理器。

        参数:
            algorithm (str): 算法名称，对应 `fl.algorithms` 中的处理器工厂键。
        返回:
            None: 算法实例保存在服务器对象中。
        异常:
            KeyError: 当传入算法不存在时由工厂方法抛出。
        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)。

        费曼学习法:
            (A) 函数让服务器知道如何根据聚合结果更新全局模型。
            (B) 类比总教练选择训练计划，决定如何把运动员成绩转化为团队战术。
            (C) 步骤拆解:
                - 调用算法工厂 `get_algorithm_handler` 获取指定算法。
                - 传入全局模型构建算法实例，准备后续的更新步骤。
            (D) 示例:
                >>> server.set_algorithm("fedavg")
                >>> hasattr(server, "algorithm")
                True
            (E) 边界与测试:
                - 若未调用该函数，`update_global` 将因缺少 `self.algorithm` 而失败。
                - 建议测试: 1) 对支持的算法逐一调用并检查返回实例类型; 2) 对非法名称断言抛出 KeyError。
            (F) 背景参考:
                - 概念: 联邦优化算法处理器（FedAvg、FedOpt 等）。
                - 参考书籍: 《Federated Learning》综述章节。
        """
        self.algorithm = get_algorithm_handler(
            algorithm)(self.args, self.global_model)

    def collect_updates(self, global_epoch):
        """
        收集所有客户端在当前全局轮次上报的更新向量。

        参数:
            global_epoch (int): 当前全局通信轮次编号。
        返回:
            None: 更新结果存储在 `self.client_updates` 数组中。
        异常:
            AttributeError: 若某客户端未设置 `update` 属性，访问时会触发。
        复杂度:
            时间复杂度 O(num_clients × |update|)，空间复杂度 O(num_clients × |update|)。

        费曼学习法:
            (A) 函数把所有客户端提交的更新收集到服务器端。
            (B) 好比教练收集每位队员的训练成绩单，准备后续汇总。
            (C) 步骤拆解:
                - 记录当前全局轮次编号，方便聚合器或算法使用。
                - 遍历客户端列表，提取各自的 `update` 并转为 NumPy 数组。
            (D) 示例:
                >>> server.collect_updates(global_epoch=5)
                >>> server.client_updates.shape[0] == len(server.clients)
                True
            (E) 边界与测试:
                - 若 `clients` 为空，`self.client_updates` 将是空数组，聚合器需能处理。
                - 若某客户端未调用 `fetch_updates`，其 `update` 可能不存在或为 None。
                - 建议测试: 1) 模拟不同数量客户端的收集过程; 2) 针对空更新的处理行为进行断言。
            (F) 背景参考:
                - 概念: 联邦学习通信轮次与更新收集。
                - 参考书籍: 《Federated Learning》相关章节。
        """
        self.global_epoch = global_epoch
        # 将各客户端的更新向量堆叠为二维数组，供聚合器统一处理
        device = self.global_weights_vec.device
        updates = []
        for client in self.clients:
            upd = client.update
            if torch.is_tensor(upd):
                updates.append(upd.to(device=device))
            else:
                updates.append(torch.as_tensor(upd, device=device))
        self.client_updates = torch.stack(updates, dim=0) if updates else torch.empty((0, 0), device=device)

    def aggregation(self):
        """
        使用配置的聚合策略整合客户端更新，生成聚合后的全局更新。

        参数:
            None
        返回:
            None: 聚合结果保存在 `self.aggregated_update` 中。
        异常:
            ValueError: 若聚合器在输入格式不符时可能抛出。
        复杂度:
            时间复杂度取决于聚合器，常见情况下为 O(num_clients × |update|)。

        费曼学习法:
            (A) 函数将各客户端的更新整合成一个全局更新向量。
            (B) 类比教练根据队员成绩单做加权平均，制定整体改进方向。
            (C) 步骤拆解:
                - 调用聚合器的 `aggregate` 方法，传入所有客户端更新。
                - 同时传递当前全局模型、扁平化权重和轮次信息，供鲁棒聚合使用。
                - 将聚合出的更新向量存入 `self.aggregated_update`。
            (D) 示例:
                >>> server.aggregation()
                >>> server.aggregated_update.shape == server.global_weights_vec.shape
                True
            (E) 边界与测试:
                - 若 `self.client_updates` 为空，需确认聚合器是否返回零向量。
                - 若聚合器依赖 `train_dataset` 等额外信息，应确保初始化正确。
                - 建议测试: 1) 使用已知更新向量验证聚合结果; 2) 模拟异常输入检测聚合器的鲁棒性。
            (F) 背景参考:
                - 概念: 鲁棒聚合（如 Krum、Trimmed Mean）。
                - 参考书籍: 《Adversarial Robustness in Federated Learning》。
        """
        # 聚合器内部决定是求梯度平均 (FedSGD) 还是模型平均 (FedAvg) 等
        self.aggregated_update = self.aggregator.aggregate(
            self.client_updates,
            last_global_model=self.global_model,
            global_weights_vec=self.global_weights_vec,
            global_epoch=self.global_epoch,
        )
        if not torch.is_tensor(self.aggregated_update):
            self.aggregated_update = torch.as_tensor(
                self.aggregated_update, device=self.global_weights_vec.device
            )

        # 增加L2 clip防止梯度爆炸
        max_norm = 100.0
        update_norm = torch.linalg.norm(self.aggregated_update)
        if update_norm > max_norm:
            self.aggregated_update = self.aggregated_update * (
                max_norm / (update_norm + 1e-12)
            )

    def update_global(self):
        """
        调用联邦算法处理器，将聚合更新应用于全局模型参数。

        参数:
            None
        返回:
            None: 更新后的扁平化权重存回 `self.global_weights_vec`。
        异常:
            AttributeError: 若未先调用 `set_algorithm`，`self.algorithm` 不存在。
        复杂度:
            时间复杂度 O(|model|)，空间复杂度 O(|model|)。

        费曼学习法:
            (A) 函数根据聚合结果更新服务器持有的全局模型。
            (B) 类比总教练根据综合成绩调整团队训练参数。
            (C) 步骤拆解:
                - 调用算法实例的 `update` 方法，将聚合向量与当前权重结合。
                - 将返回的新权重向量写回 `self.global_weights_vec`，待后续同步给客户端。
            (D) 示例:
                >>> server.update_global()
                >>> server.global_weights_vec is not None
                True
            (E) 边界与测试:
                - 若 `self.aggregated_update` 未设置，算法可能使用全零更新，需提前检查。
                - 建议测试: 1) 使用简单模型验证一次聚合更新的正确性; 2) 模拟重复调用保持权重一致性。
            (F) 背景参考:
                - 概念: 联邦模型更新步骤。
                - 参考书籍: 《Federated Learning》算法实现章节。
        """
        # 基于当前算法处理聚合结果，更新扁平化权重向量
        self.global_weights_vec = self.algorithm.update(
            self.aggregated_update, global_weights_vec=self.global_weights_vec
        )


# __AI_ANNOTATION_SUMMARY__
# - Server.__init__: 初始化全局模型、聚合器与计时器，为服务器协调流程铺垫。
# - Server.set_algorithm: 绑定联邦算法处理器，决定全局参数更新方式。
# - Server.collect_updates: 收集客户端提交的更新向量并记录当前轮次。
# - Server.aggregation: 依据聚合策略整合客户端更新，生成全局更新向量。
# - Server.update_global: 调用算法将聚合结果应用到全局模型参数上。
