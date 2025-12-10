import torch
from fl.algorithms import get_algorithm_handler
from fl.models import get_model
from fl.models.model_utils import vec2model
from fl.worker import Worker
from global_utils import actor, avg_value
from global_utils import TimingRecorder


@actor("benign", "always")
class Client(Worker):
    """
    联邦学习客户端基类，封装本地训练、更新提交与性能评估流程。

    属性:
        args (argparse.Namespace): 全局运行配置，包含模型、优化器与数据相关参数。
        worker_id (int): 客户端唯一标识符，用于区分不同参与节点。
        train_dataset (Dataset): 客户端可访问的本地训练数据集。
        test_dataset (Dataset, optional): 客户端维护的测试数据集，可选。
        model (torch.nn.Module): 客户端当前持有的模型副本。
        optimizer (torch.optim.Optimizer): 与模型绑定的优化器实例。
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器。
        train_loader (Iterable): 训练数据迭代器，通常为 DataLoader。
        global_weights_vec (torch.Tensor): 最近一次接收的全局参数向量。
        update (torch.Tensor): 将要上报给服务器的本地更新。
        global_epoch (int): 已参与的全局通信轮次计数。

    说明:
        该类为良性与恶意客户端提供统一接口，便于扩展多种攻击与防御策略。
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset=None):
        """
        初始化客户端，加载模型、优化流程与默认数据迭代器。

        参数:
            args (argparse.Namespace): 框架运行配置，包括模型结构、优化器设置等。
            worker_id (int): 客户端在联邦系统中的唯一编号。
            train_dataset (Dataset): 客户端拥有的本地训练数据。
            test_dataset (Dataset, 可选): 用于本地评估的测试数据，缺省为 None。
        返回:
            None: 构造函数仅完成状态初始化，不返回值。
        异常:
            RuntimeError: 若模型或优化器构建失败，底层库可能抛出该异常。
        复杂度:
            时间复杂度 O(|model| + |dataset|)，空间复杂度 O(|model|)。

        费曼学习法:
            (A) 函数负责准备本地训练所需的全部资源与默认配置。
            (B) 好比运动员入队，领取教练安排的装备、训练计划与考勤表。
            (C) 步骤拆解:
                - 调用父类构造函数，继承通信与日志能力，避免重复实现基础功能。
                - 保存训练/测试数据引用，为后续构建迭代器提供数据源。
                - 初始化全局轮次计数器，记录客户端与服务器的同步进度。
                - 通过模型工厂 `get_model` 获得与全局一致的模型结构。
                - 依据模型实例化优化器与学习率调度器，确定参数更新策略。
                - 构建训练数据加载器，将原始数据整理为批次供训练使用。
                - 根据配置决定是否启用时间记录器，为后续性能分析奠定基础。
            (D) 示例:
                >>> client = Client(args, worker_id=1, train_dataset=train_ds, test_dataset=test_ds)
                >>> isinstance(client.model, torch.nn.Module)
                True
            (E) 边界与测试:
                - 若 `train_dataset` 为空或类型不兼容，`get_dataloader` 会触发异常。
                - 若 `args` 缺少优化器参数，`get_optimizer_scheduler` 会失败。
                - 建议测试: 1) 使用虚拟数据集验证初始化流程; 2) 模拟缺失配置时捕获异常信息。
            (F) 背景参考:
                - 概念: 联邦学习客户端初始化。
                - 参考书籍: 《Federated Learning》对客户端工作流有系统介绍。
        """
        Worker.__init__(self, args, worker_id)
        # 保存本地数据引用，确保训练与评估可访问原始样本
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        # 初始化全局轮次计数器，跟踪客户端参与情况
        self.global_epoch = 0
        # XXX???????????????????????????
        self.prev_local_update = None
        # XXX???????????????????
        self.local_cos_history = []
        # 通过工厂函数创建模型，保证结构与全局配置一致
        self.model = get_model(args)
        # 初始化优化器与学习率调度器，决定参数更新策略
        self.optimizer, self.lr_scheduler = self.get_optimizer_scheduler(
            self.model)

        # 构建默认训练数据加载器，后续可按需替换
        self.train_loader = self.get_dataloader(
            self.train_dataset, train_flag=True)

        self.record_time(self.args.record_time)

    def record_time(self, record_time):
        """
        按需为关键方法启用计时装饰器，以评估运行时间开销。

        参数:
            record_time (bool): 是否开启计时功能。
        返回:
            None: 若开启，会替换目标方法为计时包装版本。
        异常:
            AttributeError: 当尝试为不存在的 `omniscient` 方法装饰时可能出现。
        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)。

        费曼学习法:
            (A) 函数决定是否给耗时操作加上计时器。
            (B) 像给运动员佩戴计步器，帮助记录训练表现。
            (C) 步骤拆解:
                - 若允许记录时间，实例化 `TimingRecorder`，准备写入日志。
                - 用计时装饰器包装 `local_training` 与 `fetch_updates` 等核心方法。
                - 若客户端具备 `omniscient` 行为，同样进行包装以保持度量一致。
            (D) 示例:
                >>> client.record_time(True)
                >>> isinstance(client.time_recorder, TimingRecorder)
                True
            (E) 边界与测试:
                - 若 `record_time` 为 False，此函数应保持方法引用不变。
                - 建议测试: 1) 装饰前后调用函数并检查时间日志输出; 2) 对缺失 `omniscient` 方法的客户端调用确保不报错。
            (F) 背景参考:
                - 概念: 函数装饰器与性能剖析。
                - 参考书籍: 《Python Cookbook》中关于装饰器的章节。
        """
        if record_time:
            # 创建时间记录器，将测量结果写入指定输出目录
            self.time_recorder = TimingRecorder(self.worker_id,
                                                self.args.output)
            # 为本地训练与更新提交方法注入计时逻辑
            self.local_training = self.time_recorder.timing_decorator(
                self.local_training)
            self.fetch_updates = self.time_recorder.timing_decorator(
                self.fetch_updates)
            self.omniscient = self.time_recorder.timing_decorator(
                self.omniscient) if hasattr(self, "omniscient") else None

    def set_algorithm(self, algorithm):
        """
        加载指定联邦算法处理器，并初始化本地训练轮次配置。

        参数:
            algorithm (str): 算法名称，对应 `fl.algorithms` 中的处理器工厂键。
        返回:
            None: 完成算法处理器初始化与本地轮次设定。
        异常:
            KeyError: 当算法名称无对应处理器时，由工厂函数抛出。
        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)。

        费曼学习法:
            (A) 函数让客户端知道本轮训练应遵循哪种优化规则。
            (B) 好比给运动员分发训练计划，说明本阶段重点练习内容。
            (C) 步骤拆解:
                - 通过处理器工厂获取算法实现，确保与服务器策略一致。
                - 调用算法的 `init_local_epochs` 方法，确定本地迭代次数。
            (D) 示例:
                >>> client.set_algorithm("fedavg")
                >>> client.local_epochs > 0
                True
            (E) 边界与测试:
                - 若传入未知算法名称，将触发 KeyError。
                - 建议测试: 1) 对受支持算法逐一调用并断言 `local_epochs` 合理; 2) 对非法名称断言抛出异常。
            (F) 背景参考:
                - 概念: 联邦优化算法（如 FedAvg、FedOpt）。
                - 参考书籍: 《Federated Learning》对主流算法的综述。
        """
        self.algorithm = get_algorithm_handler(
            algorithm)(self.args, self.model, self.optimizer)
        # 客户端侧可自定义本地训练轮数，以平衡通信与计算开销
        self.local_epochs = self.algorithm.init_local_epochs()

    def load_global_model(self, global_weights_vec):
        """
        接收服务器的全局模型向量，并同步更新本地模型参数。

        参数:
            global_weights_vec (torch.Tensor): 扁平化的全局权重表示。
        返回:
            None: 更新本地模型参数，无显式返回值。
        异常:
            ValueError: 当权重向量维度与模型不匹配时，底层函数可能抛出。
        复杂度:
            时间复杂度 O(|model|)，空间复杂度 O(|model|)。

        费曼学习法:
            (A) 函数让客户端保持与服务器最新模型同步。
            (B) 好比收到总教练发来的最新动作示范，立刻校准姿势。
            (C) 步骤拆解:
                - 缓存全局权重向量，供后续生成更新时参考。
                - 调用 `vec2model` 将向量映射回模型参数，完成同步。
            (D) 示例:
                >>> global_vec = torch.nn.utils.parameters_to_vector(client.model.parameters())
                >>> client.load_global_model(global_vec)
            (E) 边界与测试:
                - 若 `global_weights_vec` 尺寸不符，`vec2model` 会触发异常。
                - 建议测试: 1) 使用已知参数向量还原模型并校验数值; 2) 对错误尺寸向量断言报错。
            (F) 背景参考:
                - 概念: 参数下发与模型同步。
                - 参考书籍: 《Federated Learning》关于模型聚合的章节。
        """
        self.global_weights_vec = global_weights_vec
        # 将服务器下发的参数向量写入本地模型，实现同步
        vec2model(self.global_weights_vec, self.model)

    def local_training(self, model=None, train_loader=None, optimizer=None, criterion_fn=None, local_epochs=None):
        """
        执行本地训练过程，返回平均准确率与损失值。

        参数:
            model (torch.nn.Module, 可选): 自定义模型实例，默认使用客户端当前模型。
            train_loader (Iterable, 可选): 训练数据迭代器，可重载以支持自定义数据流。
            optimizer (torch.optim.Optimizer, 可选): 指定优化器，默认沿用 `self.optimizer`。
            criterion_fn (Callable, 可选): 损失函数，默认使用 `self.criterion_fn`。
            local_epochs (int, 可选): 本地迭代轮数，默认等于 `self.local_epochs`。
        返回:
            Tuple[float, float]: 平均准确率与平均损失。
        异常:
            StopIteration: 若提供的迭代器在训练期内耗尽且不能自动重置。
            RuntimeError: 由底层训练过程（如前向/反向传播）可能抛出的异常。
        复杂度:
            时间复杂度 O(local_epochs × batch_cost)，空间复杂度 O(|model|)。

        费曼学习法:
            (A) 函数在本地数据上训练模型并返回统计指标。
            (B) 像运动员按计划完成多轮训练，再统计平均成绩与体能消耗。
            (C) 步骤拆解:
                - 逐一确认是否使用外部传入的模型、数据、优化器与损失函数，保持灵活性。
                - 将数据加载器转换为迭代器，确保在需要时可以连续取批次。
                - 切换模型到训练模式，启用 BatchNorm/Dropout 等训练行为。
                - 初始化准确率与损失缓存，用于计算平均表现。
                - 对每个本地轮次调用 `self.train` 完成训练并获取指标。
                - 将每轮指标存入列表，方便后续求平均。
                - 调用 `self.step` 更新调度逻辑（如学习率调整）。
                - 所有轮次完成后，触发学习率调度器的步进。
                - 使用 `avg_value` 计算平均指标并返回。
            (D) 示例:
                >>> acc, loss = client.local_training(local_epochs=2)
                >>> isinstance(acc, float) and isinstance(loss, float)
                True
            (E) 边界与测试:
                - 若 `local_epochs` 为 0，应确认函数返回合理的平均值。
                - 若训练迭代器耗尽，需保证 `self.train` 内部能够重置或抛出清晰异常。
                - 建议测试: 1) 使用固定随机种子验证返回值稳定性; 2) 模拟单步训练验证平均计算正确。
            (F) 背景参考:
                - 概念: 本地 SGD 训练与指标聚合。
                - 参考书籍: 《Deep Learning》关于优化与训练循环的章节。
        """
        # 根据传入参数决定是否替换默认模型与组件，提升灵活度
        model = self.new_if_given(model, self.model)
        train_loader = self.new_if_given(train_loader, self.train_loader)
        optimizer = self.new_if_given(optimizer, self.optimizer)
        criterion_fn = self.new_if_given(criterion_fn, self.criterion_fn)
        local_epochs = self.new_if_given(local_epochs, self.local_epochs)

        # 若提供的是 DataLoader，则转为迭代器以支持跨 epoch 连续采样
        train_iterator = iter(train_loader) if isinstance(
            train_loader, torch.utils.data.DataLoader) else train_loader
        # 切换模型为训练模式，启用梯度更新所需的行为
        model.train()
        acc_values, loss_values = [], []
        for epoch in range(local_epochs):
            # 执行单轮训练，返回该轮的准确率与损失
            acc, loss = self.train(model, train_iterator,
                                   optimizer, criterion_fn)
            # 缓存度量指标，便于后续取平均
            acc_values.append(acc)
            loss_values.append(loss)
            # 根据调度策略更新优化器状态（如学习率、动量等）
            self.step(optimizer, cur_local_epoch=epoch)
        # 完成本地迭代后推进学习率调度器
        self.lr_scheduler.step()

        return avg_value(acc_values), avg_value(loss_values)

    def fetch_updates(self, benign_flag=False):
        """
        生成提交给服务器的更新向量，并在需要时注入攻击者操作。

        参数:
            benign_flag (bool): 若为 True，强制以良性方式返回更新。
        返回:
            None: 更新结果保存在 `self.update` 属性中。
        异常:
            AttributeError: 当攻击者未实现 `non_omniscient` 方法却被调用时出现。
        复杂度:
            时间复杂度 O(|model|)，空间复杂度 O(|model|)。

        费曼学习法:
            (A) 函数准备客户端要上交的训练成果。
            (B) 像运动员提交成绩单，若怀有恶意可在交表前动手脚。
            (C) 步骤拆解:
                - 调用算法处理器生成标准更新（可能是梯度或模型参数）。
                - 若当前客户端是攻击者且允许篡改，则执行 `non_omniscient` 攻击。
                - 更新全局轮次计数，记录已完成一次通信。
            (D) 示例:
                >>> client.fetch_updates(benign_flag=True)
                >>> client.update is not None
                True
            (E) 边界与测试:
                - 若 `global_weights_vec` 尚未设置，算法处理器可能无法生成更新。
                - 若攻击者未实现 `non_omniscient` 却被标记为攻击者，会触发 AttributeError。
                - 建议测试: 1) 良性客户端生成更新并断言尺寸; 2) 恶意客户端在 `benign_flag=True` 时结果应与良性一致。
            (F) 背景参考:
                - 概念: 联邦参数上报与模型投毒。
                - 参考书籍: 《Federated Learning》以及模型投毒综述。
        """
        # 先生成标准更新，确保遵循所选算法的通信协议
        self.update = self.algorithm.get_local_update(
            global_weights_vec=self.global_weights_vec)
        if not benign_flag:
            # 恶意客户端可在上报前对更新施加篡改
            if self.category == "attacker" and "non_omniscient" in self.attributes:
                self.update = self.non_omniscient()

        # XXX???????????????????????????
        # cos_similarity = None
        # if self.prev_local_update is not None and self.update is not None:
        #     try:
        #         prev_vec = torch.as_tensor(
        #             self.prev_local_update, device=self.args.device).flatten().float()
        #         cur_vec = torch.as_tensor(
        #             self.update, device=self.args.device).flatten().float()
        #         prev_norm = torch.norm(prev_vec)
        #         cur_norm = torch.norm(cur_vec)
        #         if prev_norm > 0 and cur_norm > 0:
        #             cos_similarity = torch.dot(prev_vec, cur_vec) / (prev_norm * cur_norm)
        #             cos_similarity = float(cos_similarity.detach().cpu().item())
        #     except Exception:
        #         cos_similarity = None
        # if cos_similarity is not None:
        #     print(f"Client {self.worker_id} local cos: {cos_similarity:.6f}")
        # self.prev_local_update = (self.update.clone() if isinstance(self.update, torch.Tensor)
        #                           else torch.as_tensor(self.update).clone() if self.update is not None else None)

        # 成功准备更新后，推进全局通信轮次计数
        self.global_epoch += 1

    def client_test(self, model=None, test_dataset=None):
        """
        在本地测试集上评估模型性能，返回准确率与损失。

        参数:
            model (torch.nn.Module, 可选): 指定评估模型，默认使用当前模型。
            test_dataset (Dataset, 可选): 指定评估数据集，默认使用 `self.test_dataset`。
        返回:
            Tuple[float, float]: 测试准确率与测试损失。
        异常:
            ValueError: 当测试数据集缺失或无法被 DataLoader 处理时可能抛出。
        复杂度:
            时间复杂度 O(|test_dataset|)，空间复杂度 O(|model|)。

        费曼学习法:
            (A) 函数衡量模型在本地测试数据上的表现。
            (B) 像运动员参加模拟考试，检验训练成果。
            (C) 步骤拆解:
                - 若传入自定义模型或数据集，则按需替换默认对象。
                - 构建测试数据加载器，确保以评估模式迭代数据。
                - 调用 `self.test` 获取准确率与损失并返回。
            (D) 示例:
                >>> acc, loss = client.client_test()
                >>> isinstance(acc, float) and isinstance(loss, float)
                True
            (E) 边界与测试:
                - 若未提供测试数据集，`get_dataloader` 会报错，需在调用前确认。
                - 建议测试: 1) 使用小批量数据验证测试流程; 2) 模拟缺失数据集时捕获异常。
            (F) 背景参考:
                - 概念: 模型评估与性能度量。
                - 参考书籍: 《Pattern Recognition and Machine Learning》关于模型评估的章节。
        """
        model = self.new_if_given(model, self.model)
        test_dataset = self.new_if_given(test_dataset, self.test_dataset)
        test_loader = self.get_dataloader(test_dataset, train_flag=False)
        test_acc, test_loss = self.test(model, test_loader)
        return test_acc, test_loss


# __AI_ANNOTATION_SUMMARY__
# - Client.__init__: 初始化模型、优化器与数据加载器，搭建本地训练环境。
# - Client.record_time: 为关键方法添加计时装饰器，辅助分析性能开销。
# - Client.set_algorithm: 绑定指定联邦算法处理器并设定本地训练轮次。
# - Client.load_global_model: 将服务器下发的全局权重同步到本地模型。
# - Client.local_training: 在本地数据上迭代训练并输出平均准确率与损失。
# - Client.fetch_updates: 生成并可选篡改上报服务器的模型更新。
# - Client.client_test: 使用测试集评估模型表现并返回准确率与损失。
