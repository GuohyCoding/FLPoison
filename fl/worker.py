"""
Worker 基类封装了客户端与服务器共享的训练、评估与优化器构建逻辑。

核心功能:
    - 构建数据加载器并支持训练阶段的无限循环；
    - 定义统一的损失函数、训练/评估流程；
    - 提供优化器与学习率调度器的工厂方法。
"""

import torch


class Worker:
    """
    联邦学习中客户端与服务器的共同父类，提供训练与评估的通用接口。

    属性:
        args (argparse.Namespace): 全局配置，包含设备、优化器等训练参数。
        worker_id (int): 工作节点标识，客户端为正整数，服务器固定为 -1。
        synthesizer (Any): 预留的合成器接口，子类可重写用于数据/梯度合成。
    """

    def __init__(self, args, worker_id):
        """
        初始化 Worker 基类，记录运行配置与节点标识。

        参数:
            args (argparse.Namespace): 包含数据、优化器、设备等通用配置。
            worker_id (int): 节点编号，客户端应为正整数，服务器为 -1。
        返回:
            None: 构造函数仅做属性初始化。
        异常:
            ValueError: 当 worker_id 不符合约定范围时，子类可选择抛出。
        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)。

        费曼学习法:
            (A) 函数记录节点的配置与身份，为后续操作打基础。
            (B) 类比登记运动员信息：先记下姓名号码，方便后续调度。
            (C) 步骤拆解:
                - 保存运行配置 `args` 以备成员方法使用。
                - 记录 `worker_id` 确定当前节点身份。
                - 预留 `synthesizer` 属性给子类扩展。
            (D) 示例:
                >>> worker = Worker(args, worker_id=1)
                >>> worker.worker_id
                1
            (E) 边界与测试:
                - 若 `worker_id` 为 None，后续日志中无法区分节点，应检查配置。
                - 建议测试: 1) 初始化客户端与服务器并确认属性正确; 2) 验证 `synthesizer` 默认值为 None。
            (F) 背景参考:
                - 概念: 联邦系统中客户端/服务器身份管理。
                - 参考书籍: 《Federated Learning》对节点抽象的介绍。
        """
        self.args = args
        self.worker_id = worker_id
        self.synthesizer = None  # 子类可根据任务需要自定义

    def __str__(self):
        """
        返回节点的字符串表示，便于日志记录。

        返回:
            str: 包含 worker_id 的可读字符串。
        复杂度:
            时间复杂度 O(1)。
        """
        return f"worker id: {self.worker_id}"

    def get_dataloader(self, dataset, train_flag=True, **kwargs):
        """
        构建数据迭代器，训练时默认无限循环，测试时单次遍历。

        参数:
            dataset (Dataset): 输入数据集，需实现 __len__ 和 __getitem__。
            train_flag (bool): True 表示训练模式（打乱并无限循环），False 为测试模式。
            **kwargs: 预留扩展参数，目前未使用。
        返回:
            Iterator[Tuple[Tensor, Tensor]]: 依次产出图像与标签的迭代器。
        异常:
            RuntimeError: 当 DataLoader 构建失败或数据格式不符时。
        复杂度:
            单次迭代时间复杂度 O(batch_size)，总体随数据量线性增长。

        费曼学习法:
            (A) 函数根据模式返回批量数据迭代器，训练模式支持无限循环。
            (B) 类比操场上训练：训练时不停绕圈跑，测试时只跑一圈计成绩。
            (C) 步骤拆解:
                - 使用 PyTorch DataLoader 按批次切分原始数据。
                - 若是训练模式，进入无限循环以避免 DataLoader 在长训练中耗尽。
                - 若是测试模式，按顺序遍历一次数据集即可。
            (D) 示例:
                >>> train_iter = worker.get_dataloader(train_ds, train_flag=True)
                >>> images, targets = next(train_iter)
            (E) 边界与测试:
                - 若 dataset 为空，迭代器将立即抛出 StopIteration，应提前校验。
                - 建议测试: 1) 使用小批量数据验证无限循环行为; 2) 对测试模式确保无 shuffle。
            (F) 背景参考:
                - 概念: DataLoader 与迭代器。
                - 参考书籍: 《Deep Learning with PyTorch》相关章节。
        """
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=train_flag,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        if train_flag:
            # 训练阶段通过无限循环提供数据，避免单 epoch 后迭代器耗尽
            while True:
                for images, targets in dataloader:
                    yield images, targets
        else:
            # 测试阶段保持单次遍历，确保指标统计可控
            for images, targets in dataloader:
                yield images, targets

    def criterion_fn(self, y_pred, y_true, **kwargs):
        """
        默认交叉熵损失函数，适用于多分类任务。

        参数:
            y_pred (Tensor): 模型输出的类别概率或 logits。
            y_true (Tensor): 真实标签。
            **kwargs: 预留给子类扩展（如自定义权重）。
        返回:
            Tensor: 标量损失值。
        异常:
            ValueError: 当预测与标签形状不匹配时由 PyTorch 抛出。
        复杂度:
            时间复杂度 O(batch_size × num_classes)。

        费曼学习法:
            (A) 函数计算预测与真实标签之间的交叉熵损失。
            (B) 类比考试评分，交叉熵像按概率扣分：越自信错答扣得越多。
            (C) 步骤拆解:
                - 直接调用 `torch.nn.CrossEntropyLoss` 完成计算。
            (D) 示例:
                >>> loss = worker.criterion_fn(pred_logits, labels)
            (E) 边界与测试:
                - 仅支持分类任务，若是回归需在子类重写。
                - 建议测试: 1) 验证正确标签输入返回非负值; 2) 检查不同批量大小下行为一致。
            (F) 背景参考:
                - 概念: 交叉熵损失。
                - 参考书籍: 《Pattern Recognition and Machine Learning》。
        """
        return torch.nn.CrossEntropyLoss()(y_pred, y_true)

    def new_if_given(self, value, default):
        """
        若外部传入值为 None，则回退到默认值。

        参数:
            value (Any): 外部传入的可选参数。
            default (Any): 默认使用的对象。
        返回:
            Any: 传入值或默认值。
        复杂度:
            时间复杂度 O(1)。

        费曼学习法:
            (A) 函数在自定义值缺失时返回默认配置。
            (B) 类比点餐时，如果没特别说明，就按店家默认口味。
            (C) 步骤拆解:
                - 判断 `value` 是否为 None。
                - 若为 None 返回默认值，否则返回自定义值。
            (D) 示例:
                >>> worker.new_if_given(None, 10)
                10
            (E) 边界与测试:
                - 若默认值本身为可变对象，返回时需确保不会被外部修改破坏。
                - 建议测试: 1) 传入不同类型值验证逻辑; 2) 针对可变对象检查引用问题。
            (F) 背景参考:
                - 概念: 参数默认值处理。
        """
        return default if value is None else value

    def train(self, model, train_iterator, optimizer, criterion_fn=None):
        """
        执行单个训练步骤，返回本批次准确率与损失。

        参数:
            model (torch.nn.Module): 待训练的模型。
            train_iterator (Iterator): 产出 (images, targets) 的迭代器。
            optimizer (torch.optim.Optimizer): 优化器实例。
            criterion_fn (Callable, 可选): 损失函数，缺省为 `self.criterion_fn`。
        返回:
            Tuple[float, float]: 当前批次的准确率与平均损失。
        异常:
            StopIteration: 当迭代器无数据时触发，应在外部确保迭代器可用。
            RuntimeError: 前向或反向传播异常时由 PyTorch 抛出。
        复杂度:
            时间复杂度 O(batch_size × model_cost)，空间复杂度 O(|model|)。

        费曼学习法:
            (A) 函数完成一次前向、反向传播并返回性能指标。
            (B) 类比运动员完成一次训练回合，并记录成绩与体力消耗。
            (C) 步骤拆解:
                - 若未指定损失函数，回退到默认交叉熵。
                - 清空优化器梯度，避免累积影响。
                - 从迭代器取出一个批次并移动到目标设备。
                - 前向传播得到预测概率。
                - 计算损失并反向传播累积梯度。
                - 统计预测准确数目与平均损失。
            (D) 示例:
                >>> acc, loss = worker.train(model, train_iter, optimizer)
            (E) 边界与测试:
                - 若模型未调用 `optimizer.step()`，梯度不会被应用，需配合 `step` 方法使用。
                - 建议测试: 1) 使用固定随机种子验证返回值稳定； 2) 检查不同损失函数输入的兼容性。
            (F) 背景参考:
                - 概念: 前向与反向传播。
                - 参考书籍: 《Deep Learning》。
        """
        criterion_fn = self.new_if_given(criterion_fn, self.criterion_fn)
        optimizer.zero_grad()
        # 从训练迭代器获取一个批次并迁移到目标设备
        images, targets = next(train_iterator)
        images, targets = images.to(
            self.args.device), targets.to(self.args.device)
        # 前向传播获取预测概率或 logits
        pred_probs = model(images)
        loss = criterion_fn(pred_probs, targets)
        # 反向传播累积梯度
        loss.backward()
        # 统计预测结果，计算准确率与平均损失
        predicted = torch.argmax(pred_probs.data, 1)
        train_acc = (predicted == targets).sum().item()
        train_loss = loss.item()
        train_loss /= len(images)
        train_acc /= len(images)
        return train_acc, train_loss

    def step(self, optimizer, **kwargs):
        """
        执行优化器的参数更新步骤。

        参数:
            optimizer (torch.optim.Optimizer): 需要更新的优化器。
            **kwargs: 预留扩展参数（例如自适应学习率策略）。
        返回:
            None: 直接调用 `optimizer.step()`。
        异常:
            RuntimeError: 若前向/反向传播未正确执行，步进时可能报错。
        复杂度:
            时间复杂度 O(|model|)。

        费曼学习法:
            (A) 函数实际应用梯度，更新模型参数。
            (B) 类比训练后根据表现调整动作细节。
            (C) 步骤拆解:
                - 调用优化器的 `step` 方法，让参数沿梯度方向前进。
            (D) 示例:
                >>> worker.step(optimizer)
            (E) 边界与测试:
                - 若未调用 `loss.backward()`，参数不会更新，应检测梯度是否存在。
                - 建议测试: 1) 对固定梯度输入验证参数变化； 2) 与不同优化器组合使用。
            (F) 背景参考:
                - 概念: 优化器步进。
        """
        optimizer.step()

    def test(self, model, test_loader, imbalanced=False):
        """
        在测试集上评估模型，支持类不平衡场景的尾类准确率统计。

        参数:
            model (torch.nn.Module): 待评估模型。
            test_loader (Iterator): 测试数据迭代器。
            imbalanced (bool): 是否统计类不平衡场景的尾类准确率。
        返回:
            Tuple[float, float] 或 Tuple[float, float, float]:
                - 常规模式返回 (整体准确率, 平均损失)；
                - 不平衡模式返回 (整体准确率, 尾类准确率, 平均损失)。
        异常:
            ZeroDivisionError: 若测试集中无样本或尾类样本为空时可能出现。
        复杂度:
            时间复杂度 O(|test_dataset|)，空间复杂度 O(1)。

        费曼学习法:
            (A) 函数测量模型在测试集上的准确率和损失。
            (B) 类比给运动员期末考，全面评估掌握情况，并特别关注薄弱项目。
            (C) 步骤拆解:
                - 切换模型至评估模式，关闭 Dropout 等训练特性。
                - 遍历测试数据，累计预测正确样本数与损失。
                - 若处理类不平衡，额外统计尾类样本的准确率。
                - 计算整体指标并返回。
            (D) 示例:
                >>> acc, loss = worker.test(model, test_loader)
            (E) 边界与测试:
                - 若 `test_loader` 为空，需提前检查数据准备。
                - 建议测试: 1) 对平衡/不平衡数据分别验证输出维度; 2) 使用已知模型验证准确率计算。
            (F) 背景参考:
                - 概念: 模型评估与类不平衡问题。
                - 参考书籍: 《Imbalanced Learning》。
        """
        model.eval()
        tail_cls_from = self.args.tail_cls_from if imbalanced else 0
        overall_correct, test_loss, num_samples = 0, 0, 0
        rest_correct, rest_samples = 0, 0

        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(
                    self.args.device), targets.to(self.args.device)
                pred_probs = model(images)
                loss = self.criterion_fn(pred_probs, targets)
                predicted = torch.argmax(pred_probs.data, 1)
                num_samples += len(targets)
                overall_correct += (predicted == targets).sum().item()
                test_loss += loss.item()

                if imbalanced:
                    # 仅统计尾类（>= tail_cls_from）的准确率
                    rest_mask = targets >= tail_cls_from
                    rest_correct += (predicted[rest_mask]
                                     == targets[rest_mask]).sum().item()
                    rest_samples += rest_mask.sum().item()

        overall_accuracy = overall_correct / num_samples if num_samples > 0 else 0
        test_loss /= num_samples if num_samples > 0 else 1

        if imbalanced:
            rest_accuracy = rest_correct / rest_samples if rest_samples > 0 else 0
            return overall_accuracy, rest_accuracy, test_loss

        return overall_accuracy, test_loss

    def get_optimizer_scheduler(self, model, learning_rate=None, weight_decay=None):
        """
        构建优化器与学习率调度器，支持多种常见配置。

        参数:
            model (torch.nn.Module): 待优化模型。
            learning_rate (float, 可选): 学习率，默认取自 `self.args.learning_rate`。
            weight_decay (float, 可选): 权重衰减系数，默认取自 `self.args.weight_decay`。
        返回:
            Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
                优化器与对应的学习率调度器。
        异常:
            NotImplementedError: 当指定的学习率调度器尚未实现时抛出。
        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)（初始化阶段忽略模型大小差异）。

        费曼学习法:
            (A) 函数根据配置创建优化器及其学习率调度策略。
            (B) 类比为运动员配备教练与训练节奏表。
            (C) 步骤拆解:
                - 若外部未指定，使用默认学习率与权重衰减。
                - 根据优化器类型（SGD/Adam）构造实例。
                - 根据 `lr_scheduler` 名称创建相应调度器，若缺省则使用恒定学习率。
            (D) 示例:
                >>> optimizer, scheduler = worker.get_optimizer_scheduler(model)
            (E) 边界与测试:
                - 若 `args.optimizer` 为未支持类型，将跳过设置，需在配置前验证。
                - 建议测试: 1) 对不同调度器配置验证学习率变化; 2) 检查 FedOpt/FedAvg 等算法下 total_epoch 计算是否正确。
            (F) 背景参考:
                - 概念: 优化器与学习率调度策略。
                - 参考书籍: 《Deep Learning》优化章节。
        """
        learning_rate = self.new_if_given(
            learning_rate, self.args.learning_rate)
        weight_decay = self.new_if_given(self.args.weight_decay, weight_decay)
        if self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=learning_rate, momentum=self.args.momentum, weight_decay=weight_decay)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise NotImplementedError(f"{self.args.optimizer} is not implemented currently.")
        if hasattr(self.args, 'lr_scheduler') and self.args.lr_scheduler is not None:
            if self.args.lr_scheduler == 'MultiStepLR':
                milestones = [int(i*self.args.epochs) if i <
                              1 else i for i in self.args.milestones]
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=milestones, gamma=0.1)
            elif self.args.lr_scheduler == 'StepLR':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)
            elif self.args.lr_scheduler == 'ExponentialLR':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.9)
            elif self.args.lr_scheduler == "CosineAnnealingLR":
                if self.args.algorithm == "FedSGD":
                    total_epoch = self.args.epochs
                elif self.args.algorithm in ["FedOpt", "FedAvg"]:
                    total_epoch = self.args.epochs * self.args.local_epochs
                else:
                    total_epoch = self.args.epochs
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch)
            else:
                raise NotImplementedError(f"{self.args.lr_scheduler} is not implemented currently.")
        else:
            # 若未设置调度器，则保持学习率常数
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 1.0)

        return optimizer, lr_scheduler

    def cycle(self, dataloader):
        """
        将有限迭代器转为无限循环迭代器，适用于训练阶段。

        参数:
            dataloader (Iterable): 可迭代产生批量数据的对象。
        返回:
            Iterator: 无限循环的批量数据迭代器。
        复杂度:
            单次迭代 O(batch_size)。

        费曼学习法:
            (A) 函数让一个只能跑一圈的迭代器变成永不停歇的跑道。
            (B) 类比把有终点的跑道接成循环赛道，选手可以一直训练。
            (C) 步骤拆解:
                - 外部提供的迭代器遍历完一轮后重新开始。
            (D) 示例:
                >>> endless_loader = worker.cycle(dataloader)
                >>> next(endless_loader)
            (E) 边界与测试:
                - 若原始迭代器内部已实现无限循环，会导致嵌套循环，应避免重复包装。
                - 建议测试: 1) 对只迭代一轮的数据验证循环行为; 2) 结合训练流程确保不会产生多层嵌套迭代。
            (F) 背景参考:
                - 概念: 数据迭代重复机制。
        """
        while True:
            for images, targets in dataloader:
                yield images, targets


# __AI_ANNOTATION_SUMMARY__
# - Worker.__init__: 初始化运行配置与节点标识，为子类提供基础属性。
# - Worker.get_dataloader: 构建训练/测试数据迭代器，训练模式支持无限循环。
# - Worker.criterion_fn: 默认交叉熵损失，适用于分类任务。
# - Worker.new_if_given: 提供可选参数的默认回退逻辑。
# - Worker.train: 执行单步训练并返回准确率与损失。
# - Worker.step: 调用优化器步进函数应用梯度更新。
# - Worker.test: 在测试集上评估模型表现，支持尾类准确率统计。
# - Worker.get_optimizer_scheduler: 创建优化器与学习率调度器组合。
# - Worker.cycle: 将有限迭代器包装为无限循环迭代器。
