# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from datapreprocessor.cinic10 import CINIC10
from datapreprocessor.chmnist import CHMNIST
from plot_utils import plot_label_distribution
from datapreprocessor.tinyimagenet import TinyImageNet


def load_data(args):
    """根据配置加载指定数据集，并返回训练/测试 `Dataset` 对象。

    概述:
        根据 `args.dataset` 选择对应的数据集（官方或自定义），应用 `get_transform` 返回的变换，
        自动处理 CIFAR 系列 `targets` 的列表转张量问题。

    参数:
        args (argparse.Namespace): 配置对象，须包含 `dataset`、`mean`、`std` 等字段。

    返回:
        Tuple[Dataset, Dataset]: `(train_dataset, test_dataset)`。

    异常:
        ValueError: 当数据集名称未实现时抛出。

    复杂度:
        时间复杂度主要取决于数据集初始化过程（O(n)）；空间复杂度取决于底层数据加载实现。

    费曼学习法:
        (A) 函数根据配置选择并实例化训练/测试数据集。
        (B) 类比餐厅后厨根据菜单点菜：先看菜名，再取对应食材与做法。
        (C) 步骤拆解:
            1. 调用 `get_transform` 获取训练与测试变换。
            2. 按数据集名称分支，分别实例化官方或自定义数据集。
            3. 将 `targets` 列表转换为张量以便后续操作。
            4. 返回训练与测试数据集。
        (D) 示例:
            >>> train_ds, test_ds = load_data(args)
            >>> len(train_ds), len(test_ds)
        (E) 边界条件与测试建议: 需确保 `args.dataset` 在实现列表中；建议测试不同数据集名称是否正常加载。
        (F) 背景参考: PyTorch `torchvision.datasets` 的通用使用方式与自定义数据集接口。
    """
    # load dataset
    trans, test_trans = get_transform(args)
    data_directory = './data'
    if args.dataset == "EMNIST":
        train_dataset = datasets.EMNIST(data_directory, split="digits", train=True, download=True,
                                        transform=trans)
        test_dataset = datasets.EMNIST(
            data_directory, split="digits", train=False, transform=test_trans)
    elif args.dataset in ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]:
        # 使用 eval 以便通用调用 torchvision 官方数据集构造函数。
        train_dataset = eval(f"datasets.{args.dataset}")(root=data_directory, train=True,
                                                         download=True, transform=trans)
        test_dataset = eval(f"datasets.{args.dataset}")(root=data_directory, train=False,
                                                        download=True, transform=test_trans)
    elif args.dataset in ["CHMNIST", "CINIC10", "TinyImageNet"]:
        """
        dataset in custom datasets, such as CHMNIST, CINIC10, TinyImageNet
        """
        train_dataset = eval(args.dataset)(root=data_directory, train=True, download=True,
                                transform=trans)
        test_dataset = eval(args.dataset)(root=data_directory, train=False, download=True,
                               transform=test_trans)
    else:
        raise ValueError("Dataset not implemented yet")

    # deal with CIFAR10 list-type targets. CIFAR10 data is numpy array defaultly.
    train_dataset.targets = list_to_tensor(train_dataset.targets)
    test_dataset.targets = list_to_tensor(test_dataset.targets)
    return train_dataset, test_dataset


def list_to_tensor(vector):
    """将标签或索引列表转换为张量，保持兼容性。

    概述:
        部分数据集（如 CIFAR-10）默认返回列表形式标签，本函数统一转换为 PyTorch 张量。

    参数:
        vector (Union[list, Tensor]): 待处理的标签或索引集合。

    返回:
        Tensor: 若输入为列表则转换为张量，否则原样返回。

    费曼学习法:
        (A) 函数检查输入是否为列表，若是则转成张量。
        (B) 类比将纸质名单录入电脑，方便后续计算。
        (C) 步骤拆解:
            1. 判断输入是否为 `list`。
            2. 若是，则调用 `torch.tensor` 转换。
            3. 返回张量或原始对象。
        (D) 示例:
            >>> list_to_tensor([1, 2, 3])
            tensor([1, 2, 3])
        (E) 边界条件与测试建议: 输入为张量时应保持不变；建议测试不同数据类型。
        (F) 背景参考: PyTorch 对张量与 Python 列表的互操作。
    """
    if isinstance(vector, list):
        vector = torch.tensor(vector)
    return vector


def subset_by_idx(args, dataset, indices, train=True):
    """根据索引选取数据子集，并应用训练/测试对应的变换。

    概述:
        使用 `Partition` 包装器构建子数据集，常用于构造客户端私有数据切片或投毒样本集合。

    参数:
        args (argparse.Namespace): 配置对象，用于获取变换函数。
        dataset (Dataset): 原始数据集对象。
        indices (Sequence[int]): 需要选取的样本索引列表或张量。
        train (bool): 是否使用训练阶段的变换。

    返回:
        Partition: 包含所选样本和变换的子数据集对象。

    费曼学习法:
        (A) 函数选取给定索引对应的数据，并施加适当的预处理。
        (B) 类比图书管理员根据清单从书库中挑选指定图书，并贴上不同用途的封条。
        (C) 步骤拆解:
            1. 根据 `train` 标志调用 `get_transform` 获取对应变换。
            2. 使用 `Partition` 包装原始数据集和索引。
            3. 返回新的子数据集供客户端使用。
        (D) 示例:
            >>> subset = subset_by_idx(args, dataset, indices=[0,1,2], train=True)
            >>> len(subset)
            3
        (E) 边界条件与测试建议: 需确保索引合法；建议测试训练/测试模式下变换是否切换。
        (F) 背景参考: 联邦学习中客户端数据划分策略。
    """
    trans = get_transform(args)[0] if train else get_transform(args)[1]
    dataset = Partition(
        dataset, indices, transform=trans)
    return dataset


def get_transform(args):
    """为不同数据集与模型组合返回训练与测试的图像变换流水线。

    概述:
        根据数据集名称和模型类型配置尺寸、归一化与数据增强策略，并更新 `args.num_dims`。

    参数:
        args (argparse.Namespace): 配置对象，应包含 `dataset`、`model`、`mean`、`std` 等字段。

    返回:
        Tuple[transforms.Compose, transforms.Compose]: `(train_transform, test_transform)`。

    异常:
        ValueError: 当数据集名称未实现时抛出。

    复杂度:
        常数时间，用于构造 `Compose` 对象。

    费曼学习法:
        (A) 函数根据数据集/模型类型返回恰当的数据增强与归一化方案。
        (B) 类比裁缝在量身前先决定裁剪尺寸与布料处理方式。
        (C) 步骤拆解:
            1. 判断数据集与模型组合，配置 Resize、ToTensor、Normalize 等步骤。
            2. 对 CIFAR/Tiny ImageNet 设置 `args.num_dims` 并可按需添加数据增强。
            3. 返回训练与测试阶段的变换。
        (D) 示例:
            >>> train_tf, test_tf = get_transform(args)
        (E) 边界条件与测试建议: 需确保 `args` 包含所需属性；测试不同组合是否抛出异常或返回预期变换。
        (F) 背景参考: 计算机视觉中常见的数据增强策略、模型输入尺寸要求。
    """
    if args.dataset in ["MNIST", "FashionMNIST", "EMNIST", "FEMNIST"] and args.model in ['lenet', "lr"]:
        # resize MNIST to 32x32 for LeNet5
        train_tran = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(args.mean, args.std)])
        test_trans = train_tran
        # define the image dimensions for self.args, so that others can use it, such as DeepSight, lr model
        args.num_dims = 32
    elif args.dataset in ["CINIC10"]:
        train_tran = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=args.mean, std=args.std)])
        test_trans = train_tran
    elif args.dataset in ["CIFAR10", "CIFAR100", "TinyImageNet"]:
        args.num_dims = 32 if args.dataset in ['CIFAR10', 'CIFAR100'] else 64
        # data augmentation
        train_tran = transforms.Compose([
            # transforms.RandomCrop(args.num_dims, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(args.mean, args.std)
        ])
        test_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(args.mean, args.std)
        ])
    else:
        raise ValueError("Dataset not implemented yet")

    return train_tran, test_trans


def split_dataset(args, train_dataset, test_dataset):
    """按照指定分布策略划分训练数据并返回客户端索引列表。

    概述:
        支持 IID、类不均衡 IID、Dirichlet 非 IID 等多种划分方式，同时提供缓存以避免重复计算。

    参数:
        args (argparse.Namespace): 配置对象，包含 `distribution`、`num_clients`、`cache_partition` 等。
        train_dataset (Dataset): 完整训练数据集。
        test_dataset (Dataset): 测试数据集（可按需进一步划分）。

    返回:
        Tuple[List[Tensor], Dataset]: `(client_indices, test_dataset)`。

    异常:
        无显式异常，错误由内部函数抛出。

    复杂度:
        取决于划分方法，IID 大致 O(n)，Dirichlet 需遍历类别 O(K + n)。

    费曼学习法:
        (A) 函数生成每个客户端拥有的样本索引，可选缓存以节省时间。
        (B) 类比食堂按人群口味分配菜品，事先记下分配方案供下次使用。
        (C) 步骤拆解:
            1. 判断是否启用缓存；若有缓存直接加载并返回。
            2. 根据 `distribution` 选择 IID 或 Dirichlet 等划分策略。
            3. 对类不均衡情况，进一步调用 `class_imbalanced_partition` 调整样本数量。
            4. 如启用缓存，则将索引写入缓存文件。
            5. 返回客户端索引与测试集（当前保持原状）。
        (D) 示例:
            >>> client_indices, test_ds = split_dataset(args, train_ds, test_ds)
        (E) 边界条件与测试建议: 需确保 `args.num_clients > 0`；测试不同 `distribution` 与缓存组合。
        (F) 背景参考: 联邦学习常见划分策略、Dirichlet 非 IID 模型。
    """
    # agrs.cache_partition: True, False, non-iid, iid, class-imbalanced-iid
    cache_flag = (args.cache_partition ==
                  True or args.cache_partition == args.distribution)
    if cache_flag:
        # ready for cache usage
        # check if the indices are already generated in running_caches folder
        cache_exist, file_path = check_partition_cache(args)
        if cache_exist:
            args.logger.info("Target indices caches to save time")
            with open(file_path, 'rb') as f:
                client_indices = pickle.load(f)
            return client_indices, test_dataset

    args.logger.info("Generating new indices")
    if args.distribution in ['iid', 'class-imbalanced_iid']:
        client_indices = iid_partition(args, train_dataset)
        args.logger.info("Doing iid partition")
        if "class-imbalanced" in args.distribution:
            args.logger.info("Doing class-imbalanced iid partition")
            # class-imbalanced iid partition
            for i in range(args.num_clients):
                class_indices = client_class_indices(
                    client_indices[i], train_dataset)
                client_indices[i] = class_imbalanced_partition(
                    class_indices, args.im_iid_gamma)
    elif args.distribution in ['non-iid']:
        # dirichlet partition
        args.logger.info("Doing non-iid partition")
        client_indices = dirichlet_split_noniid(
            train_dataset.targets, args.dirichlet_alpha, args.num_clients)
        args.logger.info(f"dirichlet alpha: {args.dirichlet_alpha}")
    if cache_flag:
        save_partition_cache(client_indices, file_path)
    args.logger.info(f"{args.distribution} partition finished")
    # plot_label_distribution(train_dataset, client_indices, args.num_clients, args.dataset, args.distribution)
    return client_indices, test_dataset


def save_partition_cache(client_indices, file_path):
    """将客户端数据划分结果缓存到本地文件。

    概述:
        通过 pickle 序列化索引列表，避免重复划分开销。

    参数:
        client_indices (List[Tensor]): 每个客户端对应的样本索引集合。
        file_path (str): 缓存文件路径。

    费曼学习法:
        (A) 函数把已经分好的“样本清单”存到磁盘。
        (B) 类比将分好的任务分配表复印并存档，省得下次重写。
        (C) 步骤拆解:
            1. 以写入模式打开目标文件。
            2. 使用 `pickle.dump` 持久化索引列表。
        (D) 示例:
            >>> save_partition_cache(indices, 'cache.pkl')
        (E) 边界条件与测试建议: 需确保目录存在且可写；可测试文件写入后能否成功读取。
        (F) 背景参考: Python 序列化与缓存机制。
    """
    with open(file_path, 'wb') as f:
        pickle.dump(client_indices, f)


def check_partition_cache(args):
    """检查划分缓存是否存在，并返回其路径。

    概述:
        若缓存目录不存在则创建；构造基于数据集名称、分布与客户端数量的缓存文件名。

    参数:
        args (argparse.Namespace): 配置对象。

    返回:
        Tuple[bool, str]: `(cache_exist, file_path)`。

    费曼学习法:
        (A) 函数判断“之前是否已经存好了任务分配表”，并返回文件路径。
        (B) 类比秘书检查档案柜中是否已有对应档案。
        (C) 步骤拆解:
            1. 构造缓存目录与文件名。
            2. 若目录不存在则创建。
            3. 判断缓存文件是否存在并返回结果。
        (D) 示例:
            >>> exists, path = check_partition_cache(args)
        (E) 边界条件与测试建议: 需具备写权限；测试不同分布配置下的文件命名是否区分。
        (F) 背景参考: 文件系统管理与缓存命名策略。
    """
    cache_exist = None
    folder_path = 'running_caches'
    file_name = f'{args.dataset}_balanced_{args.distribution}_{args.num_clients}_indices'
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        cache_exist = True if os.path.exists(file_path) else False
    return cache_exist, file_path


def check_noniid_labels(args, train_dataset, client_indices):
    """统计非 IID 划分下各客户端的标签集合与交集，供调试参考。

    参数:
        args (argparse.Namespace): 配置对象（需包含 `logger`）。
        train_dataset (Dataset): 完整训练数据集。
        client_indices (List[Tensor]): 各客户端的样本索引列表。

    费曼学习法:
        (A) 函数汇总每个客户端拥有的标签种类，并计算所有客户端共有的标签。
        (B) 类比学校了解每个班级有哪些社团成员，以及哪些活动是所有班级都参加的。
        (C) 步骤拆解:
            1. 遍历客户端索引，抽取对应标签集合。
            2. 记录每个客户端的唯一标签，并逐步求交集。
            3. 将结果写入日志，便于排查是否出现标签缺失。
        (D) 示例:
            >>> check_noniid_labels(args, train_ds, client_indices)
        (E) 边界条件与测试建议: 需确保 `train_dataset.targets` 可索引；建议对非 IID 划分结果进行可视化验证。
        (F) 背景参考: 非 IID 数据划分分析方法。
    """
    client_unique_labels = {}
    common_labels = None
    for client_id, indices in enumerate(client_indices):
        # get the labels of the corresponding indices
        labels = train_dataset.targets[indices]
        # get the unique labels of the client
        unique_labels = set(labels.tolist())
        client_unique_labels[client_id] = unique_labels
        # for the first client, initialize common_labels as the unique labels
        if common_labels is None:
            common_labels = unique_labels
        else:
            # update common_labels by taking the intersection of the unique labels
            common_labels = common_labels.intersection(unique_labels)

    # log the unique labels of each client and the common labels across all clients
    args.logger.info(
        f"Common unique labels across all clients: {common_labels}")
    for client_id, unique_labels in client_unique_labels.items():
        args.logger.info(
            f"Client {client_id} has unique labels: {unique_labels}")


class Partition(Dataset):
    """数据分片包装器：根据索引从原始数据集中取样，并可选注入后门。

    属性:
        dataset (Dataset): 原始数据集引用。
        classes (Sequence): 类别名称列表，与原始数据集保持一致。
        indices (Sequence[int]): 当前子数据集包含的样本索引。
        data (ndarray/Tensor): 被选中样本的底层数据。
        targets (Tensor): 对应的标签张量。
        mode (str): 图像模式（'L' 或 'RGB'），用于 PIL 转换。
        transform (Callable): 图像预处理变换。
        poison (bool): 是否启用后门注入流程。
        synthesizer (object): 数据投毒合成器，当 `poison=True` 时使用。
    """

    def __init__(self, dataset, indices=None, transform=None):
        """初始化分片数据集，提取指定索引的样本。

        参数:
            dataset (Dataset): 原始数据集。
            indices (Sequence[int]): 需要提取的样本索引；若为 None 则包含全部。
            transform (Callable): 待应用的图像变换。

        费曼学习法:
            (A) 构造函数像是把原数据集“切片”成一个子仓库。
            (B) 类比图书馆将部分图书借给分馆，并记录书目与分类。
            (C) 步骤拆解:
                1. 记录原始数据集及类别信息。
                2. 根据索引切出数据与标签。
                3. 推断图像模式（灰度或彩色）以保证 PIL 转换正确。
                4. 存储变换函数，并默认未投毒。
        """
        self.dataset = dataset
        self.classes = dataset.classes
        self.indices = indices if indices is not None else range(len(dataset))
        self.data, self.targets = dataset.data[self.indices], dataset.targets[self.indices]
        # (N, C, H, W) or (N, H, W) for MNIST-like grey images, mode='L'; CIFAR10-like color images, mode='RGB'
        self.mode = 'L' if len(self.data.shape) == 3 else 'RGB'
        self.transform = transform
        self.poison = False

    def __len__(self):
        """返回当前分片中样本数量。"""
        return len(self.data)

    def __getitem__(self, idx):
        """按索引返回（可选后门注入后的）图像与标签。

        参数:
            idx (int): 样本索引。

        返回:
            Tuple[Tensor, int]: `(image, target)`。

        费曼学习法:
            (A) 取出第 `idx` 个样本，按需转换与注入后门。
            (B) 类比从分馆书架取书，若标记为“特殊用途”则先处理后再借出。
            (C) 步骤拆解:
                1. 从缓存数组取得图像与标签。
                2. 将图像转换成 PIL（若 transform 要求）。
                3. 应用图像变换。
                4. 若启用后门，则调用合成器修改图像/标签。
                5. 返回处理后的数据。
        """
        image, target = self.data[idx], self.targets[idx]

        # doing this so that it is consistent with all other datasets
        # convert image to numpy array. for MNIST-like dataset, image is torch tensor, for CIFAR10-like dataset, image type is numpy array.
        if not isinstance(image, (np.ndarray, np.generic)):
            image = image.numpy()
        # to return a PIL Image
        image = Image.fromarray(image, mode=self.mode)
        if self.transform:
            image = self.transform(image)

        if self.poison:
            image, target = self.synthesizer.backdoor_batch(
                image, target.reshape(-1, 1))
        return image, target.squeeze()

    def poison_setup(self, synthesizer):
        """启用后门模式并注册数据合成器。"""
        self.poison = True
        self.synthesizer = synthesizer


def iid_partition(args, train_dataset):
    """执行近乎均衡且类别均衡的 IID 划分。

    参数:
        args (argparse.Namespace): 需包含 `num_clients` 与随机种子等设置。
        train_dataset (Dataset): 训练数据集。

    返回:
        List[Tensor]: 每个客户端的样本索引张量。

    费曼学习法:
        (A) 将每个类别的样本均匀分配给所有客户端。
        (B) 类比老师将不同难度的题平均分给所有学生。
        (C) 步骤拆解:
            1. 遍历每个类别，随机打乱该类样本索引。
            2. 计算每个客户端应分配的样本数量。
            3. 均匀地将样本分给各客户端，并处理余数。
            4. 将结果转换为张量列表返回。
        (D) 示例:
            >>> indices = iid_partition(args, train_ds)
            >>> len(indices)
            args.num_clients
        (E) 边界条件与测试建议: 需确保每个类别样本数 >= 客户端数量；建议测试余数分配是否合理。
        (F) 背景参考: IID 划分策略与类别平衡概念。
    """
    labels = train_dataset.targets
    client_indices = [[] for _ in range(args.num_clients)]
    for cls in range(len(train_dataset.classes)):
        # get the indices of current class
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)

        # get the number of sample class=cls indices for each client
        class_indices = (labels == cls).nonzero(as_tuple=True)[0]
        # random permutation
        class_indices = class_indices[torch.randperm(len(class_indices))]

        # calculate the number of samples for each client
        num_samples = len(class_indices)
        num_samples_per_client_per_class = num_samples // args.num_clients
        # other remaining samples
        remainder_samples = num_samples % args.num_clients

        # uniformly distribute the samples to each client
        for client_id in range(args.num_clients):
            start_idx = client_id * num_samples_per_client_per_class
            end_idx = start_idx + num_samples_per_client_per_class
            client_indices[client_id].extend(
                class_indices[start_idx:end_idx].tolist())
        # distribute the remaining samples to the first few clients
        for i in range(remainder_samples):
            client_indices[i].append(
                class_indices[-(i + 1)].item())
    client_indices = [torch.tensor(indices) for indices in client_indices]
    return client_indices


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''按照 Dirichlet 分布划分样本索引，实现非 IID 数据划分。

    参数:
        train_labels (Tensor/ndarray): 训练集标签。
        alpha (float): Dirichlet 分布参数，越小越非 IID。
        n_clients (int): 客户端数量。

    返回:
        List[np.ndarray]: 每个客户端的样本索引数组。

    费曼学习法:
        (A) 利用 Dirichlet 分布为各客户端生成不同比例的类别样本。
        (B) 类比给不同口味的饮料分发不同的配方比例。
        (C) 步骤拆解:
            1. 计算类别数量与对应样本索引集合。
            2. 生成形状为 (类别数, 客户端数) 的 Dirichlet 分布矩阵。
            3. 按照比例将每个类别的样本切分给各客户端。
            4. 拼接得到每个客户端的索引集合。
        (D) 示例:
            >>> client_indices = dirichlet_split_noniid(labels, 0.5, 10)
        (E) 边界条件与测试建议: 需确保 `alpha>0`；建议测试不同 `alpha` 下类别分布情况。
        (F) 背景参考: Dirichlet 分布及其在联邦学习数据划分中的应用。
    '''
    n_classes = train_labels.max()+1
    # (K, N) category label distribution matrix X, recording the proportion of each category assigned to each client
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) records the sample index set corresponding to K classes
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # Record the sample index sets corresponding to N clients
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split divides the sample index k_idcs of class k into N subsets according to the proportion fracs
        # i represents the i-th client, idcs represents its corresponding sample index set
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


def dataset_class_indices(dataset, class_label=None):
    """返回数据集中指定类别或全部类别的索引列表。

    参数:
        dataset (Dataset): 数据集对象，需包含 `targets` 与 `classes` 属性。
        class_label (Optional[int]): 若提供则返回该类别索引，否则返回所有类别索引列表。

    费曼学习法:
        (A) 函数找出哪些坐标属于特定类别，或列出全部类别的索引集合。
        (B) 类比从花园地图中标记每种花的位置。
        (C) 步骤拆解:
            1. 如果指定类别，则直接筛选该类别索引并转换为张量。
            2. 否则遍历所有类别，收集各类索引列表。
        (D) 示例:
            >>> indices = dataset_class_indices(train_ds, class_label=3)
        (E) 边界条件与测试建议: 需确保 `dataset.targets` 可索引；测试在多标签数据集上是否适用。
        (F) 背景参考: 分类数据索引管理。
    """
    num_classes = len(dataset.classes)
    if class_label:
        return torch.tensor(np.where(dataset.targets == class_label)[0])
    else:
        class_indices = [torch.tensor(np.where(dataset.targets == i)[
            0]) for i in range(num_classes)]
        return class_indices


def client_class_indices(client_indice, train_dataset):
    """返回某客户端在每个类别上的样本索引列表。

    参数:
        client_indice (Tensor): 客户端对应的样本索引张量。
        train_dataset (Dataset): 原始训练数据集。

    返回:
        List[Tensor]: 每个类别的索引张量。

    费曼学习法:
        (A) 函数将客户端数据按类别再细分一次。
        (B) 类比将杂货店收到的库存按品类重新整理。
        (C) 步骤拆解:
            1. 获取客户端索引对应的标签。
            2. 对每个类别筛选出该类别的样本索引。
        (D) 示例:
            >>> per_class = client_class_indices(client_indices[0], train_ds)
        (E) 边界条件与测试建议: 需确保标签可索引；测试类别数量变化时是否正常。
        (F) 背景参考: 类别分布解析工具。
    """
    labels = train_dataset.targets
    return [client_indice[labels[client_indice] == cls] for cls in range(len(train_dataset.classes))]


def class_imbalanced_partition(class_indices, im_iid_gamma, method='exponential'):
    """根据指数衰减策略采样各类别样本，构造类不均衡划分。

    参数:
        class_indices (List[Tensor]): 每个类别的样本索引集合。
        im_iid_gamma (float): 指数衰减系数，越小代表尾部类别越少。
        method (str): 采样方法，目前仅支持 'exponential'。

    返回:
        Tensor: 采样后的索引拼接结果。

    费曼学习法:
        (A) 函数让不同类别的样本数量按指数规律递减，模拟类不均衡。
        (B) 类比将奖品按价值从高到低逐渐减少数量发放。
        (C) 步骤拆解:
            1. 根据衰减系数计算每个类别需要保留的样本数量。
            2. 随机打乱并截取相应数量的样本索引。
            3. 将结果拼接成单个张量返回。
        (D) 示例:
            >>> imbalanced_indices = class_imbalanced_partition(class_indices, 0.8)
        (E) 边界条件与测试建议: 需确保各类别样本充足；测试不同 gamma 对分布的影响。
        (F) 背景参考: 类别不均衡采样策略、长尾分布模拟。
    """
    num_classes = len(class_indices)
    num_sample_per_class = [max(1, int(im_iid_gamma**(i / (num_classes-1)) * len(class_indices[i])))
                            for i in range(num_classes)]
    sampled_class_indices = [class_indices[i][torch.randperm(
        len(class_indices[i]))[:num_sample_per_class[i]]] for i in range(num_classes)]
    # print(f"num_sample_per_class: {num_sample_per_class}")
    return torch.cat(sampled_class_indices)


if __name__ == "__main__":
    pass


# __AI_ANNOTATION_SUMMARY__
# 函数 load_data: 按配置加载官方与自定义数据集，并统一标签格式。
# 函数 list_to_tensor: 将列表形式标签转换为张量。
# 函数 subset_by_idx: 基于索引构造带变换的子数据集。
# 函数 get_transform: 根据数据集/模型组合返回训练与测试变换。
# 函数 split_dataset: 生成客户端数据划分并可选缓存。
# 函数 save_partition_cache: 持久化客户端索引缓存。
# 函数 check_partition_cache: 检查并返回缓存路径。
# 函数 check_noniid_labels: 统计非 IID 划分的标签覆盖情况。
# 类 Partition 及方法: 管理数据分片与后门注入流程。
# 函数 iid_partition: 生成近乎均衡的 IID 划分索引。
# 函数 dirichlet_split_noniid: 按 Dirichlet 分布生成非 IID 划分。
# 函数 dataset_class_indices: 获取指定或全部类别的样本索引。
# 函数 client_class_indices: 提取客户端在各类别的索引列表。
# 函数 class_imbalanced_partition: 按指数衰减采样构建类不均衡划分。
