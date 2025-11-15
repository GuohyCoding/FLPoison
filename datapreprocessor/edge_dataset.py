# -*- coding: utf-8 -*-

import copy
import os
import pickle

import numpy as np
import rarfile
import torch
from torchvision import datasets

from .data_utils import Partition, get_transform


class EdgeDataset:
    """边缘样本数据集包装器，用于生成 Neurotoxin 等攻击所需的极端样本。

    根据联邦学习任务的数据集类型（MNIST 系列或 CIFAR10），分别加载 ARDIS 或 Southwest Airline
    外部数据集，并提供后门训练/测试集的混合、采样与标签替换功能。

    属性:
        args (argparse.Namespace): 全局配置对象，决定当前主数据集名称等。
        root (str): 外部数据集存储根目录。
        target_label (int): 后门攻击的目标标签。
        data_obj (Union[SouthwestAirline, ARDIS]): 具体的边缘数据集实例。
    """

    def __init__(self, args, target_label, root="./data"):
        """根据主数据集类型选择合适的边缘样本来源。

        概述:
            - CIFAR10 → SouthwestAirline
            - MNIST 系列 → ARDIS
            若数据集名称不支持，则抛出异常。

        参数:
            args (argparse.Namespace): 全局配置，需包含 `dataset` 字段。
            target_label (int): 后门目标标签。
            root (str): 外部数据集存放根目录。

        费曼学习法:
            (A) 该函数根据主数据集决定采用哪个外部边缘数据集。
            (B) 类比医生根据病种选择使用的特效药。
            (C) 步骤拆解:
                1. 保存传入的参数与目标标签。
                2. 判断主数据集类型，实例化对应边缘数据集对象。
                3. 若不支持该数据集组合则抛出异常提醒开发者。
            (D) 示例:
                >>> edge_ds = EdgeDataset(args, target_label=1)
            (E) 边界条件与测试建议: 需确保 `args.dataset` 包含支持的关键字；测试不同主数据集是否得到正确子类。
            (F) 背景参考: 《Neurotoxin》论文中对边缘样本的定义与使用。
        """
        self.args = args
        self.root = root
        self.target_label = target_label
        # ARDIS 的源标签为 7、SouthwestAirline 为 0，因此目标标签需与源标签不同。
        if self.args.dataset == "CIFAR10":
            self.data_obj = SouthwestAirline(args, self.target_label)
        elif "MNIST" in self.args.dataset:
            self.data_obj = ARDIS(args, self.target_label)
        else:
            raise ValueError(
                f"Unsupported dataset for edgecase: {self.args.dataset}")

    def get_poisoned_trainset(self, sample_len=None):
        """返回边缘样本训练集，可选随机采样指定数量。

        参数:
            sample_len (Optional[int]): 若提供则从完整集合中随机采样该数量。

        返回:
            Tuple[Tensor/ndarray, Tensor]: `(x, y)` 边缘训练样本与其目标标签。

        费曼学习法:
            (A) 函数调取外部数据集中已准备好的边缘训练样本。
            (B) 类比从“秘密仓库”中取出一批特殊材料，可按需抽样缩减数量。
            (C) 步骤拆解:
                1. 调用 `data_obj` 的 `get_poisoned_trainset` 获取全量样本。
                2. 若给定 `sample_len`，则随机抽取对应数量以控制规模。
                3. 返回采样后的 `(x, y)`。
            (D) 示例:
                >>> poisoned_x, poisoned_y = edge_ds.get_poisoned_trainset(sample_len=100)
            (E) 边界条件与测试建议: 应确认 `sample_len` 不超过数据量；测试随机采样是否稳定。
            (F) 背景参考: 边缘样本采样策略与后门注入概念。
        """
        x, y = self.data_obj.get_poisoned_trainset()
        if sample_len:
            indices = np.random.choice(
                range(len(x)), sample_len, replace=False)
            x, y = x[indices], y[indices]
        return x, y

    def get_poisoned_testset(self, sample_len=None):
        """获取边缘样本测试集，可选随机抽样。

        参数:
            sample_len (Optional[int]): 若提供则从测试集中抽取指定数量。

        返回:
            Dataset: 处理后的测试数据集对象。

        费曼学习法:
            (A) 函数产出用于评估后门 ASR 的边缘测试集。
            (B) 类比制作一套“突击检查题”，可根据需要缩减题量。
            (C) 步骤拆解:
                1. 调用数据对象的 `get_poisoned_testset` 获取测试集。
                2. 若指定 `sample_len`，随机抽取对应数量并更新数据/标签。
                3. 返回处理后的测试数据集。
            (D) 示例:
                >>> poisoned_test = edge_ds.get_poisoned_testset(sample_len=200)
            (E) 边界条件与测试建议: 需确保抽样不会破坏数据集结构；测试抽样后标签是否保持目标值。
            (F) 背景参考: 边缘样本在后门攻击评估中的作用。
        """
        test_dataset = self.data_obj.get_poisoned_testset()
        if sample_len:
            indices = np.random.choice(
                range(len(test_dataset.data)), sample_len, replace=False)
            test_dataset.data, test_dataset.targets = test_dataset.data[
                indices], test_dataset.targets[indices]
        return test_dataset

    def mix_trainset(self, clean_dataset, poisoning_ratio):
        """根据投毒比例混合干净训练集与边缘样本。

        概述:
            先按投毒比例计算恶意样本数量，并确保不超过边缘数据上限；
            之后随机抽取干净样本，与边缘样本拼接形成新的投毒训练集。

        参数:
            clean_dataset (Dataset): 原始干净训练数据集。
            poisoning_ratio (float): 投毒比例，范围 (0,1)。

        返回:
            Partition: 混合后的投毒训练数据集。

        费曼学习法:
            (A) 函数把“干净汤底”和“特殊调料”按比例混合成新菜品。
            (B) 类比调酒师控制酒与果汁的比例，既要保持口感又要藏住酒味。
            (C) 步骤拆解:
                1. 计算初始投毒样本数，并限制在边缘数据量上限。
                2. 根据投毒比例求出需保留的干净样本数。
                3. 随机抽取对应数量的干净样本并构建 `Partition`。
                4. 获取等量边缘样本，并与干净样本拼接。
                5. 返回混合后的数据分片。
            (D) 示例:
                >>> poisoned_train = edge_ds.mix_trainset(clean_ds, poisoning_ratio=0.1)
            (E) 边界条件与测试建议: 需确保边缘数据量足够；测试比例过大时是否正确截断。
            (F) 背景参考: 《Neurotoxin》中的边缘样本混合策略。
        """
        # 1. define sample length for benign and malicious data.
        # It trys to keep the total number of data in the mixed dataset to be the same as the clean dataset. However, if the poisoning_ratio is too high and total number it too large, there is no enough edge data to meet the poisoning_ratio, so the total number of data in the mixed dataset will be shorten accordingly.
        total_num = len(clean_dataset)
        poison_num = int(total_num * poisoning_ratio)
        # because the total southwest train dataset is 784, and the total ardis train sdataset is 660
        if self.args.dataset == "CIFAR10":
            poison_num = min(poison_num, 784)
        elif "MNIST" in self.args.dataset:
            poison_num = min(poison_num, 660)
        else:
            raise ValueError(
                f"Unsupported dataset for mix: {self.args.dataset}")
        # determine the number of benign data to meet the poisoning_ratio
        benign_num = int(poison_num / poisoning_ratio - poison_num)

        # 2. sample the benign and malicious data
        train_tran, _ = get_transform(self.args)
        sampled_benign_indices = np.random.choice(
            range(total_num), benign_num, replace=False)
        train_dataset = Partition(
            clean_dataset, sampled_benign_indices, transform=train_tran)

        # 3. get the tensor-type sampled poisoned dataset
        poisoned_x, poisoned_y = self.get_poisoned_trainset(
            poison_num)

        # 4. mix the sampled clean dataset with the poisoned dataset.
        # Note that poisoned_trainset.data and poisoned_x is tensor-type images, so torch.cat is used to concatenate them.
        poisoned_trainset = copy.deepcopy(train_dataset)

        if isinstance(poisoned_trainset.data, np.ndarray):
            poisoned_trainset.data = np.concatenate(
                (poisoned_trainset.data, poisoned_x), axis=0)
            poisoned_trainset.targets = np.concatenate(
                (poisoned_trainset.targets, poisoned_y), axis=0)
        elif isinstance(poisoned_trainset.data, torch.Tensor):
            poisoned_trainset.data = torch.cat(
                (poisoned_trainset.data, poisoned_x), dim=0)
            poisoned_trainset.targets = torch.cat(
                (poisoned_trainset.targets, poisoned_y), dim=0)
        return poisoned_trainset


class SouthwestAirline:
    """Southwest Airline 边缘样本集：用于 CIFAR10 的飞机类后门。

    将飞机（airplane, 标签 0）图像标注为目标标签（默认 9，trunk），作为后门触发样本来源。
    数据集来自公开仓库 `OOD_Federated_Learning`。
    """

    def __init__(self, args, target_label=None, root="./data/southwest"):
        """初始化 Southwest Airline 数据集并确保数据完整。

        参数:
            args (argparse.Namespace): 全局配置。
            target_label (Optional[int]): 后门目标标签，默认为 9。
            root (str): 数据存放目录。

        费曼学习法:
            (A) 函数确保飞机图片数据存在并载入。
            (B) 类比档案员检查是否有飞机照片档案，没有则从远程档案馆复制。
            (C) 步骤拆解:
                1. 设定源/目标标签并校验不冲突。
                2. 检查数据完整性，不足则下载。
                3. 读取训练与测试图像数据。
        """
        self.root = root
        self.source_label = 0  # airplane index=0 in CIFAR10
        self.args = args
        self.target_label = target_label if target_label else 9
        source_target_check(self.source_label,  self.target_label)
        self.check_integrity()
        self.read_dataset()

    def check_integrity(self):
        """检查本地是否存在所需文件，若缺失则从官方仓库下载。

        费曼学习法:
            (A) 函数确认“飞机照片档案”是否齐全。
            (B) 类比在仓库清点货物，不足则联系供应商补货。
            (C) 步骤拆解:
                1. 枚举所需 pickle 文件，确认在本地目录存在。
                2. 若缺失，则创建目录并逐个下载。
        """
        """
        If there is no ./saved_datasets, please download it from [southwest dataset link](https://github.com/ksreenivasan/OOD_Federated_Learning/tree/master/saved_datasets)
        load the dataset from the saved file
        """
        url_location = "https://raw.githubusercontent.com/ksreenivasan/OOD_Federated_Learning/master/saved_datasets/"
        self.filenames = ['southwest_images_new_train.pkl',
                          'southwest_images_new_test.pkl']
        all_files_exist = all(os.path.exists(os.path.join(self.root, file))
                              for file in self.filenames)
        if not all_files_exist:
            # mkdir root data
            os.makedirs(self.root, exist_ok=True)
            # download dataset
            try:
                for file in self.filenames:
                    download_link = os.path.join(url_location, file)
                    datasets.utils.download_url(download_link, self.root)
                print("Successfully downloaded the southwest dataset")
            except Exception as e:
                raise Exception(f"Exception: {e}")

    def read_dataset(self):
        """读取训练/测试飞机图像，并存入内存。

        费曼学习法:
            (A) 函数打开本地 pickle 文件，加载飞机图像数组。
            (B) 类比将档案盒中的图片整批摆放到工作桌上备用。
            (C) 步骤拆解:
                1. 打开训练集 pickle，加载图像列表。
                2. 打开测试集 pickle，加载图像列表。
        """
        # load the dataset from the saved file
        # Note that the labels are plane types, and we will convert them to target types
        with open(os.path.join(self.root, self.filenames[0]), 'rb') as train_f:
            self.southwest_train_images = pickle.load(train_f)

        with open(os.path.join(self.root, self.filenames[1]), 'rb') as test_f:
            self.southwest_test_images = pickle.load(test_f)

    def get_poisoned_trainset(self):
        """返回飞机图像及其目标标签，用于混入训练集。

        费曼学习法:
            (A) 函数提供全部训练飞机样本，并统一标注为目标标签。
            (B) 类比把所有飞机照片都贴上篡改后的标签。
        """
        # train dataset is provided to be tailed to the benign dataset
        self.train_images = self.southwest_train_images
        # transform is done in mix_trainset later
        # 9 is the default target label, trunk in CIAFR10
        self.train_labels = np.array(
            [self.target_label] * len(self.train_images))

        return self.train_images, self.train_labels

    def get_poisoned_testset(self):
        """构造带有目标标签的飞机测试集，便于评估后门攻击成功率。

        费曼学习法:
            (A) 函数返回全部飞机测试样本，并将其伪装成目标类。
            (B) 类比准备一整套伪装好的检查题目，用以评估攻击是否生效。
        """
        # test dataset is provided to be feeded into test loader
        self.test_images = self.southwest_test_images
        # 9 is the default target label, trunk in CIAFR10
        self.test_labels = np.array(
            [self.target_label] * len(self.test_images))

        test_trans = get_transform(self.args)[1]
        test_dataset = datasets.CIFAR10(
            './data', train=False, download=False, transform=test_trans)
        test_dataset.data, test_dataset.targets = self.test_images, self.test_labels
        return test_dataset


class ARDIS:
    """ARDIS 边缘样本集：用于 MNIST 系列的瑞典手写数字攻击。

    取手写数字 7 作为源标签，并统一替换为目标标签（默认 1），实现边缘样本后门。
    数据来自公开项目 `ARDIS_DATASET_IV`。
    """

    def __init__(self, args, target_label=None, root="./data"):
        """初始化 ARDIS 数据集，确保数据可用且源/目标标签不冲突。

        参数:
            args (argparse.Namespace): 全局配置对象。
            target_label (Optional[int]): 后门目标标签，默认为 1。
            root (str): 数据集根目录。
        """
        self.root = root
        self.source_label = 7
        self.args = args
        self.target_label = target_label if target_label else 1
        source_target_check(self.source_label,  self.target_label)
        self.check_integrity()
        self.read_dataset()

    def check_integrity(self):
        """校验 ARDIS 数据文件是否存在，如缺失则下载并解压 RAR 包。

        费曼学习法:
            (A) 函数查看“瑞典数字档案”是否完整，缺失则联网下载并解压。
            (B) 类比图书馆管理员从国外分馆调档案并拆封整理。
        """
        # check the integrity of the dataset, download if not exist
        self.data_path = f'{self.root}/ARDIS/'
        self.filenames = ['ARDIS_train_2828.csv', 'ARDIS_train_labels.csv',
                          'ARDIS_test_2828.csv', 'ARDIS_test_labels.csv']
        all_files_exist = all(os.path.exists(os.path.join(self.data_path, file))
                              for file in self.filenames)

        if not all_files_exist:
            # download dataset
            download_link = 'https://raw.githubusercontent.com/ardisdataset/ARDIS/master/ARDIS_DATASET_IV.rar'
            datasets.utils.download_url(download_link, self.data_path)
            raw_filename = download_link.split('/')[-1]
            # extract rar to csv, which requires the rarfile and unrar package, and system unrar command
            try:
                with rarfile.RarFile(os.path.join(self.data_path, raw_filename)) as rf:
                    rf.extractall(path=self.data_path)
                print("Successfully downloaded the southwest dataset")

            except Exception as e:
                raise Exception(
                    f"Extraction failed: Please install `rarfile` and `unrar` using `conda install rarfile unrar` or `pip install rarfile unrar`. If the issue persists, ensure `unrar` is installed on your system with `sudo apt install unrar` for Linux, `sud brew install unrar` for MacOS.\n Exception: {e}")
            else:  # no Exception, that is, the extraction is successful
                # remove the downloaded package if the extraction is successful
                os.remove(os.path.join(self.data_path, raw_filename))

    def read_dataset(self):
        """读取 CSV 数据并转换为 MNIST 格式，提取源标签样本。

        费曼学习法:
            (A) 函数将 CSV 中的图像重塑为 28×28，并找出所有数字 7。
            (B) 类比从大型图像档案中挑选所有写有“7”的纸张，并统一转换格式保存。
        """
        # load the data from csv's and convert to tensor, becuase MNIST dataset's data and target are tensor-type format
        def load_cvs(idx): return torch.from_numpy(np.loadtxt(
            os.path.join(self.data_path, self.filenames[idx]), dtype='float32'))
        self.train_images, self.train_labels = load_cvs(0), load_cvs(1)
        self.test_images, self.test_labels = load_cvs(2), load_cvs(3)

        # reshape to be [samples][width][height]
        def to_MNIST(x): return x.reshape(x.shape[0], 28, 28)
        self.train_images, self.test_images = to_MNIST(
            self.train_images), to_MNIST(self.test_images)

        # raw labels are one-hot encoded
        # 1. convert one-hot encoded labels to integer labels
        def onehot_to_label(y): return np.argmax(
            y, axis=1)  # default return y-type: tensor
        self.train_labels, self.test_labels = onehot_to_label(
            self.train_labels), onehot_to_label(self.test_labels)

        # 2. get the images and labels for digit 7
        def filter(train_x, train_y):
            # Note that if use argwhere the 1-dimension will added, which is not expected
            indices = train_y[train_y == self.source_label]
            sampled_images = train_x[indices]
            sampled_labels = torch.tensor([self.source_label] * len(indices))
            return sampled_images, sampled_labels

        # sample source label=7 train images as sampled_train_images. test images are fully used wihout sampling.
        self.sampled_train_images, self.sampled_train_labels = filter(
            self.train_images, self.train_labels)
        self.sampled_test_images, self.sampled_test_labels = self.test_images, self.test_labels

    def get_poisoned_trainset(self):
        """返回源标签样本但标签全部改为目标标签的训练集。

        费曼学习法:
            (A) 函数将所有数字 7 标签改成攻击目标数字。
            (B) 类比更换试卷答案，让所有“7”都被重写成“1”。
        """
        # train_trans = get_transform(self.args)[0]
        # train_dataset = datasets.MNIST('./data', train=True, download=True,
        #                                transform=train_trans)
        # transform is done in mix_trainset later
        self.posioned_labels = torch.tensor(
            [self.target_label] * len(self.sampled_train_labels))
        # train_dataset.data, train_dataset.targets = self.images, self.labels
        # return train_dataset
        return self.sampled_train_images, self.posioned_labels

    def get_poisoned_testset(self):
        """构造带后门标签的 MNIST 测试集，用于评估攻击成功率。

        费曼学习法:
            (A) 函数对所有测试样本施加目标标签，以便检验后门效果。
            (B) 类比给评测员发放只包含“伪造答案”的试卷。
        """
        # args.dataset
        test_trans = get_transform(self.args)[1]
        test_dataset = datasets.MNIST(
            './data', train=False, download=False, transform=test_trans)
        self.posioned_labels = torch.tensor(
            [self.target_label] * len(self.sampled_test_labels))
        test_dataset.data, test_dataset.targets = self.sampled_test_images, self.posioned_labels
        return test_dataset


def source_target_check(source_label, target_label):
    """校验源标签与目标标签不同，防止生成无效后门。

    费曼学习法:
        (A) 相当于提醒攻击者不要“自我否定”，源目标不可相同。
        (B) 类比在开局前确认棋子阵营不同，否则对弈失去意义。
    """
    if source_label == target_label:
        raise ValueError(
            f"The source label and target label should not be the same, currently they are both {source_label}")


if __name__ == "__main__":
    class args:
        dataset = "CIFAR10"
    tmp = EdgeDataset(args)


# __AI_ANNOTATION_SUMMARY__
# 类 EdgeDataset: 选择外部边缘样本数据源并提供获取/混合接口。
# 方法 __init__ (EdgeDataset): 根据主数据集实例化 ARDIS 或 SouthwestAirline。
# 方法 get_poisoned_trainset: 返回边缘训练样本，可选随机抽样。
# 方法 get_poisoned_testset: 返回边缘测试集，可选随机抽样。
# 方法 mix_trainset: 按投毒比例将干净数据与边缘样本混合。
# 类 SouthwestAirline 及方法: 管理飞机图像边缘样本的下载、读取与标签转换。
# 类 ARDIS 及方法: 处理瑞典手写数字边缘样本的下载、解析与过滤。
# 函数 source_target_check: 校验源/目标标签不同，避免配置错误。
