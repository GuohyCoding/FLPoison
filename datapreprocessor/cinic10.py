# -*- coding: utf-8 -*-

import os
import pickle
import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive, extract_archive, check_integrity
import numpy as np


class CINIC10(ImageFolder):
    """CINIC-10 图像数据集包装器，支持自动下载、缓存与按训练/测试划分加载。

    CINIC-10 是介于 CIFAR-10 与 ImageNet 之间的中等规模数据集，由 ImageNet 与 CIFAR-10 图像混合而成。
    该包装类继承自 `torchvision.datasets.ImageFolder`，保留原有目录结构优势，并提供 pickle 缓存以加速重复加载。

    类属性:
        base_folder (str): 默认数据子目录名称 `cinic-10`。
    """

    base_folder = 'cinic-10'

    def __init__(self, root, train, download, transform=None, target_transform=None):
        """初始化 CINIC-10 数据集对象，必要时自动下载并构建缓存。

        概述:
            若 `download=True` 且本地数据缺失，则从官方地址下载压缩包并解压。
            继承 `ImageFolder` 收集样本路径，同时将图像与标签缓存至 pickle 文件，后续可直接加载 numpy 数组。

        参数:
            root (str): 数据集根目录，解压后将出现 `cinic-10` 文件夹。
            train (bool): 是否加载训练集；若为 False 则加载测试集（CINIC-10 官方亦提供验证集，可按需扩展）。
            download (bool): 是否在缺失时自动下载数据。
            transform (Callable): 图像转换流水线（通常含归一化与数据增强）。
            target_transform (Callable): 标签转换函数（可选）。

        返回:
            None。

        异常:
            RuntimeError: 当数据既未下载又不存在本地缓存时抛出。

        复杂度:
            初始化阶段时间复杂度与磁盘扫描规模有关，约 O(n)；空间复杂度 O(n)（缓存图像数组）。

        费曼学习法:
            (A) 构造函数负责确保数据就绪（下载、校验、缓存）并完成继承初始化。
            (B) 类比档案管理员：若档案缺失则先从总库调取，之后制作副本以便快速调用。
            (C) 步骤拆解:
                1. 设置数据下载信息（URL、MD5 等）。
                2. 根据 `train` 标志确定读取的文件夹与缓存路径。
                3. 若需要下载则调用 `download()`。
                4. 调用 `_check_integrity`，若失败则抛出异常提示用户处理。
                5. 使用父类 `ImageFolder` 初始化样本路径与标签。
                6. 若缓存存在则直接加载，否则读取图像转为数组并写入 pickle。
            (D) 示例:
                >>> dataset = CINIC10(root="data", train=True, download=True, transform=transforms.ToTensor())
                >>> len(dataset)
                90000  # 训练集中样本数量，视官方划分而定
            (E) 边界条件与测试建议: 确保磁盘空间足够；测试首次下载与再次加载时的缓存差异。
            (F) 背景参考: CINIC-10 官方介绍、PyTorch `ImageFolder` 使用指南。
        """
        self.url = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz'
        self.archive_filename = 'CINIC-10.tar.gz'
        self.tgz_md5 = '6ee4d0c996905fe93221de577967a372'
        self.root = root
        self.train = train
        self.base_folder = os.path.join(self.root, self.base_folder)
        self.data_folder = os.path.join(
            self.base_folder, 'train' if train else 'test')
        self.pkl_names = os.path.join(
            self.base_folder, f'cinic10_{"train" if self.train else "test"}.pkl')
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')
        super().__init__(self.data_folder, transform, target_transform)

        if self.check_pkls():
            with open(self.pkl_names, 'rb') as f:
                self.data, self.targets = pickle.load(f)
        else:
            # 扫描所有图像路径，加载为 numpy 数组后缓存，便于后续快速读取。
            self.data = np.array([np.array(self.loader(s[0]))
                                  for s in self.samples])
            self.targets = np.array(self.targets)
            with open(self.pkl_names, 'wb') as f:
                pickle.dump((self.data, self.targets), f)

    def check_pkls(self):
        """判断对应的 pickle 缓存是否已存在。

        返回:
            bool: 若缓存文件存在则返回 True，否则 False。

        费曼学习法:
            (A) 相当于问：“之前有没有把图像打包成缓存？”
            (B) 类比检查仓库里是否已有现成的清单。
            (C) 步骤拆解:
                1. 使用 `os.path.exists` 判断缓存文件是否存在。
            (D) 示例:
                >>> dataset.check_pkls()
                True
            (E) 边界条件与测试建议: 确保路径正确；可删除缓存文件后再次调用观察返回值。
            (F) 背景参考: 文件系统操作基础。
        """
        if os.path.exists(self.pkl_names):
            return True
        return False

    def __getitem__(self, index):
        """按索引返回图像张量与标签。

        概述:
            由于 `ImageFolder` 默认从磁盘读取，此实现改为直接从内存中的 numpy 数组取图像，再应用 transform。

        参数:
            index (int): 样本索引。

        返回:
            Tuple[Tensor 或 ndarray, int]: `(image, target)`，其中 `image` 经 transform 处理后通常为张量。

        异常:
            IndexError: 当索引越界时由 numpy 自动抛出。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)。

        费曼学习法:
            (A) 函数像仓库管理员按编号取出已经缓存好的箱子，再贴上统一格式的标签。
            (B) 类比从存档柜中取出某页文档，并复印成指定格式。
            (C) 步骤拆解:
                1. 从缓存的 `self.data` 中取出图像数组。
                2. 将标签转为 `int` 类型。
                3. 若设置变换则对图像执行预处理。
                4. 若设置标签变换则对标签处理。
                5. 返回处理后的 `(image, target)`。
            (D) 示例:
                >>> img, label = dataset[10]
                >>> img.shape
                torch.Size([3, 32, 32])
            (E) 边界条件与测试建议: transform 应能处理 numpy 数组；若需要 PIL，可在 transform 内先转换。
            (F) 背景参考: PyTorch 数据集规范、numpy 与 tensor 转换。
        """
        img, target = self.data[index], int(self.targets[index])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def _check_data_integrity(self):
        """检查数据目录是否存在，作为完整性判据的一部分。"""
        return os.path.exists(self.data_folder)

    def _check_integrity(self):
        """综合验证数据与压缩包完整性，必要时尝试解压。

        概述:
            若数据文件夹不存在，则检查压缩包并验证 MD5，验证通过后解压；若解压成功则视为数据完整。
            该方法在初始化阶段用于判断是否需要重新下载或提示用户。

        返回:
            bool: 数据可用返回 True，否则 False。

        费曼学习法:
            (A) 函数确保数据“货源”可靠：要么已有整理好的仓库，要么有完好的压缩包供解压。
            (B) 类比检查快递包裹，若箱子完好则拆封补货，否则要求重新发货。
            (C) 步骤拆解:
                1. 检查数据目录是否存在，若存在直接返回 True。
                2. 若不存在，则寻找压缩包。
                3. 若压缩包存在则校验 MD5，成功则解压并返回 True。
                4. 若压缩包缺失或校验失败，则提示用户重新下载。
            (D) 示例:
                >>> dataset._check_integrity()
                True
            (E) 边界条件与测试建议: 压缩包校验依赖 MD5；可测试损坏压缩包时的提示信息。
            (F) 背景参考: 数据集完整性验证、MD5 校验机制。
        """
        data_exist = self._check_data_integrity()
        if not data_exist:
            print('Dataset not found. Checking archive...')
            archive = os.path.join(self.root, self.archive_filename)
            archive_exist = os.path.exists(archive)
            if archive_exist:
                print('Archive found. Checking integrity...')
                if check_integrity(archive, self.tgz_md5):
                    print('Archive integrity verified. Extracting...')
                    extract_archive(from_path=archive,
                                    to_path=self.base_folder)
                    print('Extraction completed.')
                    return True
                else:
                    print(
                        'Archive corrupted. Please remove the archive and re-download the dataset.')
                    return False
            else:
                print('Archive not found.')
                return False
        return self._check_data_integrity()

    def download(self):
        """下载并解压 CINIC-10 数据集压缩包（若本地未存在或损坏）。

        概述:
            先调用 `_check_integrity` 判断是否已具备数据或有效压缩包；若无数据，则使用 torchvision 提供的下载工具获取并解压。

        费曼学习法:
            (A) 相当于派人去仓库取货：若库里已有就不再取。
            (B) 类比家庭主妇先检查冰箱是否有食材，缺货才去超市采购。
            (C) 步骤拆解:
                1. 若 `_check_integrity` 通过，说明已有数据，无需下载。
                2. 否则调用 `download_and_extract_archive` 下载压缩包并解压至目标目录。
            (D) 示例:
                >>> dataset.download()
            (E) 边界条件与测试建议: 需要网络访问权限与足够磁盘空间；下载失败应提示用户重试。
            (F) 背景参考: torchvision 数据集下载 API。
        """
        if self._check_integrity():
            print('Files already downloaded and verified.')
            return
        print('Downloading dataset...')
        download_and_extract_archive(
            self.url, self.root, self.base_folder, filename=self.archive_filename, md5=self.tgz_md5)
        print('Download completed.')


# __AI_ANNOTATION_SUMMARY__
# 类 CINIC10: 扩展 ImageFolder，提供下载、完整性检测与缓存功能。
# 方法 __init__: 初始化数据目录、可选下载并构建或加载缓存。
# 方法 check_pkls: 判断缓存 pickle 是否存在。
# 方法 __getitem__: 返回缓存图像与标签，可应用转换。
# 方法 _check_data_integrity: 简单检查数据目录存在性。
# 方法 _check_integrity: 验证数据或压缩包完整性并视情况解压。
# 方法 download: 下载并解压 CINIC-10 数据集。
