# -*- coding: utf-8 -*-

import hashlib
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, extract_archive

dir_structure_help = r"""
TinyImageNetPath
├── test
│  └── images
│      ├── test_0.JPEG
│      ├── t...
│      └── ...
├── train
│  ├── n01443537
│  │  ├── images
│  │  │  ├── n01443537_0.JPEG
│  │  │  ├── n...
│  │  │  └── ...
│  │  └── n01443537_boxes.txt
│  ├── n01629819
│  │  ├── images
│  │  │  ├── n01629819_0.JPEG
│  │  │  ├── n...
│  │  │  └── ...
│  │  └── n01629819_boxes.txt
│  ├── n...
│  │  ├── images
│  │  │  ├── ...
│  │  │  └── ...
├── val
│  ├── images
│  │  ├── val_0.JPEG
│  │  ├── v...
│  │  └── ...
│  └── val_annotations.txt
├── wnids.txt
└── words.txt
"""


class TinyImageNet(Dataset):
    """TinyImageNet 数据集包装器，与 `torchvision.datasets` 风格保持一致。

    TinyImageNet 包含 200 个类别、每类 500 张训练图像，图像尺寸为 64×64。
    官方未提供测试标签，因此仅封装 train/val 两个 split。该类负责下载、校验并解析标签文件。

    类属性:
        base_folder (str): 数据集目录名 `tiny-imagenet-200`。
        url (str): 官方下载地址。
        filename (str): 压缩包文件名。
        md5 (str): 压缩包 MD5 校验值。
        splits (List[str]): 支持的划分列表，包含 `'train'` 与 `'val'`。
    """

    base_folder = 'tiny-imagenet-200'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'
    splits = ['train', 'val']  # test folder has no labels, so not included

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """初始化 TinyImageNet 数据集，根据需求选择训练或验证集。

        参数:
            root (str): 数据集根目录。
            train (bool): 是否加载训练集；若为 False 则加载验证集。
            transform (Callable): 图像数据的预处理/增强函数。
            target_transform (Callable): 标签转换函数。
            download (bool): 若为 True 且本地数据缺失，则自动下载并解压。

        返回:
            None。

        异常:
            RuntimeError: 下载或解压失败会由下游函数抛出。

        复杂度:
            初始化开销主要来自 `_load_metadata`，约 O(n)。

        费曼学习法:
            (A) 构造函数确保数据存在并加载路径/标签。
            (B) 类比档案管理员在查阅文件前，先确认资料是否齐全并整理索引。
            (C) 步骤拆解:
                1. 记录根目录、变换函数与数据 split。
                2. 如需下载且本地缺失则调用 `download()`。
                3. 初始化空的 `data` 与 `targets` 列表。
                4. 调用 `_load_metadata` 从文件结构中解析图像路径与标签。
            (D) 示例:
                >>> dataset = TinyImageNet(root='data', train=True, download=True)
                >>> len(dataset)
            (E) 边界条件与测试建议: 需确保磁盘空间充足；建议测试 train/val 两个分支加载情况。
            (F) 背景参考: PyTorch 官方 TinyImageNet 教程与数据结构介绍。
        """
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        self.data = []
        self.targets = []
        split = 'train' if train else 'val'
        self._load_metadata(split)

    def _path(self, *paths):
        """构造数据集子路径的便捷函数。"""
        return os.path.join(self.root, self.base_folder, *paths)

    def _load_metadata(self, split):
        """解析 TinyImageNet 文件结构，生成图像路径与标签列表。

        参数:
            split (str): `'train'` 或 `'val'`。

        费曼学习法:
            (A) 函数读取官方索引文件，根据 split 加载对应图像及标签。
            (B) 类比图书管理员按照目录将每本书与分类编号对应起来。
            (C) 步骤拆解:
                1. 读取 `wnids.txt`，构建类别到索引的映射。
                2. 若为训练集，遍历每个类别文件夹中的图像并记录标签。
                3. 若为验证集，读取 `val_annotations.txt` 获取图像与标签对应关系。
                4. 生成 `self.data` 与 `self.targets`，并统计类别集合。
            (D) 示例:
                >>> dataset._load_metadata('train')
            (E) 边界条件与测试建议: 需确保目录结构完整；测试标签映射是否正确。
            (F) 背景参考: TinyImageNet 官方白皮书与目录说明。
        """
        data_dir = os.path.join(self.root, self.base_folder)
        with open(self._path('wnids.txt')) as f:
            wnids = [line.strip() for line in f.readlines()]
            wnid_to_label = {wnid: idx for idx, wnid in enumerate(wnids)}

        if split == 'train':
            for wnid in wnids:
                imgs_dir = self._path('train', wnid, 'images')
                for img_name in os.listdir(imgs_dir):
                    self.data.append(os.path.join(imgs_dir, img_name))
                    self.targets.append(wnid_to_label[wnid])
        else:
            val_img_dir = self._path('val', 'images')
            val_labels_path = os.path.join(data_dir, 'val', 'val_annotations.txt')
            val_dict = {}
            with open(val_labels_path) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    val_dict[parts[0]] = wnid_to_label[parts[1]]
            for img_name in os.listdir(val_img_dir):
                if img_name in val_dict:
                    self.data.append(os.path.join(val_img_dir, img_name))
                    self.targets.append(val_dict[img_name])
        self.classes = np.unique(self.targets)

    def __len__(self):
        """返回样本数量，用于兼容 PyTorch 数据集接口。"""
        return len(self.data)

    def __getitem__(self, idx):
        """按索引加载图像并返回处理后的样本与标签。

        参数:
            idx (int): 样本索引。

        返回:
            Tuple[Tensor, int]: `(image, label)`。

        费曼学习法:
            (A) 函数取出第 `idx` 张图像，转换为 RGB 并应用变换。
            (B) 类比从档案柜抽取某张照片，按规定格式扫描后附上标签。
        """
        img_path = self.data[idx]
        label = self.targets[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def download(self):
        """下载并解压 TinyImageNet 数据集，若本地已存在则跳过。

        费曼学习法:
            (A) 函数检查当前是否已解压完毕；若否则下载并校验压缩包。
            (B) 类比先检查仓库是否已有货物，没有再向中央仓库调货。
        """
        if os.path.exists(self._path('train')) and os.path.exists(self._path('val')):
            return
        else:
            print("Dataset folder incomplete. Check zip file...")
            # Check if the zip file exists to verify MD5
            zip_path = os.path.join(self.root, self.filename)
            if os.path.exists(zip_path):
                # Calculate MD5 of the existing file
                md5_hash = hashlib.md5()
                with open(zip_path, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        md5_hash.update(byte_block)
                md5_actual = md5_hash.hexdigest()

                if md5_actual == self.md5:
                    print("Zip file exists and is complete. Unzipping...")
                    extract_archive(zip_path, self.root)
                else:
                    print("Dataset exists but is corrupted. Re-downloading...")
                    download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)
            else:
                download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)


# __AI_ANNOTATION_SUMMARY__
# 类 TinyImageNet: 封装 TinyImageNet 数据集的下载、解析与访问流程。
# 方法 __init__: 初始化数据集，必要时下载并加载元数据。
# 方法 _path: 构造数据目录下的子路径。
# 方法 _load_metadata: 解析 train/val 目录，生成图像路径与标签列表。
# 方法 __len__: 返回样本数量，兼容 PyTorch 接口。
# 方法 __getitem__: 加载指定索引的样本并应用变换。
# 方法 download: 判断数据是否存在，若缺失或损坏则下载并解压。
