# -*- coding: utf-8 -*-

import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torchvision.models as models
from torchvision.datasets.utils import download_and_extract_archive


class CHMNIST(Dataset):
    """Kather 肠道组织病理图像（CHMNIST）数据集包装器，支持自动下载、划分与变换。

    该数据集来自 Kather 等人在 2016 年发布的组织切片纹理数据，被广泛用于图像分类与医学影像研究。
    类实现兼容 PyTorch `Dataset` 接口，可直接贴合 `DataLoader` 与深度模型训练流程。

    类属性:
        dataset_url (str): 官方压缩包下载地址。
        zip_md5 (str): 压缩包 MD5 校验值，用于验证下载完整性。
    """

    dataset_url = "https://zenodo.org/records/53169/files/Kather_texture_2016_image_tiles_5000.zip?download=1"
    zip_md5 = "0ddbebfc56344752028fda72602aaade"

    def __init__(self, root, train=True, download=False, transform=None, test_split=0.2, random_seed=42):
        """初始化 CHMNIST 数据集实例，对原始图像按需自动下载并划分训练/测试集。

        概述:
            如果 `download=True` 且本地不存在数据，则自动从官方地址下载并解压。
            对所有图像路径进行收集，将标签映射为整数索引，并使用 `train_test_split` 进行分割。

        参数:
            root (str): 数据集根目录，解压后应包含 `Kather_texture_2016_image_tiles_5000` 文件夹。
            train (bool): 是否加载训练集；若为 False 则加载测试集。
            download (bool): 是否在缺失时自动下载并解压数据集。
            transform (Callable): 对 PIL 图像执行的变换流水线。
            test_split (float): 数据集中作为测试集的比例。
            random_seed (int): 随机种子，确保训练/测试划分可复现。

        返回:
            None。

        异常:
            FileNotFoundError: 当本地未找到数据且 `download=False` 时访问文件可能失败。

        复杂度:
            时间复杂度 O(n)，n 为图像数量；空间复杂度 O(n)，存储路径与标签。

        费曼学习法:
            (A) 该构造函数负责把磁盘上的原始图像整理成可迭代的训练/测试集合。
            (B) 可以将其比作“图书管理员”为图书贴标签、分区并记录位置。
            (C) 步骤拆解:
                1. 记录数据集根目录与变换配置。
                2. 如需下载且本地缺失，则自动下载并解压。
                3. 遍历类别子目录，收集所有图像路径及对应标签。
                4. 使用 `train_test_split` 将数据集划分为训练与测试集合。
                5. 根据 `train` 标记选择最终的图像路径列表与标签列表。
            (D) 示例:
                >>> dataset = CHMNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
                >>> len(dataset)
                4000  # 具体数量取决于划分比例
            (E) 边界条件与测试建议: 需要确保 `test_split` 处于 (0,1)；建议测试自动下载流程与划分是否稳定。
            (F) 背景参考: 《Practical Guide to Medical Image Analysis》、PyTorch 数据集实现范式。
        """
        self.root_dir = os.path.join(root, "Kather_texture_2016_image_tiles_5000")
        self.transform = transform
        self.train = train

        # 自动下载并解压数据集（若用户允许）。
        if download:
            if not os.path.exists(self.root_dir):
                print("Dataset not found. Downloading...")
                download_and_extract_archive(
                    url=self.dataset_url,
                    download_root=root,
                    extract_root=root,
                    filename="chmnist.zip",
                    md5=self.zip_md5,
                    remove_finished=True
                )

        self.classes = sorted(os.listdir(self.root_dir))  # 假设子文件夹名称即类别名
        self.image_paths = []
        self.targets = []

        # 收集所有图像路径及标签索引。
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_file)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.targets.append(label)

        # 划分训练/测试集合，确保类别比例一致。
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            self.image_paths, self.targets, test_size=test_split, random_state=random_seed, stratify=self.targets
        )

        if self.train:
            self.image_paths = train_paths
            self.targets = train_labels
        else:
            self.image_paths = test_paths
            self.targets = test_labels

    def __len__(self):
        """返回数据集样本数量。

        概述:
            基于当前选择的训练或测试集合，返回图像路径列表长度。

        返回:
            int: 样本数量。

        费曼学习法:
            (A) 这个方法告诉你数据集中有多少张图。
            (B) 像清点仓库库存一样数一数货架上有多少盒子。
            (C) 步骤拆解:
                1. 直接返回 `self.image_paths` 的长度。
            (D) 示例:
                >>> len(dataset)
                4000
            (E) 边界条件与测试建议: 无特别限制；可测试初始化后长度是否匹配划分比例。
            (F) 背景参考: PyTorch `Dataset` 接口要求。
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """按索引获取一条图像样本及其标签。

        概述:
            读取指定索引的图像路径，将图像转换为 RGB 格式并应用预处理，返回张量与标签。

        参数:
            idx (int): 样本索引。

        返回:
            Tuple[Tensor, int]: `(image, label)`，其中 `image` 经过 `transform` 后为张量。

        异常:
            IndexError: 当索引越界时由列表访问自动抛出。

        复杂度:
            时间复杂度 O(1)（磁盘读取视为常数级），空间复杂度 O(1)。

        费曼学习法:
            (A) 函数像仓库管理员按货架编号取出对应的盒子。
            (B) 比喻为打开文件柜中的第 `idx` 个文件，翻译后交给使用者。
            (C) 步骤拆解:
                1. 获取索引对应的图像路径与标签。
                2. 使用 PIL 打开图像，并转换为 RGB 模式。
                3. 若提供变换函数则应用之，将图像转为张量。
                4. 返回图像张量与标签。
            (D) 示例:
                >>> img, label = dataset[0]
                >>> img.shape
                torch.Size([3, 224, 224])
            (E) 边界条件与测试建议: 确保图像文件存在且可读；测试不同 `transform` 是否生效。
            (F) 背景参考: 影像预处理流程、PyTorch 数据集操作习惯。
        """
        img_path = self.image_paths[idx]
        label = self.targets[idx]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    # ================== 示例用法（仅演示数据加载与模型构建流程） ==================

    # 超参数设置
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    num_classes = 8

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 训练数据增强流水线（ResNet 预设尺寸与归一化）
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet requires 224x224
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

    # 测试集预处理，不包含随机增强
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 注意：以下示例需要传入参数 `root`，而非 `root_dir`
    dataset_root = "data"

    train_dataset = CHMNIST(root=dataset_root, transform=train_transform, train=True, test_split=0.2)
    test_dataset = CHMNIST(root=dataset_root, transform=test_transform, train=False, test_split=0.2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Class names: {train_dataset.classes}")

    # 使用 ImageNet 预训练的 ResNet34 作为基线模型
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)


# __AI_ANNOTATION_SUMMARY__
# 类 CHMNIST: 封装 Kather 图像数据集的下载、划分与访问逻辑。
# 方法 __init__: 自动下载数据、收集路径并完成训练/测试划分。
# 方法 __len__: 返回当前子集的样本数量。
# 方法 __getitem__: 读取索引样本并应用图像预处理。
