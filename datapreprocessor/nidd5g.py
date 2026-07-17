# -*- coding: utf-8 -*-

"""NIDD5G: 5G-NIDD 网络入侵检测数据集的 PyTorch Dataset 封装。

数据集来源:
    Samarakoon, S. et al. "5G-NIDD: A Comprehensive Network Intrusion Detection
    Dataset Generated over 5G Wireless Network." IEEE Data Descriptions, 2025.
    https://ieee-dataport.org/documents/5g-nidd-...

使用前须知:
    本数据集不支持自动下载，需从 IEEE DataPort 手动获取 Encoded.csv
    并放置于 <data_root>/5GNIDD/Encoded.csv。

预处理流程（与论文一致）:
    1. 读取 Encoded.csv（已含 one-hot 编码的分类特征，共约 112 列）
    2. 按 70:30 分层抽样划分训练/测试集
    3. 对训练集用 ANOVA F-score 选取 top-k 特征（默认 k=10）
    4. 用训练集统计量对选定特征做 Z-score 归一化
    5. 将特征 / 标签转为 FloatTensor / LongTensor
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


# 9 类标签（来自 Encoded.csv 的 "Attack Type" 列实测值）
_CLASSES = [
    "Benign",
    "HTTPFlood",
    "ICMPFlood",
    "SYNFlood",
    "SYNScan",
    "SlowrateDoS",
    "TCPConnectScan",
    "UDPFlood",
    "UDPScan",
]

# CSV 中存储 9 类标签的列名（Attack Type 为多分类，Label 为二分类）
_LABEL_COL = "Attack Type"

# 需从特征矩阵中剔除的非特征列（含 pandas 导出的匿名索引列）
_NON_FEATURE_COLS = {"", "Label", "Attack Type", "Attack Tool"}

# 训练 / 测试分割比例
_TEST_RATIO = 0.3

# 预处理缓存文件名（避免重复加载大 CSV）
_CACHE_FILE = "5gnidd_preprocessed.npz"


class NIDD5G(Dataset):
    """5G-NIDD 表格型网络流量数据集。

    属性:
        data (FloatTensor): 形状 (N, num_features)，经特征选择与 Z-score 归一化后的特征矩阵。
        targets (LongTensor): 形状 (N,)，类别标签（0-8）。
        classes (List[str]): 9 类标签名称列表。
    """

    classes = _CLASSES

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,  # 保留参数以兼容 FLPoison 调用约定，实际不支持自动下载
        transform=None,
        num_features: int = 10,
        random_state: int = 42,
    ):
        """
        参数:
            root (str): 数据根目录，期望 CSV 在 <root>/5GNIDD/Encoded.csv。
            train (bool): True 返回训练集，False 返回测试集。
            num_features (int): ANOVA 特征选取数量，论文默认为 10。
            random_state (int): 划分与特征选取的随机种子。
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.num_features = num_features
        self.random_state = random_state

        data_dir = os.path.join(root, "5GNIDD")
        csv_path = os.path.join(data_dir, "Encoded.csv")
        cache_path = os.path.join(data_dir, _CACHE_FILE)

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"目录 {data_dir} 不存在。\n"
                "请从 IEEE DataPort 下载 5G-NIDD 数据集：\n"
                "  https://ieee-dataport.org/documents/5g-nidd-...\n"
                f"并将 Encoded.csv 放置于 {data_dir}/"
            )

        # 优先使用缓存，避免重复加载 1.2M 行 CSV
        if os.path.exists(cache_path):
            cache = np.load(cache_path)
            X_train = cache["X_train"]
            X_test = cache["X_test"]
            y_train = cache["y_train"]
            y_test = cache["y_test"]
        else:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"未找到 {csv_path}。\n"
                    "请从 IEEE DataPort 下载 Encoded.csv 并放置于该路径。"
                )
            X_train, X_test, y_train, y_test = _preprocess(
                csv_path, num_features, random_state
            )
            np.savez(cache_path, X_train=X_train, X_test=X_test,
                     y_train=y_train, y_test=y_test)

        if train:
            self.data = torch.tensor(X_train, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.long)
        else:
            self.data = torch.tensor(X_test, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx].item()


def _preprocess(csv_path: str, num_features: int, random_state: int):
    """加载 CSV，执行特征选择与 Z-score 归一化，返回 numpy 数组四元组。

    返回:
        (X_train, X_test, y_train, y_test): 全部为 float32 / int64 numpy 数组。
    """
    try:
        import pandas as pd
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise ImportError(
            "处理 5G-NIDD 需要 pandas 与 scikit-learn：\n"
            "  pip install pandas scikit-learn"
        ) from e

    print(f"[NIDD5G] 正在加载 {csv_path} ...")
    df = pd.read_csv(csv_path)

    # 自动探测标签列（按优先级尝试常见列名）
    label_col = _detect_label_col(df)
    print(f"[NIDD5G] 检测到标签列: '{label_col}'，共 {len(df)} 条样本")

    # 编码字符串标签 → 整数
    label_map = {name: i for i, name in enumerate(_CLASSES)}
    y_raw = df[label_col].astype(str).str.strip()
    # 未知标签映射到 -1（后续过滤）
    y = y_raw.map(lambda x: label_map.get(x, -1)).values.astype(np.int64)
    valid_mask = y >= 0
    if valid_mask.sum() < len(y):
        unknown = set(y_raw[~valid_mask].unique())
        print(f"[NIDD5G] 警告：发现未知标签 {unknown}，已过滤 {(~valid_mask).sum()} 条")
    df = df[valid_mask]
    y = y[valid_mask]

    drop_cols = [c for c in df.columns if c in _NON_FEATURE_COLS]
    X = df.drop(columns=drop_cols).values.astype(np.float32)

    # 替换 inf / nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 分层 70:30 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=_TEST_RATIO, random_state=random_state, stratify=y
    )
    print(f"[NIDD5G] 训练集: {len(X_train)} 条，测试集: {len(X_test)} 条")

    # ANOVA F-score 特征选取（仅在训练集上 fit）
    selector = SelectKBest(f_classif, k=num_features)
    selector.fit(X_train, y_train)
    X_train = selector.transform(X_train).astype(np.float32)
    X_test = selector.transform(X_test).astype(np.float32)
    print(f"[NIDD5G] 已选取 Top-{num_features} 特征（ANOVA F-score）")

    # Z-score 归一化（仅用训练集统计量）
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    print("[NIDD5G] Z-score 归一化完成")

    return X_train, X_test, y_train, y_test


def _detect_label_col(df):
    """按优先级探测 DataFrame 中的标签列名。"""
    candidates = [_LABEL_COL, "Label", "Attack_type", "attack_type", "class", "Class"]
    for col in candidates:
        if col in df.columns:
            return col
    # 回退：取最后一列
    last_col = df.columns[-1]
    print(f"[NIDD5G] 警告：未找到已知标签列，回退到最后一列 '{last_col}'")
    return last_col
