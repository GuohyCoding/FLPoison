"""
绘图与日志解析工具，支持从训练日志中提取指标、绘制准确率/ASR 曲线以及展示客户端标签分布。

主要功能:
    - `parse_logs`: 基于正则表达式解析联邦训练日志中的各类指标；
    - `plot_accuracy`: 从日志绘制准确率与攻击成功率曲线；
    - `plot_label_distribution`: 可视化客户端本地数据的标签分布情况。
"""

import re
import matplotlib.pyplot as plt
import numpy as np


def parse_logs(filename):
    """
    从训练日志文件中解析 epoch、准确率、损失等指标。

    参数:
        filename (str): 日志文件路径，通常由 `fl_run` 生成。
    返回:
        Tuple[List[int], List[float], List[float], List[Optional[float]], List[Optional[float]]]:
            分别为 epoch 序列、测试准确率、测试损失、ASR、ASR 损失（若无则为 None）。
    异常:
        FileNotFoundError: 当日志文件不存在时抛出。
        re.error: 正则表达式无法匹配时可能抛出。
    复杂度:
        时间复杂度 O(|file|)，空间复杂度 O(#records)。

    费曼学习法:
        (A) 函数读取日志文本并提取每轮训练的关键指标。
        (B) 类比实验助手从笔记中抄出每次实验成绩，整理成表格。
        (C) 步骤拆解:
            - 清空当前 Matplotlib 图（与后续绘图共用状态）。
            - 读取日志文件内容。
            - 使用正则表达式按行匹配 epoch、Test Acc、Test loss 以及可选的 ASR 指标。
            - 将提取的字符串转换为整数/浮点数，并保存在列表中。
        (D) 示例:
            >>> epochs, accs, losses, asrs, asr_losses = parse_logs("logs/example.txt")
        (E) 边界与测试:
            - 若日志格式发生改变（例如键名不同），正则需同步调整。
            - 建议测试: 1) 使用包含 ASR 的日志校验可选字段解析； 2) 使用无 ASR 记录的日志验证 None 填充逻辑。
        (F) 背景参考:
            - 概念: 正则表达式解析结构化文本。
            - 参考书籍: 《Mastering Regular Expressions》。
    """
    plt.clf()
    # read log file
    with open(filename, 'r', encoding="utf-8", errors="replace") as f:
        content = f.read()
    epochs, accs, losses, asrs, asr_losses = [], [], [], [], []
    # regular expression pattern to extract epoch, test accuracy, test loss, asr, asr loss
    regex = (
        r"Epoch (?P<epoch>\d+)\s.*?Test Acc: (?P<test_acc>[\d\.]+)\s.*?Test loss: (?P<test_loss>[\d\.]+)(?:\s.*?ASR: (?P<asr>[\d\.]+))?(?:\s.*?ASR loss: (?P<asr_loss>[\d\.]+))?"
    )

    for match in re.finditer(regex, content):
        epochs.append(int(match.group('epoch')))
        accs.append(float(match.group('test_acc')))
        losses.append(float(match.group('test_loss')))

        # if asr and asr loss exist, add them, or add None
        asr = match.group('asr')
        asr_loss = match.group('asr_loss')
        asrs.append(float(asr) if asr else None)
        asr_losses.append(float(asr_loss) if asr_loss else None)

    return epochs, accs, losses, asrs, asr_losses


def plot_accuracy(filename):
    """
    绘制测试准确率与攻击成功率（若存在）随 epoch 变化的曲线。

    参数:
        filename (str): 输入日志文件路径，同名 PNG 图将保存在相同目录。
    返回:
        None
    异常:
        FileNotFoundError: 当日志文件不存在时抛出。
        ValueError: 当解析结果为空时 Matplotlib 可能报错。
    复杂度:
        时间复杂度 O(#epochs)，空间复杂度 O(#epochs)。

    费曼学习法:
        (A) 函数读取日志数据并绘制准确率/ASR 随训练轮次的变化曲线。
        (B) 类比教练把每场比赛的得分画成折线图，一目了然地看到趋势。
        (C) 步骤拆解:
            - 调用 `parse_logs` 获取指标序列。
            - 绘制准确率曲线；若存在 ASR，使用虚线绘制对比。
            - 设置坐标轴、标题、图例与网格线。
            - 保存为与日志同名不同后缀的 PNG 图像。
        (D) 示例:
            >>> plot_accuracy("./logs/demo.txt")  # 生成 demo.png
        (E) 边界与测试:
            - 若 `parse_logs` 返回的 epoch 列表为空，绘图前需检查避免空图。
            - 建议测试: 1) 验证无 ASR 时仅绘制准确率； 2) 验证带 ASR 日志生成双曲线图。
        (F) 背景参考:
            - 概念: Matplotlib 折线图绘制。
            - 参考书籍: 《Python Data Visualization Cookbook》。
    """
    epochs, accs, _, asr, _ = parse_logs(filename)

    plt.plot(epochs, accs, label='Accuracy')

    # if asr statistics exists, plot asr curve
    if any(asr):
        plt.plot(epochs, asr, label='ASR', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename[:-4] + ".png")


def plot_label_distribution(train_data, client_idcs, n_clients, dataset, distribution):
    """
    可视化各客户端本地训练样本的标签分布情况。

    参数:
        train_data (Dataset): 训练数据集对象，需包含 `targets`/`classes` 属性。
        client_idcs (List[List[int]]): 每个客户端对应的数据索引列表。
        n_clients (int): 客户端数量。
        dataset (str): 数据集名称（用于标题展示）。
        distribution (str): 数据划分策略，用于确定标题补充信息。
    返回:
        None: 保存生成的标签分布 PDF 图。
    异常:
        AttributeError: 当数据集缺少 `targets` 或 `classes` 属性时。
        ValueError: 当分布类型不在预设字典中时。
    复杂度:
        时间复杂度 O(|train_data|)，空间复杂度 O(n_clients × n_classes)。

    费曼学习法:
        (A) 函数统计每个客户端持有的标签数量并绘制堆叠直方图。
        (B) 类比教师统计各班同学选修课程的分布，用图表展示偏好。
        (C) 步骤拆解:
            - 根据分布类型构造图表标题，并设置 Matplotlib 全局样式。
            - 依据 `xy_type` 将数据按「每个标签对应的客户端」或相反方式组织。
            - 调用 `plt.hist` 生成堆叠直方图，设置刻度、标签、标题与网格。
            - 保存为 PDF 文件，便于打印或嵌入报告。
        (D) 示例:
            >>> plot_label_distribution(train_dataset, client_indices, 10, "CIFAR10", "iid")
        (E) 边界与测试:
            - 当客户端数量较多时需调整字体与旋转角度，避免刻度重叠。
            - 建议测试: 1) 使用均匀分布与非均匀分布数据验证可视化结果； 2) 检查保存路径是否生成。
        (F) 背景参考:
            - 概念: 数据分布可视化与直方图。
            - 参考书籍: 《Data Visualization with Python》。
    """
    titleid_dict = {"iid": "Balanced_IID", "class-imbalanced_iid": "Class-imbalanced_IID",
                    "non-iid": "Quantity-imbalanced_Dirichlet_Non-IID", "pat": "Balanced_Pathological_Non-IID", "imbalanced_pat": "Quantity-imbalanced_Pathological_Non-IID"}
    dataset = "CIFAR-10" if dataset == "CIFAR10" else dataset
    title_id = dataset + " " + titleid_dict[distribution]
    xy_type = "client_label"  # 'label_client'
    plt.rcParams['font.size'] = 14  # set global fontsize
    # set the direction of xtick toward inside
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    labels = train_data.targets
    n_classes = labels.max()+1
    plt.figure(figsize=(12, 8))
    if xy_type == "client_label":
        label_distribution = [[] for _ in range(n_classes)]
        for c_id, idc in enumerate(client_idcs):
            for idx in idc:
                label_distribution[labels[idx]].append(c_id)

        plt.hist(label_distribution, stacked=True,
                 bins=np.arange(-0.5, n_clients + 1.5, 1),
                 label=range(n_classes), rwidth=0.5, zorder=10)
        plt.xticks(np.arange(n_clients), ["%d" %
                                          c_id for c_id in range(n_clients)])
        plt.xlabel("Client ID", fontsize=20)
    elif xy_type == "label_client":
        plt.hist([labels[idc]for idc in client_idcs], stacked=True,
                 bins=np.arange(min(labels)-0.5, max(labels) + 1.5, 1),
                 label=["Client {}".format(i) for i in range(n_clients)],
                 rwidth=0.5, zorder=10)
        plt.xticks(np.arange(n_classes), train_data.classes)
        plt.xlabel("Label type", fontsize=20)

    plt.ylabel("Number of Training Samples", fontsize=20)
    plt.title(f"{title_id} Label Distribution Across Clients", fontsize=20)
    rotation_degree = 45 if n_clients > 30 else 0
    plt.xticks(rotation=rotation_degree, fontsize=16)
    plt.legend(loc="best", prop={'size': 12}).set_zorder(100)
    plt.grid(linestyle='--', axis='y', zorder=0)
    plt.tight_layout()
    plt.savefig(f"./logs/{title_id}_label_dtb.pdf",
                format='pdf', bbox_inches='tight')


if __name__ == "__main__":
    # 示例：在直接运行该文件时绘制默认日志的准确率曲线
    plot_accuracy(
        "./logs/FedOpt/MNIST_lenet/iid/MNIST_lenet_iid_DBA_DeepSight_500_50_0.01_FedOpt.txt")


# __AI_ANNOTATION_SUMMARY__
# - parse_logs: 解析日志文本，提取 epoch、准确率、损失与 ASR 指标。
# - plot_accuracy: 根据日志绘制准确率与攻击成功率曲线并保存图像。
# - plot_label_distribution: 可视化各客户端的数据标签分布状况。
