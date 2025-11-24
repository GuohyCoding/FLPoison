"""
全局参数解析与预处理模块，负责加载 YAML 配置、命令行参数并协调攻击/防御实验设置。

主要功能:
    - 解析命令行选项，合并外部配置文件中的默认参数；
    - 根据不同实验模式（单次/benchmark）生成攻防组合；
    - 在运行前完成设备选择、输出路径创建等必要预处理工作。
"""

import os
import torch
import yaml
from aggregators import all_aggregators
from attackers import data_poisoning_attacks, model_poisoning_attacks
from fl.models import all_models
from fl.algorithms import all_algorithms
import argparse
from types import SimpleNamespace
from global_utils import frac_or_int_to_int


def read_args():
    """
    解析命令行参数并加载 YAML 配置，返回两个命名空间对象。

    参数:
        None
    返回:
        Tuple[SimpleNamespace, argparse.Namespace]:
            - args: 由配置文件构建的命名空间（若存在），否则为空命名空间；
            - cli_args: 原始命令行解析结果。
    异常:
        FileNotFoundError: 若指定的配置文件不存在。
        yaml.YAMLError: YAML 文件格式不正确时抛出。
    复杂度:
        时间复杂度 O(|yaml|)，空间复杂度 O(|yaml|)。

    费曼学习法:
        (A) 函数读取命令行参数并加载配置文件，形成可覆写的参数集。
        (B) 像科研助理先拿到实验说明书，再读取当班师兄写的补充笔记。
        (C) 步骤拆解:
            - 使用 argparse 定义所有可能的命令行选项。
            - 解析 CLI 参数，得到 `cli_args`。
            - 若提供 `-config`，调用 `read_yaml` 加载默认配置。
            - 返回两个命名空间，供后续覆盖与合并。
        (D) 示例:
            >>> args, cli = read_args()
            >>> cli.config
            './configs/FedSGD_MNIST_config.yaml'
        (E) 边界与测试:
            - 若命令行缺少 `-config`，程序将报错；需在测试时确保必需参数存在。
            - 建议测试: 1) 构造最小配置文件并运行解析； 2) 给定额外参数检查覆盖逻辑。
        (F) 背景参考:
            - 概念: 命令行参数与配置文件协同加载。
            - 参考书籍: 《Python 命令行与配置管理》。
    """
    parser = argparse.ArgumentParser(
        description="Poisoning attacks and defenses in Federated Learning")
    
    parser.add_argument('-config', '--config', type=str,
                        required=True, help='Path to the YAML configuration file')
    # command line arguments if provided
    parser.add_argument('-b', '--benchmark', default=False, type=bool,
                        help='Run all combinations of attacks and defenses')
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-seed', '--seed', type=int)
    parser.add_argument('-alg', '--algorithm', choices=all_algorithms)
    parser.add_argument('-opt', '--optimizer', choices=['SGD', 'Adam'],
                        help='optimizer for training')
    parser.add_argument('-lr_scheduler', '--lr_scheduler', type=str,
                        help='lr_scheduler for training')
    parser.add_argument('-milestones', '--milestones', type=int, nargs="+",
                        help='milestone for learning rate scheduler')
    parser.add_argument('-num_clients', '--num_clients', type=int,
                        help='number of participating clients')
    parser.add_argument('-bs', '--batch_size', type=int,
                        help='batch_size')
    parser.add_argument('-lr', '--learning_rate',
                        type=float, help='initial learning rate')
    parser.add_argument('-le', '--local_epochs', type=int,
                        help='local global_epoch')
    parser.add_argument('-model', '--model', choices=all_models)
    parser.add_argument('-data', '--dataset',
                        choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'CINIC10', 'CIFAR100', 'EMNIST'])
    parser.add_argument('-dtb', '--distribution',
                        choices=['iid', 'class-imbalanced_iid', 'non-iid', 'pat', 'imbalanced_pat'])
    parser.add_argument('-dirichlet_alpha', '--dirichlet_alpha', type=float,
                        help='smaller alpha for drichlet distribution, stronger heterogeneity, 0.1 0.5 1 5 10, normally use 0.5')
    parser.add_argument('-im_iid_gamma', '--im_iid_gamma', type=float,
                        help='smaller alpha for class imbalanced distribution, stronger heterogeneity, 0.05, 0.1, 0.5')

    # attacks and defenses settings
    all_attacks = ['NoAttack'] + \
        model_poisoning_attacks + data_poisoning_attacks
    
    parser.add_argument('-att', '--attack',
                        choices=all_attacks, help="Attacks options")
    parser.add_argument('-attack_start_epoch', '--attack_start_epoch',
                        type=int, help="the attack start epoch")
    parser.add_argument('-attparam', '--attparam', type=float,
                        help='scale for omniscient model poisoning attack, IPM,ALIE,MinMax,MinSum,Fang')
    parser.add_argument('-def', '--defense',
                        choices=all_aggregators, help="Defenses options")
    parser.add_argument('-num_adv', '--num_adv', type=float,
                        help='the proportion (float < 1) or number (int>1) of adversaries')
    parser.add_argument('-o', '--output', type=str,
                        help='output file for results')
    # poison settings
    parser.add_argument('-prate', '--poisoning_ratio',
                        help='poisoning portion (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--target_label', type=int,
                        help='The No. of target label for backdoored images (int, range from 0 to 10, default: 6)')
    parser.add_argument('--trigger_path', help='Trigger Path')
    parser.add_argument('--trigger_size', type=int,
                        help='Trigger Size (int, default: 5)')
    parser.add_argument('-gidx', '--gpu_idx', type=int, nargs="+",
                        help='Index of GPU (int, default: 3, choice: 0, 1, 2, 3...)')

    # override attack_params or defense_params with dict string
    parser.add_argument(
        '-defense_params', '--defense_params', type=str, help='Override defense parameters')
    parser.add_argument(
        '-attack_params', '--attack_params', type=str, help='Override attack parameters')
    cli_args = parser.parse_args()

    # load configurations from yaml file if provided
    args = SimpleNamespace()  # compatible with argparse.Namespace
    if cli_args.config:
        args = read_yaml(cli_args.config)
    return args, cli_args


def read_yaml(filename):
    """
    读取 YAML 文件并转换为 SimpleNamespace，便于属性式访问。

    参数:
        filename (str): YAML 配置文件路径。
    返回:
        SimpleNamespace: 包含配置键值对的命名空间对象。
    异常:
        FileNotFoundError: 文件路径无效。
        yaml.YAMLError: YAML 内容格式错误。
    复杂度:
        时间复杂度 O(|yaml|)，空间复杂度 O(|yaml|)。

    费曼学习法:
        (A) 函数将 YAML 文本转为可通过点号访问的对象。
        (B) 类比把说明书内容抄写到白板上，方便随手查看。
        (C) 步骤拆解:
            - 打开 YAML 文件并读出文本。
            - 调用 `yaml.load` 加载成 Python 字典。
            - 使用 SimpleNamespace 构造可属性访问的配置对象。
        (D) 示例:
            >>> cfg = read_yaml('./configs/FedSGD_MNIST_config.yaml')
        (E) 边界与测试:
            - 需确保 YAML 内没有自定义对象，避免 Loader 额外风险。
            - 建议测试: 1) 读取存在文件并断言字段; 2) 对缺失文件捕获异常。
        (F) 背景参考:
            - 概念: YAML 配置解析。
            - 参考书籍: 《Python YAML Cookbook》。
    """
    # read configurations from yaml file to args dict object
    with open(filename.strip(), 'r', encoding='utf-8') as file:
        args_dict = yaml.load(file, Loader=yaml.FullLoader)
    args = SimpleNamespace(**args_dict)
    return args


def override_args(args, cli_args):
    """
    用命令行参数覆盖配置文件，并保证攻防参数存在默认值。

    参数:
        args (SimpleNamespace): 从 YAML 加载的基础参数。
        cli_args (argparse.Namespace): 命令行解析结果。
    返回:
        SimpleNamespace: 更新后的参数对象。
    异常:
        AttributeError: 当配置缺失所需字段时可能抛出。
    复杂度:
        时间复杂度 O(|args|)，空间复杂度 O(1)。

    费曼学习法:
        (A) 函数把命令行输入与默认配置合并，确保攻防参数就绪。
        (B) 类比组委会先给出官方方案，再允许选手自带装备覆盖默认配置。
        (C) 步骤拆解:
            - 针对 attack/defense 参数缺失的情况，从列表中填充默认值。
            - 遍历命令行参数，对非空项覆盖 args 属性，并打印警告提示。
            - 单独处理 attack/defense 及其参数，优先取命令行指定，若未给则回退到默认项。
        (D) 示例:
            >>> cfg = override_args(cfg, cli_args)
        (E) 边界与测试:
            - 本函数通过 `eval` 动态访问属性，需确保输入经过验证（潜在风险）。
            - 建议测试: 1) 构造命令行对象覆盖部分字段； 2) 缺省 attack_params 时验证默认填充逻辑。
        (F) 背景参考:
            - 概念: 参数优先级管理。
            - 参考书籍: 《Configuration Management in Practice》。
    """
    # fill the attack and defense parameters with default
    for param_type in ['attack', 'defense']:
        if not hasattr(args, f"{param_type}_params"):
            # 使用 eval 获取列表并查找默认参数；需确保配置可信
            for i in eval(f"args.{param_type}s"):
                if i[param_type] == eval(f"args.{param_type}"):
                    setattr(args, f"{param_type}_params",
                            i.get(f'{param_type}_params'))
                    break

    # override parameters with command line inputs when available
    for key, value in vars(cli_args).items():
        if key in ['config', 'attack', 'defense', 'attack_params', 'defense_params']:
            continue
        if value is not None:
            setattr(args, key, value)
            print(f"Warning: Overriding {key} with {value}")

    # override attack, defense, attack_params, defense_params
    for param_type in ['attack', 'defense']:
        if eval(f"cli_args.{param_type}"):
            setattr(args, param_type, eval(f"cli_args.{param_type}"))
            if eval(f"cli_args.{param_type}_params"):
                setattr(args, f'{param_type}_params',
                        eval(f"cli_args.{param_type}_params"))
            else:
                for i in eval(f"args.{param_type}s"):
                    if i[param_type] == eval(f"args.{param_type}"):
                        setattr(args, f"{param_type}_params",
                                i.get(f"{param_type}_params"))
                        break
    return args


def benchmark_preprocess(args):
    """
    对基准实验模式下的所有攻防组合进行预处理调用。

    参数:
        args (SimpleNamespace): 包含攻防列表的配置对象。
    返回:
        None: 函数原地修改 args 并打印状态信息。
    异常:
        AttributeError: 当 args 缺少 attacks/defenses 字段时抛出。
    复杂度:
        时间复杂度 O(|attacks| × |defenses|)。

    费曼学习法:
        (A) 函数遍历所有攻击与防御组合，逐个调用单实验预处理。
        (B) 类似教练制定大合练计划：把每种进攻招式与每种防守策略都排练一遍。
        (C) 步骤拆解:
            - 遍历攻防列表，更新 args.attack/args.defense 及其参数。
            - 调用 `single_preprocess` 执行通用准备工作。
            - 若输出文件已存在，打印信息并跳过，以节省计算资源。
        (D) 示例:
            >>> benchmark_preprocess(args)
        (E) 边界与测试:
            - 若 args.output 已指向实际文件，需要确保逻辑正确跳过。
            - 建议测试: 1) 准备两种攻防组合验证循环； 2) 预先创建日志文件检测跳过行为。
        (F) 背景参考:
            - 概念: 基准实验设计与枚举。
            - 参考书籍: 《Experiments in Machine Learning》。
    """
    for attack_i in args.attacks:
        for defense_j in args.defenses:
            args.attack, args.attack_params = attack_i['attack'], attack_i.get(
                'attack_params')
            args.defense, args.defense_params = defense_j['defense'], defense_j.get(
                'defense_params')
            single_preprocess(args)
            if os.path.exists(args.output):
                print(f"File {args.output.split('/')[-1]} exists, skip")
                continue
            print(
                f"Running {args.attack} with {args.defense} under {args.distribution}")


def single_preprocess(args):
    """
    针对单个攻防组合执行预处理，包括数据配置加载与设备选择。

    参数:
        args (SimpleNamespace): 当前组合的参数对象。
    返回:
        SimpleNamespace: 更新后的参数对象。
    异常:
        KeyError: dataset_config 中缺失对应数据集键时抛出。
        ValueError: 当 GPU 索引非法时可能抛出。
    复杂度:
        时间复杂度 O(|dataset_config|)，空间复杂度 O(1)。

    费曼学习法:
        (A) 函数为单次实验设定数据统计、设备与输出路径。
        (B) 类比运动员上场前的准备：先熟悉场地参数，再确认穿什么装备和把成绩记在哪。
        (C) 步骤拆解:
            - 读取数据集配置文件，将均值、方差等统计写入 args。
            - 根据硬件可用性选择 CUDA > MPS > CPU 的顺序设置 device。
            - 使用 `frac_or_int_to_int` 将攻击者比例转换为整数数量。
            - 确保攻防参数字段存在，即便为空也要设置为 None。
            - 构建默认输出日志路径并创建目录。
        (D) 示例:
            >>> single_preprocess(args)
            >>> args.device
            device(type='cuda', index=0)
        (E) 边界与测试:
            - 如果 `args.gpu_idx` 未设置或为空列表，会导致索引错误，需要在测试里检查。
            - 建议测试: 1) 在 GPU/MPS/CPU 不同环境下验证设备选择； 2) 模拟已存在输出文件确保目录创建正确。
        (F) 背景参考:
            - 概念: 实验前参数归一化与设备分配。
            - 参考书籍: 《Federated Learning》实验实践章节。
    """
    # load dataset configurations, also include learning rate and epochs
    with open("./configs/dataset_config.yaml", 'r', encoding='utf-8') as file:
        dataset_config = yaml.load(file, Loader=yaml.FullLoader)
    for key, value in dataset_config[args.dataset].items():
        if key in ['mean', 'std']:
            # mean/std 在 YAML 中以字符串存储，这里解析成数值列表/元组
            value = eval(value)  # eval() 会把传入的字符串当作一段 Python 表达式来执行，并返回结果
        # 把每个配置条目动态挂到 args 上，后续训练流程可以通过 args.mean、args.image_size 等属性读取
        setattr(args, key, value)

    # preprocess the arguments
    # Priority: CUDA > MPS (MacOS) > CPU
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_idx[0]}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    args.device = device
    args.num_adv = frac_or_int_to_int(args.num_adv, args.num_clients)

    # ensure attack_params and defense_params attributes exist. when there is no params, set it to None.
    ensure_attr(args, 'attack_params')
    ensure_attr(args, 'defense_params')

    # generate output path if not provided
    args.output = f'./logs/{args.algorithm}/{args.dataset}_{args.model}/{args.distribution}/{args.dataset}_{args.model}_{args.distribution}_{args.attack}_{args.defense}_{args.epochs}_{args.num_clients}_{args.learning_rate}_{args.algorithm}.txt'

    # check output path, if exists, skip, otherwise create the directories
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    return args


def ensure_attr(obj, attr_name):
    """
    保证对象存在指定属性，若缺失则补充为 None。

    参数:
        obj (Any): 待检查的对象。
        attr_name (str): 属性名称。
    返回:
        None
    复杂度:
        时间复杂度 O(1)。

    费曼学习法:
        (A) 函数防止访问不存在的属性，引入默认值。
        (B) 类比检查实验仪器是否装上必要的安全保护罩，没有就立即安装一个。
        (C) 步骤拆解:
            - 判断对象是否已有该属性。
            - 若没有，则设置为 None。
        (D) 示例:
            >>> ensure_attr(args, 'attack_params')
        (E) 边界与测试:
            - 若对象不支持 setattr，会抛出 AttributeError。
            - 建议测试: 1) 对缺失属性的对象调用一次； 2) 对已有属性的对象确保值不被覆盖。
        (F) 背景参考:
            - 概念: 防御性编程。
            - 参考书籍: 《Writing Robust Python》。
    """
    if not hasattr(obj, attr_name):
        setattr(obj, attr_name, None)


# __AI_ANNOTATION_SUMMARY__
# - read_args: 解析命令行与 YAML 配置，返回基础参数与 CLI 参数命名空间。
# - read_yaml: 读取 YAML 文件并包装为 SimpleNamespace 以便属性访问。
# - override_args: 合并命令行输入与默认配置，补全攻防参数。
# - benchmark_preprocess: 遍历所有攻防组合并执行单次预处理。
# - single_preprocess: 为当前组合设定数据统计、设备与输出目录。
# - ensure_attr: 确保对象存在特定属性，缺失时填充默认值。
