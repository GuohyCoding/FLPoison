"""
全局通用工具集合，涵盖日志管理、模块动态导入、注册机制、参数打印、随机数种子设置、
数值转换以及函数运行时间记录等功能。

设计目的:
    - 统一项目中的常用工具函数，减少重复实现；
    - 提供装饰器与注册器，支持模块化扩展；
    - 支撑联邦学习实验的日志、随机性控制与性能分析。
"""

import time
from functools import wraps
import importlib
import os
import logging
import random
import numpy as np
import torch


def setup_logger(logger_name, log_file, level=logging.INFO, stream=False):
    """
    创建并配置日志记录器，可选文件与终端输出。

    参数:
        logger_name (str): 日志记录器名称。
        log_file (str): 日志文件路径。
        level (int): 日志等级，默认 INFO。
        stream (bool): 是否同时输出到标准流。
    返回:
        logging.Logger: 已配置的日志记录器实例。
    异常:
        OSError: 当日志目录创建失败时抛出。
    复杂度:
        时间复杂度 O(1)，空间复杂度 O(1)。

    费曼学习法:
        (A) 函数为实验创建专属日志记录器，可写入文件并视需要打印到屏幕。
        (B) 类比实验室里架设一台录音设备，既保存记录又可现场播报。
        (C) 步骤拆解:
            - 获取或创建指定名称的 logger。
            - 设置格式器并创建文件句柄，如有必要再添加流句柄。
            - 返回配置完成的 logger。
        (D) 示例:
            >>> logger = setup_logger("client", "./logs/client.log")
            >>> logger.info("start")
        (E) 边界与测试:
            - 若 log_file 无法创建目录，会抛出异常，测试时需提供可写路径。
            - 建议测试: 1) 写入文件并检查内容； 2) 设置 stream=True 验证终端输出。
        (F) 背景参考:
            - 概念: Python logging 模块。
            - 参考书籍: 《Python Cookbook》日志章节。
    """
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fileHandler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fileHandler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(fileHandler)

    if stream:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        logger.addHandler(streamHandler)

    return logger


def actor(category, *attributes):
    """
    类装饰器，用于标记客户端/攻击者类别及其属性。

    参数:
        category (str): 类别名称，如 'benign' 或 'attacker'。
        *attributes (str): 该类别的具体属性，如 'non_omniscient'。
    返回:
        Callable: 装饰器函数，装饰目标类并注入类别信息。
    异常:
        ValueError: 当属性不在预定义集合内时抛出。
    复杂度:
        时间复杂度 O(1)。

    费曼学习法:
        (A) 装饰器为类打上角色标签，方便框架识别不同客户端类型。
        (B) 类比给运动员发徽章：国家队/教练组，通过徽章识别职责。
        (C) 步骤拆解:
            - 校验传入属性是否属于合法子类目。
            - 在类上挂载 `_category`、`_attributes` 并包装 `__init__`，确保实例拥有标签。
            - 避免重复装饰，防止继承链中多次赋值。
        (D) 示例:
            >>> @actor("attacker", "non_omniscient")
            ... class MaliciousClient(Client):
            ...     pass
        (E) 边界与测试:
            - 若继承链复杂，需确保 `_decorated` 标记有效。
            - 建议测试: 1) 装饰善意/恶意类并检查属性； 2) 子类继承确认标签不会重复写入。
        (F) 背景参考:
            - 概念: 类装饰器与角色元数据。
            - 参考书籍: 《Python Decorators》。
    """

    def decorator(cls):
        # key is the actor, value is the attributes of the actor
        categories = {"benign": ['always', 'temporary'],
                      "attacker": ['data_poisoning', 'model_poisoning', "non_omniscient", "omniscient"]}

        if not set(attributes).issubset(set(categories[category])):
            raise ValueError(
                "Invalid sub-category. Please change or add the sub-category.")
        cls._category = category
        cls._attributes = attributes
        # change __init__ method to realize it in objects
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            # 如果对象尚未标记，则写入类别与属性，避免继承导致重复装饰
            if not hasattr(self, "_decorated"):
                self.category = cls._category
                self.attributes = cls._attributes
                # Mark self object as decorated to prevent re-decoration on inherited classes
                self._decorated = True
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls
    return decorator


def import_all_modules(current_dir, depth=0, depth_prefix=None):
    """
    动态导入当前目录下的所有 Python 模块，支持递归前缀。

    参数:
        current_dir (str): 目标目录。
        depth (int): 当前递归深度，用于确定包名前缀。
        depth_prefix (str): 父级包名，用于嵌套包导入。
    返回:
        None
    异常:
        ImportError: 当模块导入失败时。
    复杂度:
        时间复杂度 O(#files)。

    费曼学习法:
        (A) 函数批量导入目录中的模块，确保注册逻辑自动执行。
        (B) 类比打开抽屉把里面的工具全拿出来摆上桌子，后续随时可用。
        (C) 步骤拆解:
            - 计算包名前缀，处理根目录与子包差异。
            - 遍历目录，过滤掉 `__init__.py` 与非 Python 文件。
            - 使用 `importlib.import_module` 按包路径导入模块。
        (D) 示例:
            >>> import_all_modules("./attackers", depth=1, depth_prefix="attackers")
        (E) 边界与测试:
            - 需保证目录结构对应包结构，否则会导入失败。
            - 建议测试: 1) 对含多个模块目录执行一次； 2) 捕获导入异常检查报错信息。
        (F) 背景参考:
            - 概念: Python 动态模块加载。
            - 参考书籍: 《Fluent Python》模块章节。
    """
    pkg_name = depth_prefix + "." + os.path.basename(
        current_dir) if depth else os.path.basename(current_dir)

    for filename in os.listdir(current_dir):
        # filter our __init__.py and non-python files
        if filename.endswith(".py") and (filename != "__init__.py"):
            module_name = filename[:-3]  # remove ".py"
            importlib.import_module(
                f".{module_name}", package=pkg_name)


class Register(dict):
    """
    通用注册器，实现字典式存储与装饰器注册双重功能。

    使用方式:
        - 直接当作字典读取已注册项；
        - 通过 `@registry` 或 `@registry("alias")` 装饰函数，实现自动注册。
    """

    def __init__(self, *args, **kwargs):
        """
        初始化字典基类，保持与普通 dict 行为一致。

        参数:
            *args, **kwargs: 传给 dict 的初始化参数。
        复杂度:
            时间复杂度 O(n) 取决于初始数据。
        """
        # init the dict class, so that it can be used as a normal dict
        super().__init__(*args, **kwargs)

    def __call__(self, target):
        """
        将被装饰对象注册到字典中，支持字符串别名与直接函数注册。

        参数:
            target (Union[str, Callable]): 装饰器参数或待注册函数。
        返回:
            Callable: 装饰器或原函数本身。
        异常:
            TypeError: 当 target 类型不被支持时（当前实现仅处理 str 或 callable）。
        复杂度:
            单次注册 O(1)。

        费曼学习法:
            (A) 函数让我们用装饰器方式把函数注册到全局表里。
            (B) 类比俱乐部门口的签到簿，既可以写真名也可以留外号。
            (C) 步骤拆解:
                - 定义内部函数 `register_item`，负责将名字与函数写入字典。
                - 若 target 为字符串，返回装饰器等待函数对象。
                - 若 target 本身可调用，则直接注册并返回该函数。
            (D) 示例:
                >>> registry = Register()
                >>> @registry
                ... def foo(): pass
            (E) 边界与测试:
                - 如果重复注册同名函数，后者会覆盖前者，需要在测试里确认行为。
                - 建议测试: 1) 使用默认函数名注册； 2) 使用别名注册并检查字典键。
            (F) 背景参考:
                - 概念: 注册器模式。
                - 参考书籍: 《Design Patterns in Python》。
        """

        def register_item(name, func):
            self[name] = func
            return func

        # if target is a string, return a function to receive the callable object. @register('name')
        if isinstance(target, str):
            ret_func = (lambda x: register_item(target, x))
        # if target is a callable object, then register it, and return it, @register
        elif callable(target):
            ret_func = register_item(target.__name__, target)
        else:
            ret_func = target  # 保持原始行为，避免改变调用方假设
        return ret_func


def print_filtered_args(args, logger):
    """
    打印筛选后的配置参数，过滤敏感或冗余字段。

    参数:
        args (Namespace): 参数命名空间。
        logger (logging.Logger): 日志记录器。
    返回:
        None
    复杂度:
        时间复杂度 O(n)。

    费曼学习法:
        (A) 函数把核心配置打印出来，方便追踪实验设定。
        (B) 类比发布会前列出关键信息，不提内部实现细节。
        (C) 步骤拆解:
            - 将命名空间转换为字典。
            - 过滤掉 attacks、defenses、logger 等字段。
            - 拼接成字符串输出到日志。
        (D) 示例:
            >>> print_filtered_args(args, logger)
        (E) 边界与测试:
            - 若 logger 未配置，将不会输出，需要在测试中确认。
            - 建议测试: 1) 检查过滤列表是否生效； 2) 打印结果格式是否符合预期。
        (F) 背景参考:
            - 概念: 日志输出格式化。
    """
    args_dict = vars(args)
    filtered_args = {k: v for k, v in args_dict.items() if k not in [
        'attacks', 'defenses', 'logger']}
    msg = ', '.join([f'{key}: {value}' for key,
                     value in filtered_args.items()]) + '\n'
    logger.info(msg)


def avg_value(x):
    """
    计算数值序列的平均值。

    参数:
        x (Sequence[float]): 待求平均的序列。
    返回:
        float: 平均值。
    异常:
        ZeroDivisionError: 当序列为空时抛出。
    复杂度:
        时间复杂度 O(len(x))。

    费曼学习法:
        (A) 函数求一个列表的平均数。
        (B) 类比把考试成绩总分除以科目数。
        (C) 步骤拆解:
            - 调用 `sum` 求和。
            - 除以长度得到平均值。
        (D) 示例:
            >>> avg_value([1, 2, 3])
            2.0
        (E) 边界与测试:
            - 输入空列表会报错，调用前需检查。
            - 建议测试: 1) 正常序列； 2) 单元素序列。
        (F) 背景参考:
            - 概念: 基础统计量。
    """
    return sum(x) / len(x)


def setup_seed(seed):
    """
    固定随机种子，最大程度减少实验中的随机性。

    参数:
        seed (int): 随机种子。
    返回:
        None
    复杂度:
        时间复杂度 O(1)。

    费曼学习法:
        (A) 函数把 PyTorch、NumPy、Python 等随机源都设为同一 seed。
        (B) 像给所有投篮训练设置同样的初始姿势，结果更可复现。
        (C) 步骤拆解:
            - 设置 torch、torch.cuda、torch.cuda_all、numpy、random 的种子。
            - 通过环境变量固定 Python 哈希行为。
        (D) 示例:
            >>> setup_seed(42)
        (E) 边界与测试:
            - 某些库（如 cuDNN）可能仍存在非确定性，需要额外设置。
            - 建议测试: 1) 连续运行同一脚本观察随机输出； 2) 结合 torch.backends.cudnn 设置进一步保证。
        (F) 背景参考:
            - 概念: 随机性控制与实验复现。
            - 参考书籍: 《Reproducible Machine Learning》。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def frac_or_int_to_int(frac_or_int, total_num):
    """
    将比例或整数转换为具体数量，用于计算攻击者数量等场景。

    参数:
        frac_or_int (float): 若 >=1 表示整数数量；若 <1 表示比例。
        total_num (int): 总体数量，用于比例换算。
    返回:
        int: 对应的整数数量。
    复杂度:
        时间复杂度 O(1)。

    费曼学习法:
        (A) 函数把“比例或人数”统一转换成整数人数。
        (B) 类比抽签决定多少名选手参加：直接给人数或用比例都行。
        (C) 步骤拆解:
            - 若输入 >=1 视为整数，直接取整。
            - 否则视为比例，乘以总人数后取整。
        (D) 示例:
            >>> frac_or_int_to_int(0.2, 10)
            2
        (E) 边界与测试:
            - 当比例极小可能取整为 0，需要在调用方考虑下限。
            - 建议测试: 1) 输入整数与比例分别验证； 2) 检查 total_num=0 的异常情况。
        (F) 背景参考:
            - 概念: 比例转数量。
    """
    return int(frac_or_int) if frac_or_int >= 1 else int(frac_or_int * total_num)


class TimingRecorder:
    """
    函数运行时间记录器，可用于客户端与服务器性能分析。

    功能:
        - 通过装饰器记录函数调用时长与调用次数；
        - 支持按 epoch 条件触发日志输出；
        - 计算平均运行时间并写入日志。
    """

    def __init__(self, id, output_file):
        """
        初始化 TimingRecorder，配置日志输出路径与记录结构。

        参数:
            id (int): 记录器所属对象标识，如客户端编号。
            output_file (str): 原始输出日志路径，用于派生时间日志位置。
        返回:
            None
        复杂度:
            时间复杂度 O(1)。

        费曼学习法:
            (A) 构造函数设定记录器身份与日志文件。
            (B) 像在训练场准备计时器和成绩表。
            (C) 步骤拆解:
                - 根据 output_file 派生时间日志路径（logs -> logs/time_logs）。
                - 初始化记录字典与 logger，准备写入数据。
                - 配置需要记录的 epoch 列表（默认关闭）。
            (D) 示例:
                >>> timer = TimingRecorder(1, "./logs/client.txt")
            (E) 边界与测试:
                - 若 output_file 路径不存在，会自动创建目录；需在测试中验证。
                - 建议测试: 1) 检查日志文件是否生成； 2) 验证 record_epochs 列表可配置。
            (F) 背景参考:
                - 概念: 性能测量与日志记录。
                - 参考书籍: 《High Performance Python》。
        """
        self.id = id
        # record the duration and number of call of func
        self.global_timings = {}
        time_log_path = output_file.replace(
            "logs/", "logs/time_logs/", 1)[:-4]+'.log'
        self.logger = setup_logger(
            __name__, time_log_path, level=logging.INFO)
        self.client_log_flag = False
        epoch_level = False
        self.record_epochs = [2, 4, 6, 8, 10, 20,
                              50, 100, 150, 200] if epoch_level else []

    def timing_decorator(self, func):
        """
        装饰器：记录函数执行时间并按需输出日志。

        参数:
            func (Callable): 被装饰的函数。
        返回:
            Callable: 包裹后的函数，执行时记录耗时。
        复杂度:
            单次调用时间复杂度与原函数相同，记录开销 O(1)。

        费曼学习法:
            (A) 通过装饰器给函数加计时器。
            (B) 类比运动员佩戴计时表，每次训练自动记下用时。
            (C) 步骤拆解:
                - 记录开始与结束时间，计算持续时间。
                - 将结果累计到 `global_timings` 中。
                - 根据配置决定是否 log 当前统计。
            (D) 示例:
                >>> @timer.timing_decorator
                ... def train(): pass
            (E) 边界与测试:
                - 若函数频繁调用需关注日志量。
                - 建议测试: 1) 对简单函数装饰并验证时间记录； 2) 检查 record_epochs 行为。
            (F) 背景参考:
                - 概念: 装饰器与性能分析。
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()  # start timer
            result = func(*args, **kwargs)  # call the function
            end_time = time.time()  # end timer
            duration = end_time - start_time

            # update the global_timings with the duration
            method_name = func.__name__
            if method_name not in self.global_timings:
                self.global_timings[method_name] = {
                    "total_time": 0, "calls": 0}
            self.global_timings[method_name]["total_time"] += duration
            self.global_timings[method_name]["calls"] += 1

            if self.client_log_flag:
                # log data during training
                self.report(f"Worker ID {self.id}")

            # for client
            self.client_log_flag = True if method_name == "local_training" and self.global_timings[
                method_name]["calls"] in self.record_epochs else False

            if method_name == "aggregation" and self.global_timings[method_name]["calls"] in self.record_epochs:
                # log data during training
                self.report(f"Worker ID {self.id}")
            return result

        return wrapper

    def get_average_time(self, func_name):
        """
        计算指定函数的平均执行时间。

        参数:
            func_name (str): 函数名称。
        返回:
            float: 平均耗时（秒），若未记录返回 0。

        费曼学习法:
            (A) 函数求某方法的平均执行时间。
            (B) 类比记录多次跑步的平均速度。
            (C) 步骤拆解:
                - 查找 `global_timings` 字典。
                - 若存在，使用总时间除以调用次数。
            (D) 示例:
                >>> timer.get_average_time("local_training")
            (E) 边界与测试:
                - 当函数尚未调用时返回 0，需在测试里覆盖。
            (F) 背景参考:
                - 概念: 平均性能指标计算。
        """
        if func_name in self.global_timings:
            total_time = self.global_timings[func_name]["total_time"]
            calls = self.global_timings[func_name]["calls"]
            return total_time / calls if calls > 0 else 0
        return 0

    def report(self, id=None):
        """
        将所有记录的平均耗时写入日志。

        参数:
            id (str, optional): 日志前缀标识，如 Worker ID。
        返回:
            None

        费曼学习法:
            (A) 函数遍历所有统计并输出平均时间与调用次数。
            (B) 类比教练公布运动员训练时长榜。
            (C) 步骤拆解:
                - 遍历 `global_timings`，计算平均时间。
                - 使用 logger 写入标准化日志。
            (D) 示例:
                >>> timer.report("Worker ID 1")
            (E) 边界与测试:
                - 若无记录将输出空内容，可在测试前确认。
            (F) 背景参考:
                - 概念: 性能日志报告。
        """
        for method_name, stats in self.global_timings.items():
            avg_time = stats["total_time"] / stats["calls"]
            self.logger.info(
                f"{id}, {method_name} averge time: {avg_time:.6f} s, call time: {stats['calls']}")


# __AI_ANNOTATION_SUMMARY__
# - setup_logger: 创建支持文件/终端输出的日志记录器。
# - actor: 类装饰器，为客户端/攻击者标记类别与属性。
# - import_all_modules: 批量导入目录下模块，触发注册。
# - Register: 字典式注册器，支持别名与装饰器注册。
# - print_filtered_args: 打印筛选后的配置参数。
# - avg_value: 计算数值序列平均值。
# - setup_seed: 固定随机种子以提升实验可复现性。
# - frac_or_int_to_int: 将比例或整数转换为实际数量。
# - TimingRecorder: 统计被装饰函数的执行时间与调用次数。
