from global_utils import import_all_modules, Register
import os

# 初始化算法注册表，供各类联邦算法通过装饰器自动登记。
algorithm_registry = Register()
# import all files in the directory, so that the registry decorator can be read and used
# os.path.dirname(__file__) get the current directory path
import_all_modules(os.path.dirname(__file__), 1, "fl")
# 收集已注册算法名称，便于查看或外部使用。
all_algorithms = list(algorithm_registry.keys())


def get_algorithm_handler(name):
    """根据算法名称获取已注册的算法实现。

    概述:
        利用全局注册表 `algorithm_registry` 返回对应算法对象（通常是类或函数）。

    参数:
        name (str): 注册时使用的算法名称键。

    返回:
        Any: 对应的算法处理器（类/函数）。

    异常:
        KeyError: 当名称不存在于注册表时抛出。

    复杂度:
        O(1)，字典查找。

    费曼学习法:
        (A) 函数像是在“算法通讯录”里按名字查找联系人。
        (B) 类比拿起电话簿找某位教授的联系方式。
        (C) 步骤拆解:
            1. 使用名称在注册表字典中索引。
            2. 返回对应的算法实现。
        (D) 示例:
            >>> handler = get_algorithm_handler('FedAvg')
        (E) 边界条件与测试建议: 名称必须在注册表中；建议测试未注册名称时是否抛出 KeyError。
        (F) 背景参考: 注册表模式（Registry Pattern）在插件化架构中的常见应用。
    """
    return algorithm_registry[name]


# __AI_ANNOTATION_SUMMARY__
# 全局变量 algorithm_registry/all_algorithms: 初始化注册表并列出已登记算法。
# 函数 get_algorithm_handler: 按名称从注册表检索算法实现。
