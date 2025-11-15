"""
联邦学习聚合器模块的注册入口。

本文件负责在包导入时自动加载当前目录下的所有聚合策略模块，
并提供面向外部的统一检索接口，以便依据名称获取对应的聚合器实现。
"""

from global_utils import import_all_modules, Register
import os

aggregator_registry = Register()
# 创建一个全局注册表，用于记录所有被 @aggregator_registry.register 装饰的聚合器类/函数。

# import all files in the directory, so that the registry decorator can be read and used

# os.path.dirname(__file__) get the current directory path
# 主动导入当前包下的全部子模块，确保注册装饰器在模块导入时执行。
import_all_modules(os.path.dirname(__file__))
all_aggregators = list(aggregator_registry.keys())


def get_aggregator(name):
    """
    根据名称检索并返回已注册的聚合器实现。

    参数:
        name (str): 聚合器在注册表中的唯一字符串标识。

    返回:
        typing.Any: 与给定名称对应的聚合器类或可调用对象，由注册表返回。

    异常:
        KeyError: 当 name 未在注册表中出现时抛出。

    复杂度:
        时间复杂度 O(1)；空间复杂度 O(1)。
    """
    # 直接使用注册表的字典接口检索目标聚合器；Register 内部维护了名称到实现的映射。
    return aggregator_registry[name]


# 费曼学习法解释（get_aggregator）
# (A) 功能概述：get_aggregator 接收聚合器名称并返回预先注册好的聚合策略实现。
# (B) 类比说明：这像在图书馆前台报出书名，管理员会从索引卡片中迅速找到那本书交给你。
# (C) 逐步拆解：
#     1. 接收调用者传入的字符串名称——因为我们需要知道要取哪一个聚合器。
#     2. 读取全局的 aggregator_registry——这里存放了名称与聚合器实现之间的映射，相当于图书馆的索引。
#     3. 用名称在映射表中查找对应的实现并返回——避免重复创建对象，直接复用已登记的策略。
# (D) 最小示例：
#     >>> from aggregators import get_aggregator, aggregator_registry
#     >>> aggregator_registry.register("demo")(lambda updates: sum(updates))
#     >>> demo_agg = get_aggregator("demo")
#     >>> demo_agg([1, 2, 3])
#     6
# (E) 边界条件与测试建议：
#     - 若名称未注册会抛出 KeyError，应在调用前验证名称或捕获异常。
#     - 建议测试：1) 注册后能正确取回实现；2) 查询不存在名称时确实抛出 KeyError。
# (F) 背景参考：
#     - 背景：注册表模式（Registry Pattern）是常见的软件工程组织手法。
#     - 推荐阅读：《Design Patterns: Elements of Reusable Object-Oriented Software》《Clean Architecture》。


__AI_ANNOTATION_SUMMARY__ = """
get_aggregator: 提供基于名称的聚合器检索入口，依赖注册表模式确保策略可按需获取。
"""
