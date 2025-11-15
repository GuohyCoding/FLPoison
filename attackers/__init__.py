"""
攻击器注册入口：动态导入模块并按能力标签分类。

该模块利用全局注册表记录所有攻击类，并根据 `_attributes` 中的标签区分数据投毒、
模型投毒与混合攻击，方便外部按需检索。
"""
import os
from global_utils import import_all_modules, Register

# 全局攻击器注册表，子模块通过装饰器注册。
attacker_registry = Register()

# 递归导入当前目录下的全部攻击模块，使注册装饰器立即生效。
import_all_modules(os.path.dirname(__file__))

# 数据投毒攻击（纯 Data Poisoning）
data_poisoning_attacks = [
    name
    for name in attacker_registry.keys()
    if "data_poisoning" in attacker_registry[name]._attributes
]

# 混合攻击（同时具有数据投毒和模型投毒能力）
hybrid_attacks = [
    name
    for name in attacker_registry.keys()
    if all(
        attr in attacker_registry[name]._attributes
        for attr in ["model_poisoning", "data_poisoning"]
    )
]

# 模型投毒攻击（不含数据投毒标签）
model_poisoning_attacks = [
    name
    for name in attacker_registry.keys()
    if "data_poisoning" not in attacker_registry[name]._attributes
]


def get_attacker_handler(name):
    """
    根据攻击名称返回注册表中的攻击类。

    参数:
        name (str): 攻击器名称，对应注册表键值。

    返回:
        type: 已注册的攻击类。

    异常:
        AssertionError: 当 name 为 "NoAttack" 时抛出，提示无需注册。
        KeyError: 若名称未注册，访问字典会抛出异常。

    复杂度:
        时间复杂度 O(1)；空间复杂度 O(1)。
    """
    assert name != "NoAttack", "NoAttack 应直接跳过注册流程，无需调用处理器。"
    return attacker_registry[name]


# 费曼学习法解释（get_attacker_handler）
# (A) 功能概述：根据名称从注册表中取出对应的攻击类。
# (B) 类比说明：像在电话簿中查找联系人，拿到其名片。
# (C) 步骤拆解：
#     1. 确认查询的不是特殊的 "NoAttack"。
#     2. 在注册表中根据名称查找并返回攻击类。
# (D) 最小示例：
#     >>> handler = get_attacker_handler("BadNetsAttack")
# (E) 边界条件与测试建议：未注册名称会抛 KeyError；建议测试合法与非法名称。
# (F) 参考：注册表模式、攻击器插件化设计。


__AI_ANNOTATION_SUMMARY__ = """
get_attacker_handler: 按名称从攻击注册表中检索攻击类，确保跳过 NoAttack 特殊条目。
"""
