"""
攻击基类 PBase：为投毒攻击统一处理参数注入逻辑。

本模块定义所有攻击基类共享的 `update_and_set_attr` 方法，
用于将配置中的攻击参数更新为实例属性，方便在具体攻击实现中访问。
"""


class PBase:
    """
    攻击参数基类：提供统一的参数更新与属性注入机制。

    子类应定义 `self.default_attack_params`，并通过 `self.args.attack_params`
    传入自定义配置，调用 `update_and_set_attr` 后即可以属性形式访问最终参数。
    """

    def update_and_set_attr(self):
        """
        合并默认攻击参数与外部配置，并将结果写入实例属性。

        工作流程:
            1. 从 `self.args.attack_params` 读取用户配置；
            2. 以 `self.default_attack_params` 为基础进行浅拷贝；
            3. 用用户配置覆盖默认参数；
            4. 遍历最终字典，将键值对设置为实例属性，便于直接访问。

        异常:
            AttributeError: 当实例缺少 `args`、`default_attack_params` 或对应字段时可能抛出。

        复杂度:
            时间复杂度 O(k)，空间复杂度 O(1)，其中 k 为参数数量。
        """
        new_attack_params = self.args.attack_params
        self.attack_params = self.default_attack_params
        # 若用户提供了自定义配置，则覆盖默认参数。
        if new_attack_params:
            self.attack_params.update(new_attack_params)
        # 将合并后的参数逐一写入实例属性，便于在子类中直接访问。
        for key, value in self.attack_params.items():
            setattr(self, key, value)


# 费曼学习法解释（PBase.update_and_set_attr）
# (A) 功能概述：把默认攻击参数与用户自定义配置合并，并转换为实例属性。
# (B) 类比说明：像先拿到课程默认教材，再把学生自带的补充材料贴上标签放进自己的书架。
# (C) 步骤拆解：
#     1. 读取外部传入的攻击参数字典。
#     2. 以默认参数为基础字典。
#     3. 若有自定义配置，则覆盖默认值。
#     4. 遍历最终参数，将键值设置为对象属性，后续可直接 `self.xxx` 访问。
# (D) 最小示例：
#     >>> class Dummy(PBase):
#     ...     def __init__(self, args):
#     ...         self.args = args
#     ...         self.default_attack_params = {"scale": 1.0}
#     >>> args = type("Args", (), {"attack_params": {"scale": 2.0}})
#     >>> inst = Dummy(args)
#     >>> inst.update_and_set_attr()
#     >>> inst.scale
#     2.0
# (E) 边界条件与测试建议：
#     - 若 `attack_params` 为 None 或空字典，需确认保持默认值；
#     - 缺失 `default_attack_params` 会触发 AttributeError；
#     - 建议测试：有无自定义配置两种情况。
# (F) 参考：Python 属性注入及配置管理实践。


__AI_ANNOTATION_SUMMARY__ = """
PBase.update_and_set_attr: 合并默认与用户攻击参数并写入实例属性，统一攻击配置管理。
"""
