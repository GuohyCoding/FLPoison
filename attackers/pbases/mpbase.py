"""
模型投毒攻击基础类 MPBase：区分“非全知”与“全知”两类攻击入口。

继承该类的攻击者可根据自身能力在本地训练阶段或全局聚合阶段插入投毒逻辑，
通过重写 `non_omniscient` 或 `omniscient` 方法实现具体策略。
"""
from attackers.pbases.pbase import PBase


class MPBase(PBase):
    """
    模型投毒攻击基类，为不同攻防能力提供统一扩展接口。

    使用说明:
        - 重写 `non_omniscient` 实现独立（非全知）攻击；
        - 重写 `omniscient` 实现协同或全知攻击；
        - 亦可在客户端类中重写 `local_training`、`step` 等方法操控训练流程。
    """

    def non_omniscient(self):
        """
        非全知模型投毒入口，默认在本地训练结束后执行。

        派生类应实现该方法以定义独立攻击者的行为：通常在每轮本地训练结束、
        上传更新之前，对自身模型或梯度做篡改。

        复杂度:
            由具体实现决定；此处只提供抽象接口。
        """
        raise NotImplementedError

    def omniscient(self, clients):
        """
        全知模型投毒入口，通常在全局聚合阶段执行。

        参数:
            clients (list[Client]): 当前参与的客户端列表，允许攻击者访问他人更新或共享信息。

        派生类可借助该接口实现协同攻击、监听其他客户端的更新等高级能力。

        复杂度:
            取决于子类实现；此处仅声明接口。
        """
        raise NotImplementedError


# 费曼学习法解释（MPBase.non_omniscient）
# (A) 功能概述：为独立攻击者提供投毒入口，通常在本地训练结束时篡改更新。
# (B) 类比说明：像在每节课结束时，某个学生偷偷修改自己的作业答案再交给老师。
# (C) 步骤拆解：
#     1. 在子类中重写该方法。
#     2. 操控本地模型、梯度或待上传参数。
#     3. 返回或直接修改状态，以影响服务器聚合。
# (D) 最小示例：
#     >>> class MyAttack(MPBase):
#     ...     def non_omniscient(self):
#     ...         self.model.params += self.delta
# (E) 边界条件与测试建议：确保方法在本地训练流程中被调用；建议测试上传前后的模型差异。
# (F) 参考：模型投毒攻击综述、单客户端独立攻击策略。


# 费曼学习法解释（MPBase.omniscient）
# (A) 功能概述：为协同或全知攻击者提供在全局阶段进行投毒的入口。
# (B) 类比说明：像几位学生在交卷前互通答案，共同伪造一份对他们有利的结果。
# (C) 步骤拆解：
#     1. 子类重写该方法，接收所有客户端对象或更新。
#     2. 可分析其他客户端的梯度或共享信息。
#     3. 调整恶意客户端的提交或协同篡改。
# (D) 最小示例：
#     >>> class CollusiveAttack(MPBase):
#     ...     def omniscient(self, clients):
#     ...         malicious = [c for c in clients if c.is_malicious]
#     ...         aggregate = sum(c.update for c in malicious)
# (E) 边界条件与测试建议：需确保在全局阶段调用；建议测试协同行为对聚合结果的影响。
# (F) 参考：全知攻击模型、协同投毒研究。


__AI_ANNOTATION_SUMMARY__ = """
MPBase.non_omniscient: 抽象方法，为独立模型投毒攻击定义本地阶段篡改入口。
MPBase.omniscient: 抽象方法，为协同或全知模型投毒提供全局阶段操作接口。
"""
