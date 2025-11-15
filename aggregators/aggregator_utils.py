"""
联邦学习聚合器常用工具函数集合。

本文件封装了多个聚合策略共享的数学运算与辅助流程，避免在不同聚合器中重复实现。
注意：仅当函数会被多个聚合器复用时才应放置于此，以保持聚合器模块职责清晰。
"""
from collections import defaultdict
from copy import deepcopy
import numpy as np
from fl.models.model_utils import add_vec2model, model2vec, vec2model


def L2_distances(updates):
    """
    计算一组模型更新向量之间的两两欧氏距离。

    参数:
        updates (numpy.ndarray | list[numpy.ndarray]): 包含若干客户端模型更新的向量列表或数组。

    返回:
        collections.defaultdict: 双层字典结构，distances[i][j] 表示第 i 与第 j 个更新之间的 L2 距离。

    异常:
        TypeError: 当输入无法支持向量减法或范数运算时可能抛出。

    复杂度:
        时间复杂度 O(n^2 * d)，其中 n 为更新数量、d 为向量维度；空间复杂度 O(n^2)。
    """
    # 使用默认字典嵌套结构存储距离矩阵，以便按需访问并自动初始化子字典。
    distances = defaultdict(dict)
    # 双重循环遍历所有无序对，仅计算上三角避免重复运算。
    for i in range(len(updates)):
        for j in range(i):
            # 通过 numpy.norm 计算两个向量的 L2 距离，并对称赋值节约计算。
            distances[i][j] = distances[j][i] = np.linalg.norm(
                updates[i] - updates[j])
    return distances


# 费曼学习法解释（L2_distances）
# (A) 功能概述：L2_distances 计算每对客户端更新向量之间的欧氏距离并整理成查询友好的结构。
# (B) 类比说明：就像量尺逐对测量城市间的直线距离，最后形成一张距离表。
# (C) 逐步拆解：
#     1. 准备一个嵌套字典保存结果——方便用索引 i、j 直接查到对应距离。
#     2. 遍历所有更新的组合——因为要比较每一对客户端更新的差异。
#     3. 使用欧氏距离公式计算向量差值的长度——衡量两次更新的相似程度。
#     4. 将距离写入字典的两个方向（i→j、j→i）——确保后续查询不需要再计算。
# (D) 最小示例：
#     >>> import numpy as np
#     >>> from aggregators.aggregator_utils import L2_distances
#     >>> updates = [np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 0.0])]
#     >>> dists = L2_distances(updates)
#     >>> dists[1][0]
#     1.0
# (E) 边界条件与测试建议：
#     - 若 updates 为空将返回空字典；若元素形状不一致会导致 numpy 运算报错。
#     - 建议测试：1) 三个二维向量的距离矩阵是否对称；2) 输入单个向量时是否返回空距离集合。
# (F) 背景参考：
#     - 背景：欧氏距离源自向量范数理论，是量化差异的基础工具。
#     - 推荐阅读：《Linear Algebra and Its Applications》《Pattern Recognition and Machine Learning》。


def krum_compute_scores(distances, i, n, f):
    """
    依据 KRUM 聚合规则计算节点 i 的得分。

    参数:
        distances (dict[int, dict[int, float]]): 节点间距离表，通常由 L2_distances 生成。
        i (int): 目标节点索引。
        n (int): 参与聚合的总客户端数量。
        f (int): 假设的恶意客户端数量上界。

    返回:
        float: 节点 i 的 KRUM 得分（越小越可信）。

    异常:
        KeyError: 若 distances 未包含所需的邻接距离时抛出。

    复杂度:
        时间复杂度 O(n log n)（排序主导）；空间复杂度 O(n)。
    """
    # 将节点 i 到其他节点的距离排序后取前 n-f-1 项（即最邻近的正常节点距离）。
    _s = sorted([dist for dist in distances[i].values()])[:n-f-1]
    # KRUM 得分等于这些最近邻距离之和，用于度量节点与正常群体的接近程度。
    return sum(_s)


# 费曼学习法解释（krum_compute_scores）
# (A) 功能概述：krum_compute_scores 计算某个客户端与其最近的 n-f-1 个邻居之间的距离之和。
# (B) 类比说明：像从教室里挑出最接近你意见的同学，求他们与自己的意见差距总和。
# (C) 逐步拆解：
#     1. 取出节点 i 与所有其他节点的距离——因为 KRUM 依赖邻居距离做筛选。
#     2. 对这些距离排序——目的是找到最靠近的潜在正常节点。
#     3. 选取前 n-f-1 个最小距离——假设最多有 f 个异常点，将其排除在外。
#     4. 求和得到得分——得分越小，说明节点更接近主流更新，更可信。
# (D) 最小示例：
#     >>> distances = {0: {1: 1.0, 2: 2.0}, 1: {0: 1.0, 2: 1.5}, 2: {0: 2.0, 1: 1.5}}
#     >>> from aggregators.aggregator_utils import krum_compute_scores
#     >>> krum_compute_scores(distances, i=0, n=3, f=1)
#     1.0
# (E) 边界条件与测试建议：
#     - 若 n-f-1 <= 0 切片将为空，得分为 0；需保证参数满足 KRUM 约束 n > 2f+2。
#     - 建议测试：1) 正常参数下得分是否可比；2) 距离缺失时是否触发 KeyError。
# (F) 背景参考：
#     - 背景：KRUM 聚合来自 Byzantine-resilient FL 的经典算法。
#     - 推荐阅读：《Theoretical Advances in Federated Learning》《Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers》。


def prepare_grad_updates(algorithm, updates, global_model):
    """
    将客户端更新转换为梯度形式，以统一后续聚合接口。

    参数:
        algorithm (str): 当前联邦优化算法名称，如 'FedAvg'、'FedSGD'、'FedOpt'。
        updates (numpy.ndarray): 客户端回传的模型更新或梯度。
        global_model (torch.nn.Module): 当前全局模型，用于计算差分。

    返回:
        numpy.ndarray: 梯度形式的更新数组。

    异常:
        AttributeError: 若 global_model 不支持 model2vec/vec2model 所需接口。

    复杂度:
        时间复杂度 O(n * d)；空间复杂度 O(n * d)，n 为客户端数量，d 为参数维度。
    """
    # 客户端数量等于更新条目数量。

    num_updates = len(updates)  # equal to num_clients
    # gradient_updates
    # 注意：以下 if 条件存在潜在逻辑问题，"FedSGD" or "FedOpt" in algorithm 始终为真，应在后续重构中修正。
    gradient_updates = updates if "FedSGD" or "FedOpt" in algorithm else np.array(
            [updates[cid] - model2vec(global_model) for cid in range(num_updates)])
    
    return gradient_updates
   
def prepare_updates(algorithm, updates, global_model, vector_form=True):
    """
    根据算法类型整理客户端更新，输出模型形式与梯度形式两种视图。

    参数:
        algorithm (str): 联邦算法名称，决定 updates 的含义与转换方式。
        updates (numpy.ndarray): 客户端上传的更新（可能是模型参数或梯度向量）。
        global_model (torch.nn.Module): 当前全局模型，FedAvg 下需配合计算梯度。
        vector_form (bool): 若为 True 返回向量形式的模型更新，否则返回模型对象列表。

    返回:
        tuple[numpy.ndarray, numpy.ndarray]: (model_updates, gradient_updates)。
            - model_updates: 各客户端的模型参数向量或模型对象。
            - gradient_updates: 梯度或伪梯度向量。

    异常:
        ValueError: 若 algorithm 非预期字符串组合时可能导致逻辑错误。
        AttributeError: global_model 与转换函数接口不匹配时抛出。

    复杂度:
        时间复杂度 O(n * d)；空间复杂度 O(n * d)。
    """
    num_updates = len(updates)  # equal to num_clients
    # gradient_updates
    if algorithm == 'FedAvg':
        # FedAvg 场景中的 updates 即模型参数向量，需转换为梯度形式。
        vec_updates = updates
        gradient_updates = np.array(
            [updates[cid] - model2vec(global_model) for cid in range(num_updates)])

    elif "FedSGD" or "FedOpt" in algorithm:
        # FedSGD/FedOpt 场景：updates 表示梯度，需还原对应模型参数。
        gradient_updates = updates
        vec_updates = np.array(
            [model2vec(global_model) + updates[cid] for cid in range(num_updates)])

    if vector_form:
        # vector_form return 1d np array vector model parameters
        model_updates = vec_updates
    else:
        # model-form model_updates
        model_updates = []
        for cid in range(num_updates):
            # 深拷贝全局模型，注入对应向量参数以恢复模型对象。
            tmp = deepcopy(global_model)
            vec2model(vec_updates[cid], tmp)
            model_updates.append(tmp)
        model_updates = np.array(model_updates)

    return model_updates, gradient_updates


def wrapup_aggregated_grads(benign_grad_updates, algorithm, global_model, aggregated=False):
    """
    根据算法类型封装聚合结果，输出最终应回传给服务器的格式。

    参数:
        benign_grad_updates (numpy.ndarray): 被判定为良性客户端的梯度或伪梯度集合。
        algorithm (str): 联邦算法名称，决定返回模型增量还是梯度。
        global_model (torch.nn.Module): 当前全局模型，用于重建 FedAvg 的模型向量。
        aggregated (bool): 若为 True 表示输入已是聚合结果，否则函数内部进行平均。

    返回:
        numpy.ndarray: FedAvg 返回模型参数向量，其余算法返回梯度向量。

    异常:
        ValueError: 输入为空或形状不兼容时可能在 numpy 操作中触发。

    复杂度:
        时间复杂度 O(n * d)；空间复杂度 O(d)。
    """
    # 若 aggregated=False，则对良性更新求均值，获得聚合梯度。
    aggregated_gradient = benign_grad_updates if aggregated else np.mean(
        benign_grad_updates, axis=0)
    if algorithm == 'FedAvg':
        # 对于 FedAvg，需要把聚合梯度叠加到全局模型参数上，输出新的模型向量。
        aggregated_model = add_vec2model(
            aggregated_gradient, global_model)
        return model2vec(aggregated_model)
    else:
        # 对于基于梯度更新的算法，直接返回聚合梯度即可。
        return aggregated_gradient


def normclipping(vectors, threshold, epsilon=1e-6):
    """
    对二维向量集合执行 L2 范数裁剪，防止异常大梯度影响聚合。

    参数:
        vectors (numpy.ndarray): 形状为 (n, d) 的向量矩阵，每行对应一个客户端更新。
        threshold (float): 裁剪阈值，超过该范数的向量将被等比例缩放。
        epsilon (float, optional): 数值稳定项，避免除零；默认 1e-6。

    返回:
        numpy.ndarray: 裁剪后的向量矩阵，与输入形状一致。

    异常:
        ValueError: 输入不是二维数组时抛出。

    复杂度:
        时间复杂度 O(n * d)；空间复杂度 O(n)（存储范数）。
    """
    # 仅接受二维矩阵输入，保证每行代表一个客户端向量。
    if len(vectors.shape) != 2:
        raise ValueError(
            "The input should be 2d vectors, or you need to extend this function")
    # 计算每行的范数并按需缩放，最小值操作确保范数不超过阈值。
    return vectors * np.minimum(1, threshold / (np.linalg.norm(vectors, axis=1)+epsilon)).reshape(-1, 1)


def addnoise(vector, noise_mean, noise_std):
    """
    向向量添加独立同分布的高斯噪声，常用于差分隐私与随机化。

    参数:
        vector (numpy.ndarray): 目标向量，可为任意形状。
        noise_mean (float): 噪声均值。
        noise_std (float): 噪声标准差（方差为 noise_std^2）。

    返回:
        numpy.ndarray: 与输入形状一致、已添加噪声的新向量。

    异常:
        ValueError: 当 noise_std 为负时 numpy.random.normal 将抛出错误。

    复杂度:
        时间复杂度 O(d)；空间复杂度 O(d)，d 为向量维度。
    """
    # generate gaussian noise, note that the noise should be float32 to be consistent with the future torch dtype
    noise = np.random.normal(noise_mean, noise_std,
                             vector.shape).astype(np.float32)
    return vector + noise


# 费曼学习法解释（prepare_grad_updates）
# (A) 功能概述：prepare_grad_updates 将客户端的原始回传数据转换成统一的梯度表示。
# (B) 类比说明：像是把不同单位的测量值换算成同一单位，方便比较与合并。
# (C) 逐步拆解：
#     1. 统计客户端数量——为后续遍历或构造数组提供边界。
#     2. 判断算法类型——因为不同算法回传的数据意义不同。
#     3. 若为梯度型算法，直接返回原始更新；若为参数型，减去全局模型得到伪梯度。
# (D) 最小示例：
#     >>> import numpy as np
#     >>> from aggregators.aggregator_utils import prepare_grad_updates
#     >>> grads = np.array([[0.1, 0.2], [0.0, -0.1]])
#     >>> prepare_grad_updates("FedSGD", grads, global_model=None)
#     array([[ 0.1,  0.2],
#            [ 0. , -0.1]])
# (E) 边界条件与测试建议：
#     - 当前实现中条件判断存在逻辑漏洞，会始终走“梯度型”分支，建议在测试中覆盖 FedAvg 情形。
#     - 建议测试：1) FedSGD 时返回原梯度；2) FedAvg 时应减去全局向量（需修复逻辑后验证）。
# (F) 背景参考：
#     - 背景：梯度与参数之间的互换是联邦优化的基础操作。
#     - 推荐阅读：《Deep Learning》（Goodfellow 等）、《Federated Learning》。


# 费曼学习法解释（prepare_updates）
# (A) 功能概述：prepare_updates 根据算法类型同时生成模型向量和梯度向量的视图。
# (B) 类比说明：像是把一本书翻译成不同语言版本，以便不同读者使用。
# (C) 逐步拆解：
#     1. 统计客户端数量——确保遍历准确。
#     2. 根据算法判断输入含义——FedAvg 的 updates 是模型，FedSGD/FedOpt 是梯度。
#     3. 生成梯度视图或参数视图——两种视角方便下游聚合。
#     4. 若需要模型对象形式，复制全局模型并写入向量参数。
# (D) 最小示例：
#     >>> import numpy as np
#     >>> from aggregators.aggregator_utils import prepare_updates
#     >>> vec_updates = np.array([[1.0, 2.0], [1.5, 2.5]])
#     >>> model_updates, grad_updates = prepare_updates("FedAvg", vec_updates, global_model=dummy_model)
#     >>> grad_updates.shape
#     (2, 2)
# (E) 边界条件与测试建议：
#     - 条件判断与 prepare_grad_updates 相同需注意逻辑；global_model 必须与 vec2model 接口兼容。
#     - 建议测试：1) FedAvg 时梯度等于 updates 减 global_model；2) vector_form=False 时返回模型对象。
# (F) 背景参考：
#     - 背景：不同算法的通讯语义不同，需要标准化以复用聚合逻辑。
#     - 推荐阅读：《Communication-Efficient Learning of Deep Networks from Decentralized Data》《Deep Learning》。


# 费曼学习法解释（wrapup_aggregated_grads）
# (A) 功能概述：wrapup_aggregated_grads 将良性梯度集合组合成最终广播的模型或梯度。
# (B) 类比说明：类似把多位专家意见综合成一份报告，再决定是给出结论还是给出依据。
# (C) 逐步拆解：
#     1. 判断 benign_grad_updates 是否已聚合——否则对其求平均。
#     2. 若算法为 FedAvg，将聚合梯度加回全局模型得到新参数。
#     3. 其他算法直接返回聚合梯度，因为服务器按梯度更新模型。
# (D) 最小示例：
#     >>> import numpy as np
#     >>> from aggregators.aggregator_utils import wrapup_aggregated_grads
#     >>> wrapup_aggregated_grads(np.array([[0.1, 0.2], [0.0, 0.1]]), "FedSGD", global_model=None)
#     array([0.05, 0.15])
# (E) 边界条件与测试建议：
#     - 输入为空会触发 numpy 均值警告；FedAvg 需确保 add_vec2model 与 model2vec 兼容。
#     - 建议测试：1) aggregated=True 与 False 的输出是否一致；2) FedAvg 下输出是否与期望模型向量一致。
# (F) 背景参考：
#     - 背景：FedAvg 与 FedSGD 在服务器端更新语义不同，需要区别处理。
#     - 推荐阅读：《Federated Optimization in Heterogeneous Networks》《Foundations of Machine Learning》。


# 费曼学习法解释（normclipping）
# (A) 功能概述：normclipping 将每个客户端的更新向量裁剪到指定阈值以内。
# (B) 类比说明：像给每个人的嗓音加限制器，音量超过规定就自动压低。
# (C) 逐步拆解：
#     1. 检查输入是否为二维矩阵——因为需逐行处理。
#     2. 计算每行向量的 L2 范数——用于判断是否超过阈值。
#     3. 通过最小值比较得到缩放因子——范数大于阈值的行会被缩放。
#     4. 将向量乘以缩放因子，返回裁剪后的结果。
# (D) 最小示例：
#     >>> import numpy as np
#     >>> from aggregators.aggregator_utils import normclipping
#     >>> normclipping(np.array([[3.0, 4.0], [0.0, 1.0]]), threshold=5.0)
#     array([[3., 4.],
#            [0., 1.]])
# (E) 边界条件与测试建议：
#     - 阈值应大于零；向量全零时依靠 epsilon 防止除零。
#     - 建议测试：1) 范数大于阈值的向量是否被缩放；2) 输入形状错误时是否抛出 ValueError。
# (F) 背景参考：
#     - 背景：范数裁剪是差分隐私与鲁棒聚合常用的预处理手段。
#     - 推荐阅读：《Deep Learning with Differential Privacy》《Machine Learning》。


# 费曼学习法解释（addnoise）
# (A) 功能概述：addnoise 为向量加入高斯噪声以增加随机性或保护隐私。
# (B) 类比说明：类似把真话掺入轻微噪声，别人听起来更难精确判断原话。
# (C) 逐步拆解：
#     1. 根据目标向量形状生成高斯噪声矩阵——确保维度一致。
#     2. 将噪声转换为 float32——与后续深度学习框架的数据类型保持一致。
#     3. 将噪声加到原向量上，形成扰动结果。
# (D) 最小示例：
#     >>> import numpy as np
#     >>> from aggregators.aggregator_utils import addnoise
#     >>> np.random.seed(0)
#     >>> addnoise(np.array([1.0, 2.0]), 0.0, 0.1)
#     array([1.1764053, 2.0400157], dtype=float32)
# (E) 边界条件与测试建议：
#     - noise_std 需非负；重复实验需固定随机种子以保持可重复性。
#     - 建议测试：1) 检查返回形状一致；2) 多次调用的统计分布是否符合预期均值与方差。
# (F) 背景参考：
#     - 背景：高斯噪声添加广泛用于差分隐私与鲁棒性提升。
#     - 推荐阅读：《The Algorithmic Foundations of Differential Privacy》《Probability and Random Processes》。


__AI_ANNOTATION_SUMMARY__ = """
L2_distances: 计算客户端更新之间的两两欧氏距离并缓存为字典结构。
krum_compute_scores: 基于 KRUM 规则求节点最近邻距离之和以评估可信度。
prepare_grad_updates: 将不同算法回传的数据统一转换为梯度表示。
prepare_updates: 同时生成模型视图与梯度视图以支撑多算法聚合流程。
wrapup_aggregated_grads: 按算法语义封装聚合结果为模型或梯度向量。
normclipping: 对客户端更新执行范数裁剪以抑制异常大梯度。
addnoise: 向向量添加高斯噪声以提升隐私或鲁棒性。
"""
