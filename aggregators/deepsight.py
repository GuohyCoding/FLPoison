"""
DeepSight 聚合器：通过深度模型检查识别并过滤潜在后门客户端。

算法来源于 NDSS 2022《DeepSight: Mitigating Backdoor Attacks in Federated Learning Through Deep Model Inspection》，
其核心流程包括：计算各客户端更新的能量指标（NEUP/TE）、基于随机噪声的 Division Differences（DDif），
以及偏置向量的余弦距离，随后执行集成聚类、标记可疑簇，并对被接受的更新执行范数裁剪后聚合。
"""
import numpy as np
import hdbscan
import torch
from aggregators.aggregatorbase import AggregatorBase
from aggregators.aggregator_utils import normclipping, prepare_updates, wrapup_aggregated_grads
from fl.models.model_utils import ol_from_vector
from aggregators import aggregator_registry
from sklearn.metrics.pairwise import cosine_distances


@aggregator_registry
class DeepSight(AggregatorBase):
    """
    DeepSight 聚合器实现，结合能量指标与集成聚类筛除后门客户端。

    算法通过 NEUP、TE、DDif 以及偏置余弦距离等多维特征执行聚类，
    将可疑簇剔除后，仅对剩余客户端的梯度进行裁剪与聚合。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 DeepSight 聚合器并预生成随机噪声数据集。

        参数:
            args (argparse.Namespace | Any): 运行配置对象，需包含
                - algorithm (str): 联邦算法名称，传统设置为 'FedAvg'。
                - defense_params (dict, optional): 用户覆盖默认防御参数。
                - num_clients (int): 客户端数量。
                - num_classes (int): 输出类别数，用于能量归一化。
                - num_channels (int)、num_dims (int): 数据形状参数。
                - batch_size (int)、num_workers (int)、device (torch.device): DDif 评估所需。
            **kwargs: 预留关键字参数，当前未使用。

        返回:
            None

        异常:
            AttributeError: 当 args 缺少上述字段或 defense_params 属性时可能抛出。

        复杂度:
            时间复杂度 O(S * m * d)，S 为随机种子数量，m 为噪声样本数，d 为模型输出维度；
            空间复杂度 O(S * m * d)（主要来自预生成的噪声数据）。
        """
        super().__init__(args)
        self.algorithm = "FedAvg"
        """
        num_seeds (int): 每个客户端计算 Division Difference 的随机种子数量，用于降低随机性。
        threshold_factor (float): NEUP 阈值缩放系数，用于判断阈值超限次数。
        num_samples (int): 每个噪声数据集的样本数量。
        tau (float): 集群判定阈值，若可疑比例低于该值则认为簇为良性。
        epsilon (float): 防止除零的微小常数。
        """
        self.default_defense_params = {
            "num_seeds": 3, "threshold_factor": 0.01, "num_samples": 20000, "tau": 0.33, "epsilon": 1.0e-6}
        self.update_and_set_attr()
        # 预生成随机噪声数据集，用于重复计算 DDif 时复用，节约时间。
        self.rand_datasets = self.generate_randdata()

    def aggregate(self, updates, **kwargs):
        """
        执行 DeepSight 聚合流程，输出经过筛选与裁剪的全局更新。

        参数:
            updates (numpy.ndarray | list[numpy.ndarray]): 客户端上传的模型向量或梯度。
            **kwargs: 需要包含
                - last_global_model (torch.nn.Module): 上一轮全局模型。
                - global_epoch (int, optional): 当前全局轮次，仅用于记录或调试。

        返回:
            numpy.ndarray: 被裁剪后的聚合向量，具体语义由 self.algorithm 决定。

        异常:
            KeyError: 当 kwargs 缺少 'last_global_model' 键时抛出。
            ValueError: 若聚类输出标签全为 -1（全部视作离群点）时后续步骤可能失败，需外部处理。

        复杂度:
            时间复杂度由 prepare_updates、DDif 计算与聚类共同决定，近似 O(n * d + S * m * d)；
            空间复杂度 O(n * d)。
        """
        # 记录当前轮次与全局模型，为后续特征提取与返回封装做准备。
        self.global_epoch = kwargs.get('global_epoch', None)
        self.global_model = kwargs['last_global_model']

        # 将更新转换为模型对象与梯度矩阵，便于后续逐客户端分析。
        client_updated_model, gradient_updates = prepare_updates(
            self.args.algorithm, updates, self.global_model, vector_form=False)

        # 提取各客户端输出层的权重与偏置更新，作为能量与余弦特征的基础。
        self.ol_updates = np.array([
            ol_from_vector(
                gradient_updates[cid], self.global_model, flatten=False, return_type='dict')
            for cid in range(self.args.num_clients)
        ])

        # 计算 NEUP、TE、DDif 以及偏置余弦距离，用作聚类输入特征。
        NEUPs, TEs = self.get_NEUPs_TEs()
        DDifs = self.get_DDifs(client_updated_model)
        cosine_dists = self.get_cosine_distance()

        # 执行集成聚类，整合多个特征的距离矩阵得到聚类标签。
        import warnings
        warnings.filterwarnings("error", category=RuntimeWarning)
        try:
            cluster_labels = self.clustering(NEUPs, DDifs, cosine_dists)
        except RuntimeWarning as e:
            print(f"!Warning: {e}")

        # 根据 TE 阈值标记可疑客户端（True 表示可疑），用于后续簇判定。
        suspicious_flags = TEs <= np.median(TEs)/2
        accepted_indices = []

        # 统计有效簇数量（排除 -1 标签的离群点），逐簇检查可疑比例。
        n_clusters = len(set(cluster_labels)) - \
            (1 if -1 in cluster_labels else 0)
        for i in range(n_clusters):
            indices = np.where(cluster_labels == i)[0]
            amount_of_suspicious = np.sum(suspicious_flags[indices])
            # 仅当可疑占比低于阈值 tau 时，才将该簇视为良性并保留。
            if amount_of_suspicious < self.tau * len(indices):
                accepted_indices.extend(indices)
        accepted_indices = np.array(accepted_indices, dtype=np.int64)

        # 对保留下的梯度执行范数裁剪，阈值取全体梯度范数的中位数。
        clipped_gradient_updates = normclipping(
            gradient_updates[accepted_indices],
            np.median(np.linalg.norm(gradient_updates, axis=1))
        )

        # 返回封装好的聚合结果；对 FedAvg 会转换为模型向量，其余算法返回梯度。
        return wrapup_aggregated_grads(clipped_gradient_updates, self.args.algorithm, self.global_model)

    def get_NEUPs_TEs(self):
        """
        计算每个客户端的归一化更新能量 (NEUP) 与阈值超限次数 (TE)。

        返回:
            tuple[numpy.ndarray, numpy.ndarray]:
                - NEUPs: 形状 (num_clients, num_classes) 的能量分布矩阵。
                - TEs: 形状 (num_clients,) 的阈值超限计数向量。

        异常:
            ZeroDivisionError: 若能量平方和为零，除法可能导致错误（理论上由 epsilon 与权值变化避免）。

        复杂度:
            时间复杂度 O(n * c * d)，n 为客户端数、c 为类别数、d 为输出层维度；
            空间复杂度 O(n * c)。
        """
        num_clients = self.args.num_clients
        num_classes = self.args.num_classes
        TEs = np.empty(num_clients)
        NEUPs = np.empty((num_clients, num_classes))
        threshold_factor = max(self.threshold_factor, 1 / num_classes)

        # 遍历客户端，基于输出层权重与偏置计算能量与阈值。
        for cid in range(num_clients):
            updates = self.ol_updates[cid]
            update_energy = np.abs(
                updates['bias']) + np.sum(np.abs(updates['weight']), axis=1)
            # NEUP 为能量平方后再归一化，强调差异。
            energy_squared = update_energy**2
            NEUP = energy_squared/np.sum(energy_squared)
            threshold = threshold_factor * np.max(NEUP)
            TEs[cid] = np.sum(NEUP > threshold)
            NEUPs[cid] = NEUP

        return NEUPs, TEs

    def get_cosine_distance(self):
        """
        计算客户端输出层偏置更新之间的余弦距离矩阵。

        返回:
            numpy.ndarray: 形状 (num_clients, num_clients) 的余弦距离矩阵。

        复杂度:
            时间复杂度 O(n^2 * d_bias)，空间复杂度 O(n^2)。
        """
        bias_update = np.array([self.ol_updates[cid]['bias']
                                for cid in range(self.args.num_clients)])
        cosine_dists = cosine_distances(
            bias_update.reshape(self.args.num_clients, -1))
        return cosine_dists.astype(np.float64)

    def get_DDifs(self, client_updated_model):
        """
        计算每个客户端相对全局模型的 Division Differences (DDif)。

        参数:
            client_updated_model (list[torch.nn.Module]): 每个客户端更新后的模型副本。

        返回:
            numpy.ndarray: 形状 (num_seeds, num_clients, num_classes) 的 DDif 张量。

        异常:
            RuntimeError: 若模型在 `to(device)` 过程中设备不匹配可能抛出。

        复杂度:
            时间复杂度 O(num_seeds * num_samples * num_clients * forward_cost)；
            空间复杂度 O(num_seeds * num_samples * num_classes)。
        """
        DDifs = []
        for dataset in self.rand_datasets:
            seed_ddifs = []
            rand_loader = torch.utils.data.DataLoader(
                dataset, self.args.batch_size, shuffle=False,
                num_workers=self.args.num_workers, pin_memory=True
            )
            for cid in range(self.args.num_clients):
                client_updated_model[cid].eval()
                self.global_model.eval()

                DDif = torch.zeros(self.args.num_classes)
                # 仅使用随机噪声图像产生输出概率，用于衡量模型偏移。
                for rand_images in rand_loader:
                    rand_images = rand_images.to(self.args.device)
                    with torch.no_grad():
                        output_client = client_updated_model[cid](rand_images)
                        output_global = self.global_model(rand_images)
                    temp = output_client.cpu() / (output_global.cpu()+self.epsilon)
                    DDif.add_(torch.sum(temp, dim=0))
                seed_ddifs.append((DDif / self.num_samples).numpy())
            DDifs.append(seed_ddifs)
        return np.array(DDifs)

    def clustering(self, NEUPs, DDifs, cosine_dists):
        """
        基于 NEUP、DDif 与余弦距离执行集成聚类，识别可疑客户端簇。

        参数:
            NEUPs (numpy.ndarray): 归一化能量矩阵。
            DDifs (numpy.ndarray): Division Differences 张量。
            cosine_dists (numpy.ndarray): 偏置余弦距离矩阵。

        返回:
            numpy.ndarray: 聚类标签数组，若允许单簇则正常客户端为 0。

        复杂度:
            时间复杂度受 HDBSCAN 运行影响，约 O(n log n) 到 O(n^2)；
            空间复杂度 O(n^2)。
        """

        # 内部函数：对输入统计量执行 HDBSCAN 聚类，并返回簇间距离矩阵。
        def cluster_dists(statistic, precomputed=False):
            func = hdbscan.HDBSCAN(
                min_samples=3, metric='precomputed') if precomputed else hdbscan.HDBSCAN(min_samples=3)
            cluster_labels = func.fit_predict(statistic)
            cluster_dists = dists_from_clust(cluster_labels)
            return cluster_dists

        cosine_cluster_dists = cluster_dists(cosine_dists, precomputed=True)
        neup_cluster_dists = cluster_dists(NEUPs)
        ddif_cluster_dists = np.array(
            [cluster_dists(DDifs[i]) for i in range(self.num_seeds)])

        # 将多次 DDif 聚类结果取平均，再与 NEUP、余弦距离融合。
        merged_ddif_cluster_dists = np.mean(ddif_cluster_dists, axis=0)
        merged_distances = np.mean([merged_ddif_cluster_dists,
                                    neup_cluster_dists,
                                    cosine_cluster_dists], axis=0)
        cluster_labels = hdbscan.HDBSCAN(
            metric='precomputed', allow_single_cluster=True, min_samples=3).fit_predict(merged_distances)
        return cluster_labels

    def generate_randdata(self):
        """
        预生成用于 DDif 计算的随机噪声数据集列表。

        返回:
            list[RandDataset]: 长度为 `num_seeds` 的噪声数据集。

        复杂度:
            时间复杂度 O(num_seeds * num_samples)；空间复杂度 O(num_seeds * num_samples * 数据维度)。
        """
        noise_shape = [self.args.num_channels,
                       self.args.num_dims, self.args.num_dims]
        rand_datasets = [
            RandDataset(noise_shape, self.num_samples, seed)
            for seed in range(self.num_seeds)
        ]
        return rand_datasets


class RandDataset(torch.utils.data.Dataset):
    """
    随机噪声数据集：用于生成 DDif 时提供输入。
    """

    def __init__(self, size, num_samples, seed):
        """
        初始化随机数据集并设置随机种子保证可复现。

        参数:
            size (Sequence[int]): 单个样本的形状 (C, H, W)。
            num_samples (int): 数据集中样本数量。
            seed (int): 随机种子，确保不同 dataset 可重复生成。
        """
        self.num_samples = num_samples
        torch.manual_seed(seed)
        self.dataset = torch.rand(num_samples, *size)

    def __len__(self):
        """
        返回数据集包含的样本数量。

        返回:
            int: 样本总数。
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        根据索引返回对应的噪声样本。

        参数:
            idx (int): 样本索引。

        返回:
            torch.Tensor: 形状为 `size` 的随机噪声张量。
        """
        return self.dataset[idx]


def dists_from_clust(cluster_labels):
    """
    根据聚类标签生成簇间距离矩阵。

    参数:
        cluster_labels (numpy.ndarray): 聚类标签数组。

    返回:
        numpy.ndarray: 0-1 距离矩阵，同簇为 0，异簇为 1。

    复杂度:
        时间复杂度 O(n^2)，空间复杂度 O(n^2)。
    """
    same_cluster = (cluster_labels[:, None] == cluster_labels)
    pairwise_dists = np.where(same_cluster, 0, 1)
    return pairwise_dists


# 费曼学习法解释（DeepSight.__init__）
# (A) 功能概述：设定 DeepSight 的默认超参并预生成随机噪声数据集。
# (B) 类比说明：像准备一套检测系统，先设置仪器灵敏度，再提前采集参考噪声样本。
# (C) 逐步拆解：
#     1. 调用父类构造函数保存外部配置。
#     2. 设置默认参数（num_seeds、threshold_factor 等）并允许覆盖。
#     3. 调用 update_and_set_attr 将参数注入实例属性。
#     4. 生成多个随机噪声数据集，为后续 DDif 计算节约时间。
# (D) 最小示例：
#     >>> class Args: 
#     ...     algorithm="FedAvg"; num_clients=10; num_classes=10
#     ...     num_channels=3; num_dims=32; batch_size=32
#     ...     num_workers=0; device="cpu"; defense_params=None
#     >>> ds = DeepSight(Args())
#     >>> len(ds.rand_datasets)
#     3
# (E) 边界条件与测试建议：
#     - 缺失关键字段会触发 AttributeError。
#     - 建议测试：1) 自定义防御参数是否覆盖默认值；2) 随机数据集是否按种子复现。
# (F) 背景参考：
#     - 背景：DeepSight 通过随机噪声探测后门行为。
#     - 推荐阅读：《DeepSight: Mitigating Backdoor Attacks in Federated Learning Through Deep Model Inspection》《Federated Learning》。


# 费曼学习法解释（DeepSight.aggregate）
# (A) 功能概述：整合能量特征与聚类结果，筛除可疑客户端并聚合剩余更新。
# (B) 类比说明：像先让每位成员做体检，再按照多个指标分组，剔除异常组后合并他们的意见。
# (C) 逐步拆解：
#     1. 记录当前全局模型和轮次，方便后续特征计算。
#     2. 将更新转为模型与梯度形式，以获取输出层参数。
#     3. 计算 NEUP、TE、DDif 以及余弦距离等统计量。
#     4. 使用这些特征做集成聚类，得到客户端簇标签。
#     5. 根据 TE 阈值判定可疑客户端，并按簇统计可疑比例。
#     6. 保留可疑比例低于 tau 的簇成员，其他视为潜在恶意。
#     7. 对保留的梯度执行范数裁剪并聚合，得到最终更新。
# (D) 最小示例（伪代码）：
#     >>> updates = np.random.randn(10, param_dim)
#     >>> result = ds.aggregate(updates, last_global_model=global_model, global_epoch=5)
# (E) 边界条件与测试建议：
#     - 若聚类全部标记为 -1，应加异常处理。
#     - 建议测试：1) 纯良性数据时大部分客户端被保留；2) 特意插入异常梯度时可正确剔除。
# (F) 背景参考：
#     - 背景：多特征聚类是后门检测常用策略。
#     - 推荐阅读：《DeepSight》原论文、《Pattern Recognition and Machine Learning》。


# 费曼学习法解释（DeepSight.get_NEUPs_TEs）
# (A) 功能概述：计算每个客户端更新能量的归一化分布及超阈值次数。
# (B) 类比说明：像测量每个乐器在合奏中的音量分布，并统计超过噪音阈值的次数。
# (C) 逐步拆解：
#     1. 遍历所有客户端的输出层更新。
#     2. 计算偏置与权重绝对值之和获得能量。
#     3. 将能量平方并归一化得到 NEUP。
#     4. 根据最大值与阈值因子计算阈值。
#     5. 统计超过阈值的维度数量作为 TE。
# (D) 最小示例：
#     >>> NEUPs, TEs = ds.get_NEUPs_TEs()
#     >>> NEUPs.shape, TEs.shape
#     ((10, 10), (10,))
# (E) 边界条件与测试建议：
#     - 若能量全为零需防止除零，可通过 epsilon 或加噪处理。
#     - 建议测试：1) 人为构造单个维度异常放大的情况；2) 能量平均分布时 TE 是否较低。
# (F) 背景参考：
#     - 背景：能量指标用于识别后门偏置。
#     - 推荐阅读：《DeepSight》、鲁棒统计教材。


# 费曼学习法解释（DeepSight.get_cosine_distance）
# (A) 功能概述：衡量客户端偏置更新之间的余弦距离。
# (B) 类比说明：像计算每个人意见向量之间的夹角，夹角越大表示方向越不同。
# (C) 逐步拆解：
#     1. 抽取每个客户端输出层的偏置向量。
#     2. 将偏置展平后放入 `cosine_distances` 计算矩阵。
#     3. 返回 float64 类型的距离矩阵以避免精度损失。
# (D) 最小示例：
#     >>> dists = ds.get_cosine_distance()
#     >>> dists.shape
#     (10, 10)
# (E) 边界条件与测试建议：
#     - 偏置向量若全零将导致 NaN，应在测试中覆盖。
#     - 建议测试：1) 偏置相同情况下距离为零；2) 取相反方向时距离接近 2。
# (F) 背景参考：
#     - 背景：余弦距离常用于向量相似度评估。
#     - 推荐阅读：《Pattern Recognition and Machine Learning》《Information Retrieval》。


# 费曼学习法解释（DeepSight.get_DDifs）
# (A) 功能概述：比较客户端模型与全局模型对随机噪声的响应差异。
# (B) 类比说明：像用随机噪声测试不同雷达的反应，衡量它们与标准雷达的偏差。
# (C) 逐步拆解：
#     1. 遍历预生成的噪声数据集（不同种子）。
#     2. 为每个客户端与全局模型执行前向推理。
#     3. 计算客户端输出与全局输出的比值并累加。
#     4. 将累积量除以样本数得到平均 DDif。
#     5. 汇总所有种子的结果为 3D 张量。
# (D) 最小示例：
#     >>> DDifs = ds.get_DDifs(client_models)
#     >>> DDifs.shape
#     (3, 10, 10)
# (E) 边界条件与测试建议：
#     - 需确保模型在 `eval` 模式且使用相同设备。
#     - 建议测试：1) 客户端等于全局模型时 DDif 接近全 1；2) 特定类别偏差大时 DDif 反映出异常。
# (F) 背景参考：
#     - 背景：DDif 用于评估模型输出分布变化。
#     - 推荐阅读：《DeepSight》、深度学习泛化教材。


# 费曼学习法解释（DeepSight.clustering）
# (A) 功能概述：融合多种距离矩阵并运行 HDBSCAN 聚类得到客户端簇。
# (B) 类比说明：像根据身高、体重、兴趣三种指标分别分组，再取平均形成综合分组方案。
# (C) 逐步拆解：
#     1. 定义 `cluster_dists` 函数，对给定统计量执行 HDBSCAN 并得到簇距矩阵。
#     2. 分别计算余弦距离、NEUP、DDif 的簇距。
#     3. 将多次 DDif 结果求平均，降低随机性。
#     4. 将三种距离矩阵平均融合。
#     5. 对融合距离再次运行 HDBSCAN（允许单簇）得到最终标签。
# (D) 最小示例：
#     >>> labels = ds.clustering(NEUPs, DDifs, cosine_dists)
#     >>> np.unique(labels)
#     array([0, 1])
# (E) 边界条件与测试建议：
#     - HDBSCAN 参数需与样本量匹配，避免全为 -1。
#     - 建议测试：1) 人工构造两个明显群体时能否正确区分；2) 调整 `min_samples` 是否影响敏感度。
# (F) 背景参考：
#     - 背景：HDBSCAN 属于基于密度的聚类。
#     - 推荐阅读：《Density-Based Clustering》相关教材。


# 费曼学习法解释（DeepSight.generate_randdata）
# (A) 功能概述：生成若干随机噪声数据集以复用在 DDif 计算中。
# (B) 类比说明：像预先录制多段背景噪音，随时可以拿来测试声音设备。
# (C) 逐步拆解：
#     1. 根据输入通道与尺寸确定单个样本形状。
#     2. 遍历种子编号，创建对应的 `RandDataset`。
#     3. 返回包含多个随机数据集的列表。
# (D) 最小示例：
#     >>> datasets = ds.generate_randdata()
#     >>> len(datasets)
#     3
# (E) 边界条件与测试建议：
#     - `num_samples` 过大可能导致内存消耗，需要测试不同规模。
#     - 建议测试：1) 不同种子生成的数据是否一致；2) 与外部随机样本比较方差。
# (F) 背景参考：
#     - 背景：随机噪声采样在鲁棒性研究中十分常见。
#     - 推荐阅读：《Monte Carlo Methods》《Deep Learning》。


# 费曼学习法解释（RandDataset.__init__）
# (A) 功能概述：依据给定形状与种子生成固定噪声数据集。
# (B) 类比说明：像使用固定随机种子生成一批噪声图片，保证重复实验时一致。
# (C) 逐步拆解：
#     1. 记录样本数量，便于后续访问。
#     2. 设置随机种子，保证可重复。
#     3. 生成指定形状的均匀分布随机张量。
# (D) 最小示例：
#     >>> ds = RandDataset((3, 32, 32), 128, seed=0)
#     >>> len(ds.dataset)
#     128
# (E) 边界条件与测试建议：
#     - 种子相同时应生成相同数据，可通过单元测试验证。
#     - 建议测试：1) 不同种子数据差异；2) 数据范围是否在 [0,1]。
# (F) 背景参考：
#     - 背景：随机数据生成常用于模型稳健性评估。
#     - 推荐阅读：《Probability and Random Processes》。


# 费曼学习法解释（RandDataset.__len__）
# (A) 功能概述：返回数据集的样本数量。
# (B) 类比说明：像查看一本书有多少页。
# (C) 逐步拆解：
#     1. 直接返回初始化时记录的 `num_samples`。
# (D) 最小示例：
#     >>> len(ds)
#     128
# (E) 边界条件与测试建议：
#     - 确认数值为正整数。
#     - 建议测试：初始化时提供 0 是否能正确返回。
# (F) 背景参考：
#     - 背景：遵循 PyTorch Dataset 接口规范。
#     - 推荐阅读：《Deep Learning with PyTorch》。


# 费曼学习法解释（RandDataset.__getitem__）
# (A) 功能概述：按索引返回对应的噪声样本。
# (B) 类比说明：像从一叠照片中按编号抽取某一张。
# (C) 逐步拆解：
#     1. 利用 NumPy/Torch 索引直接返回存储的张量。
# (D) 最小示例：
#     >>> sample = ds[0]
#     >>> sample.shape
#     torch.Size([3, 32, 32])
# (E) 边界条件与测试建议：
#     - 需确保索引落在合法范围内。
#     - 建议测试：索引越界是否抛出 IndexError。
# (F) 背景参考：
#     - 背景：Dataset 接口标准。
#     - 推荐阅读：《Deep Learning with PyTorch》。


# 费曼学习法解释（dists_from_clust）
# (A) 功能概述：将聚类标签转化为简单的 0-1 距离矩阵。
# (B) 类比说明：像制作一张表格，标记每对学生是否在同一个小组。
# (C) 逐步拆解：
#     1. 构造布尔矩阵表示是否同簇。
#     2. 用 `np.where` 将 True 映射为 0、False 映射为 1。
#     3. 返回得到的 0-1 距离矩阵。
# (D) 最小示例：
#     >>> dists_from_clust(np.array([0, 0, 1]))
#     array([[0, 0, 1],
#            [0, 0, 1],
#            [1, 1, 0]])
# (E) 边界条件与测试建议：
#     - 标签为 -1 会被视作单独的簇，应确认是否符合预期。
#     - 建议测试：1) 全部同簇结果是否全 0；2) 全部不同簇是否全 1（对角线除外）。
# (F) 背景参考：
#     - 背景：簇间距离矩阵常用于集成聚类。
#     - 推荐阅读：《Clustering Ensemble Methods》。


__AI_ANNOTATION_SUMMARY__ = """
DeepSight.__init__: 初始化 DeepSight 参数并预生成随机噪声数据集供 DDif 计算复用。
DeepSight.aggregate: 使用能量指标与集成聚类筛除可疑客户端后裁剪并聚合更新。
DeepSight.get_NEUPs_TEs: 计算各客户端的归一化能量分布及阈值超限次数。
DeepSight.get_cosine_distance: 求解客户端偏置更新之间的余弦距离矩阵。
DeepSight.get_DDifs: 基于随机噪声评估客户端模型与全局模型输出差异。
DeepSight.clustering: 融合多特征距离矩阵运行 HDBSCAN 获得聚类标签。
DeepSight.generate_randdata: 构建多个随机噪声数据集以反复使用。
RandDataset.__init__: 依据形状和种子生成可重现的随机噪声数据。
RandDataset.__len__: 返回随机数据集的样本数量。
RandDataset.__getitem__: 按索引提供随机噪声样本。
dists_from_clust: 将聚类标签转换为 0-1 距离矩阵以便后续集成。
"""
