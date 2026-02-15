"""
FLDetector 聚合器：基于历史梯度拟合与聚类实现的后门检测防御。

算法思想来自 NDSS 2021《Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning》。
核心流程：
1) 维护滑动窗口内的全局参数差分与梯度差分；
2) 使用 LBFGS 近似 Hessian-Vector Product 预测下一轮梯度；
3) 比较预测梯度与实际客户端梯度，得到恶意评分；
4) 结合 Gap Statistic 与 KMeans 聚类筛除可疑客户端，再对剩余梯度聚合。
"""
from copy import deepcopy
import numpy as np
import torch
from aggregators.aggregatorbase import AggregatorBase
from aggregators import aggregator_registry
from aggregators.aggregator_utils import prepare_updates, wrapup_aggregated_grads


@aggregator_registry
class FLDetector(AggregatorBase):
    """
    FLDetector 聚合器：通过梯度预测误差与聚类判别潜在恶意客户端。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 FLDetector，配置滑动窗口与检测起始轮次等防御参数。

        参数:
            args (argparse.Namespace | Any): 运行配置对象，应包含
                - num_clients (int): 客户端数量。
                - num_adv (int): 恶意客户端估计数量（用于日志）。
                - defense_params (dict, optional): 用于覆盖默认防御参数。
            **kwargs: 预留关键字参数，当前未使用。

        返回:
            None

        异常:
            AttributeError: 若 args 缺少 defense_params 等所需字段时抛出。

        复杂度:
            时间复杂度 O(1)，空间复杂度 O(1)。
        """
        super().__init__(args)
        self.default_defense_params = {"window_size": 10, "start_epoch": 50}
        self.update_and_set_attr()
        self.algorithm = "FedOpt"
        # 维护历史全局参数差分 (W^t - W^{t-1})，亦可视为聚合梯度 g^t。
        self.global_weight_diffs = []
        # 维护历史梯度差分 (g^t - g^{t-1})，用于近似 Hessian 信息。
        self.global_grad_diffs = []
        self.last_global_grad = 0
        self.last_grad_updates = 0
        self.malicious_score = []
        self.init_model = None

    def aggregate(self, updates, **kwargs):
        """
        执行 FLDetector 聚合流程，检测并屏蔽可疑客户端后返回聚合结果。

        参数:
            updates (numpy.ndarray | list[numpy.ndarray]): 客户端上传的梯度或参数更新。
            **kwargs: 需包含
                - last_global_model (torch.nn.Module): 上一轮全局模型。
                - global_epoch (int): 当前全局训练轮次。

        返回:
            numpy.ndarray: 聚合后的梯度向量（FedOpt 语义）。

        异常:
            KeyError: 缺少必需的 kwargs 键时抛出。
            RuntimeError: gap_statistics 等步骤未返回有效聚类时需由上层处理。

        复杂度:
            时间复杂度近似 O(n * d + window_size * d)，n 为客户端数、d 为参数维度；
            空间复杂度 O(window_size * d)。
        """
        # 保存本轮原始更新与全局模型副本。
        self.updates = updates
        self.global_model = deepcopy(kwargs['last_global_model'])
        self.current_epoch = kwargs['global_epoch']
        self.global_epoch = kwargs['global_epoch']

        # 在启动阶段缓存初始模型，便于检测到攻击后重置（论文建议）。
        if self.current_epoch <= self.start_epoch:
            self.init_model = self.global_model

        # 将更新转换为梯度形式，便于后续误差度量；vector_form=False 返回模型对象列表。
        _, gradient_updates = prepare_updates(
            self.args.algorithm, updates, self.global_model, vector_form=False)
        model_device = next(self.global_model.parameters()).device
        if torch.is_tensor(gradient_updates):
            gradient_updates = gradient_updates.detach().to(model_device)
        else:
            gradient_updates = torch.as_tensor(
                gradient_updates, device=model_device, dtype=torch.float32
            )
        benign_idx = torch.arange(
            gradient_updates.shape[0], device=gradient_updates.device
        )

        # 当历史记录充足时，利用 LBFGS 预测梯度并计算每个客户端的偏差得分。
        if self.current_epoch - self.start_epoch > self.window_size:
            hvp = self.LBFGS(self.global_weight_diffs, self.global_grad_diffs,
                             self.last_global_grad)
            distance = self.get_pred_real_dists(
                self.last_grad_updates, gradient_updates, hvp)
            self.malicious_score.append(distance)

        # 当恶意评分历史长度达到窗口规模时进行聚类检测。
        if len(self.malicious_score) > self.window_size:
            malicious_score = torch.stack(
                self.malicious_score[-self.window_size:], dim=0)
            score = torch.mean(malicious_score, dim=0)

            # 使用 Gap Statistic 判断最佳簇数量，若大于等于 2 则执行 KMeans 分离可疑客户端。
            if self.gap_statistics(score, num_sampling=20, K_max=10,
                                   n=self.args.num_clients) >= 2:
                score_2d = score.reshape(score.shape[0], -1)
                label_pred, _ = self._kmeans_fit_predict_gpu(
                    score_2d, n_clusters=2, n_init=10
                )
                # 均值较大的簇视为可疑，剩余索引被视为良性。
                score0 = torch.mean(score[label_pred == 0])
                score1 = torch.mean(score[label_pred == 1])
                benign_label = 1 if score0 > score1 else 0
                benign_idx = torch.nonzero(
                    label_pred == benign_label, as_tuple=False
                ).reshape(-1).to(device=gradient_updates.device, dtype=torch.long)
                self.args.logger.info(
                    f"FLDetector Defense: Benign idx: {benign_idx}")

        # 对被判定为良性的客户端梯度求均值，作为当前轮次聚合结果。
        benign_idx = benign_idx.to(device=gradient_updates.device, dtype=torch.long)
        agg_grad_update = torch.mean(
            gradient_updates.index_select(0, benign_idx), dim=0
        )

        # 更新滑动窗口：记录聚合梯度与梯度差分，维持 window_size 长度。
        self.global_weight_diffs.append(agg_grad_update)
        self.global_grad_diffs.append(
            agg_grad_update - self.last_global_grad)
        if len(self.global_weight_diffs) > self.window_size:
            del self.global_weight_diffs[0]
            del self.global_grad_diffs[0]
        self.last_global_grad = agg_grad_update
        self.last_grad_updates = gradient_updates

        return wrapup_aggregated_grads(
            agg_grad_update, self.args.algorithm, self.global_model, aggregated=True)

    def get_pred_real_dists(self, last_grad_updates, gradient_updates, hvp):
        """
        计算预测梯度与实际客户端梯度之间的距离得分。

        参数:
            last_grad_updates (numpy.ndarray): 上一轮客户端梯度矩阵。
            gradient_updates (numpy.ndarray): 当前轮客户端梯度矩阵。
            hvp (numpy.ndarray): LBFGS 近似得到的 Hessian-Vector Product。

        返回:
            numpy.ndarray: 每个客户端的归一化距离得分，越大表示越可疑。

        复杂度:
            时间复杂度 O(n * d)，空间复杂度 O(n)。
        """
        # 预测当前梯度：g_t ≈ g_{t-1} + H (w_t - w_{t-1})。
        pred_grad = last_grad_updates + hvp
        distance = torch.linalg.norm(pred_grad - gradient_updates, dim=1)
        # 归一化便于跨轮次比较。
        distance = distance / (torch.sum(distance) + 1e-12)
        return distance

    def LBFGS(self, S_k_list, Y_k_list, v):
        """
        使用有限内存 BFGS (L-BFGS) 公式近似计算 Hessian-Vector Product。

        参数:
            S_k_list (list[numpy.ndarray]): 历史参数差分向量列表。
            Y_k_list (list[numpy.ndarray]): 历史梯度差分向量列表。
            v (numpy.ndarray): 当前待乘的向量（通常为上一轮聚合梯度）。

        返回:
            numpy.ndarray: L-BFGS 近似得到的矩阵-向量积。

        复杂度:
            时间复杂度约 O(m^2 d)，m 为窗口长度；空间复杂度 O(m d)。
        """
        # 将历史记录统一 reshape 为列向量形式，便于线性代数运算。
        S_k_list = [i.reshape(-1, 1) for i in S_k_list]
        Y_k_list = [i.reshape(-1, 1) for i in Y_k_list]
        v = v.reshape(-1, 1)

        curr_S_k = torch.cat(S_k_list, dim=1)
        curr_Y_k = torch.cat(Y_k_list, dim=1)
        S_k_time_Y_k = torch.matmul(curr_S_k.T, curr_Y_k)
        S_k_time_S_k = torch.matmul(curr_S_k.T, curr_S_k)

        R_k = torch.triu(S_k_time_Y_k)
        L_k = S_k_time_Y_k - R_k
        sigma_k = torch.matmul(Y_k_list[-1].T, S_k_list[-1]) / (
            torch.matmul(S_k_list[-1].T, S_k_list[-1]) + 1e-12
        )
        D_k_diag = torch.diag(S_k_time_Y_k)
        upper_mat = torch.cat([sigma_k * S_k_time_S_k, L_k], dim=1)
        lower_mat = torch.cat([L_k.T, -torch.diag(D_k_diag)], dim=1)
        mat = torch.cat([upper_mat, lower_mat], dim=0)
        mat_inv = torch.linalg.inv(mat)

        approx_prod = sigma_k * v
        p_mat = torch.cat([torch.matmul(curr_S_k.T, sigma_k * v),
                           torch.matmul(curr_Y_k.T, v)], dim=0)
        approx_prod -= torch.matmul(torch.matmul(torch.cat([sigma_k *
                                                            curr_S_k, curr_Y_k], dim=1), mat_inv), p_mat)

        return approx_prod.squeeze()

    def _kmeans_fit_predict_gpu(self, data, n_clusters, n_init=10):
        """
        Use cuML KMeans on GPU and return (labels, inertia).
        """
        if not torch.cuda.is_available():
            raise RuntimeError("FLDetector requires CUDA for GPU KMeans.")
        try:
            import cupy as cp
            from cuml.cluster import KMeans as cuKMeans
        except Exception as exc:
            raise RuntimeError(
                "GPU KMeans requires `cupy` and `cuml`. Please install RAPIDS."
            ) from exc

        if not torch.is_tensor(data):
            data = torch.as_tensor(data, dtype=torch.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if not data.is_cuda:
            data = data.to(device="cuda", non_blocking=True)
        data = data.detach().contiguous()

        try:
            data_cp = cp.from_dlpack(data)
        except Exception:
            data_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(data))

        estimator = cuKMeans(n_clusters=n_clusters, n_init=n_init, output_type="cupy")
        labels_cp = estimator.fit_predict(data_cp)
        inertia_cp = cp.asarray(estimator.inertia_)

        try:
            labels = torch.from_dlpack(labels_cp).to(device=data.device, dtype=torch.long)
            inertia = torch.from_dlpack(inertia_cp).to(device=data.device, dtype=data.dtype).reshape(())
        except Exception:
            labels = torch.utils.dlpack.from_dlpack(labels_cp.toDlpack()).to(
                device=data.device, dtype=torch.long
            )
            inertia = torch.utils.dlpack.from_dlpack(inertia_cp.toDlpack()).to(
                device=data.device, dtype=data.dtype
            ).reshape(())
        return labels, inertia

    def gap_statistics(self, data, num_sampling, K_max, n):
        """
        计算 Gap Statistic 以估计最佳聚类簇数。

        参数:
            data (numpy.ndarray | Sequence[float]): 待评估的得分向量。
            num_sampling (int): 生成随机数据的样本次数。
            K_max (int): 评估的最大簇数上限。
            n (int): 样本数量，用于生成对照随机数据。

        返回:
            int: Gap Statistic 推荐的聚类簇数。

        复杂度:
            时间复杂度 O(K_max * num_sampling * n * d)；空间复杂度 O(n * d)。
        """
        # 数据归一化后避免尺度差异影响聚类。
        if not torch.is_tensor(data):
            data = torch.as_tensor(data, dtype=torch.float32)
        if not data.is_cuda:
            data = data.to(device="cuda", non_blocking=True)
        data = normalize_data(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        gaps, s = [], []
        K_max = min(K_max, int(data.shape[0]))

        for k in range(1, K_max + 1):
            # 真实数据的簇内误差 (inertia)。
            _, inertia = self._kmeans_fit_predict_gpu(data, n_clusters=k, n_init=10)

            # 随机数据的簇内误差，用于近似空模型。
            fake_inertia = []
            for _ in range(num_sampling):
                random_data = torch.rand(
                    (n, data.shape[1]), device=data.device, dtype=data.dtype
                )
                _, fake_inertia_k = self._kmeans_fit_predict_gpu(
                    random_data, n_clusters=k, n_init=10
                )
                fake_inertia.append(fake_inertia_k)

            fake_inertia = torch.stack(fake_inertia)
            mean_fake_inertia = torch.mean(fake_inertia)
            gap = torch.log(mean_fake_inertia + 1e-12) - torch.log(inertia + 1e-12)
            gaps.append(gap)

            sd = torch.std(torch.log(fake_inertia + 1e-12), unbiased=False)
            coeff = torch.sqrt(
                torch.tensor((1 + num_sampling) / num_sampling, device=data.device, dtype=data.dtype)
            )
            s.append(sd * coeff)

        num_cluster = 0
        for k in range(1, K_max):
            if (gaps[k - 1] - gaps[k] + s[k]).item() >= 0:
                num_cluster = k + 1
                break
        else:
            num_cluster = K_max
            print("FLDetector: No gap detected, No attack detected , return K_max")
        return num_cluster


def normalize_data(data):
    """
    将输入数据线性归一化至 [0, 1] 区间。

    参数:
        data (numpy.ndarray | Sequence[float]): 待归一化数据。

    返回:
        numpy.ndarray: 归一化后的数据数组。

    异常:
        ZeroDivisionError: 当数据最大值等于最小值时可能出现除零。

    复杂度:
        时间复杂度 O(n)，空间复杂度 O(n)。
    """
    if torch.is_tensor(data):
        min_val = torch.min(data)
        max_val = torch.max(data)
        return (data - min_val) / torch.clamp(max_val - min_val, min=1e-12)
    data = np.array(data)
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / max(max_val - min_val, 1e-12)


# 费曼学习法解释 (FLDetector.__init__)
# (A) 做什么：设定滑动窗口大小与检测起始轮次，并初始化历史缓存。
# (B) 类比：像安装监控系统前先决定录像循环天数与何时开始警戒。
# (C) 步骤：
#     1. 继承基类保存全局配置。
#     2. 设定默认窗口与起始轮次，允许外部覆盖。
#     3. 初始化历史梯度与恶意评分缓存。
# (D) 示例：
#     >>> class Args: num_clients=10; num_adv=2; defense_params=None
#     >>> detector = FLDetector(Args())
# (E) 边界/测试：缺少 defense_params 属性会报错；建议测试窗口参数覆盖是否成功。
# (F) 参考：《Manipulating the Byzantine》；有限内存优化教材。


# 费曼学习法解释 (FLDetector.aggregate)
# (A) 做什么：利用历史梯度预测误差筛除可疑客户端并聚合剩余梯度。
# (B) 类比：像根据过去的天气趋势预测今天的天气，再比较各地实测，剔除偏差过大的观测站。
# (C) 步骤：
#     1. 复制上一轮模型并记录当前轮次。
#     2. 将客户端更新转换为统一梯度格式。
#     3. 若历史足够，计算预测梯度与实际梯度的差距，记为恶意评分。
#     4. 恶意评分滑动窗口满后，通过 Gap Statistic 与 KMeans 标记可疑客户端。
#     5. 对标记为良性的客户端梯度取均值，并更新历史缓存。
# (D) 示例：
#     >>> aggregated = detector.aggregate(updates, last_global_model=model, global_epoch=60)
# (E) 边界/测试：窗口不足时仅做均值；建议测试检测阈值在攻击场景下是否生效。
# (F) 参考：《Manipulating the Byzantine》；时间序列异常检测文献。


# 费曼学习法解释 (FLDetector.get_pred_real_dists)
# (A) 做什么：比较预测梯度与实际梯度，得到每个客户端的差距得分。
# (B) 类比：像根据昨日车速预测今天，再对比真实车速，偏离越大越值得怀疑。
# (C) 步骤：
#     1. 用上一轮梯度加上 HVP 预测当前梯度。
#     2. 计算预测与实际梯度的欧氏距离。
#     3. 归一化距离以便不同轮次比较。
# (D) 示例：
#     >>> scores = detector.get_pred_real_dists(last_grad, current_grad, hvp)
# (E) 边界/测试：当距离和为 0 时需避免除零；建议测试纯良性与单个异常的差异。
# (F) 参考：梯度预测与 HVP 相关资料。


# 费曼学习法解释 (FLDetector.LBFGS)
# (A) 做什么：利用 L-BFGS 公式近似计算 Hessian 与向量的乘积。
# (B) 类比：像用过去的速度与加速度来推测当前的加速度效果。
# (C) 步骤：
#     1. 将历史参数差分和梯度差分整理成矩阵。
#     2. 计算必要的中间矩阵（R_k、L_k、D_k 等）。
#     3. 构造 L-BFGS 线性系统并求逆。
#     4. 根据公式组合出近似的矩阵向量积。
# (D) 示例：
#     >>> hvp = detector.LBFGS(S_list, Y_list, v)
# (E) 边界/测试：矩阵不可逆时会失败；建议测试窗口过短或向量线性相关的情况。
# (F) 参考：《Numerical Optimization》；LBFGS 相关教材。


# 费曼学习法解释 (FLDetector.gap_statistics)
# (A) 做什么：根据 Gap Statistic 判断得分数据的最佳聚类数。
# (B) 类比：像比较真实考试成绩和随机猜测成绩的差距来决定分几档。
# (C) 步骤：
#     1. 对数据归一化，避免尺度影响。
#     2. 针对每个候选簇数 k，计算真实数据和随机数据的簇内误差。
#     3. 计算 gap 与标准差，寻找差值开始下降的位置。
# (D) 示例：
#     >>> k_hat = detector.gap_statistics(score, num_sampling=20, K_max=10, n=len(score))
# (E) 边界/测试：若 gap 始终递增则返回 K_max；建议测试不同噪声水平下的稳定性。
# (F) 参考：《Gap Statistics》原始论文。


# 费曼学习法解释 (normalize_data)
# (A) 做什么：把数值拉伸到 0-1 范围，消除尺度差异。
# (B) 类比：像把不同单位的尺子统一到同一比例尺。
# (C) 步骤：
#     1. 计算最小值与最大值。
#     2. 做线性变换 (x - min)/(max - min)。
# (D) 示例：
#     >>> normalize_data([2, 4, 6])
#     array([0. , 0.5, 1. ])
# (E) 边界/测试：max 与 min 相等时会除零；建议添加保护或测试此场景。
# (F) 参考：标准化与数据预处理教材。


__AI_ANNOTATION_SUMMARY__ = """
FLDetector.__init__: 初始化滑动窗口与历史缓存以支撑梯度预测检测。
FLDetector.aggregate: 基于梯度预测误差与聚类筛除可疑客户端并聚合剩余梯度。
FLDetector.get_pred_real_dists: 计算预测与实际梯度的归一化距离得分。
FLDetector.LBFGS: 使用 L-BFGS 近似求解 Hessian-Vector Product。
FLDetector.gap_statistics: 依据 Gap Statistic 估计最优聚类簇数以触发检测。
normalize_data: 将输入数据线性归一化到 [0, 1] 区间。
"""
