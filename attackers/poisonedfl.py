# -*- coding: utf-8 -*-
import math
import torch
from attackers import attacker_registry
from attackers.pbases.mpbase import MPBase
from fl.client import Client
from global_utils import actor


@attacker_registry
@actor("attacker", "model_poisoning", "non_omniscient")
class PoisonedFL(MPBase, Client):
    """
    固定方向 + 动态缩放的模型投毒攻击（忠实还原 PoisonedFL/byzantine.py 流程，改用 PyTorch）。

    - 延续原流程：保留 fixed_rand 维度分支、k_95/k_99 阈值、历史梯度 history、last_grad、init_model、
      last_50_model、sf 动态缩放、deviation 计算等步骤。
    - 非全知：只用本地信息，但内部状态（history 等）按原逻辑维护，方便后续多客户端扩展。
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)

        self.default_attack_params = {
            # 原实现中外部传入，保持可配置；默认值与 MXNet 版一致
            "scaling_factor": 100000.0,
        }
        self.update_and_set_attr()

        # 持久状态，按原始实现命名
        self.fixed_rand = None
        self.history = None
        self.last_grad = None
        self.init_model = None
        # 模型快照队列，用于“距离上一轮或 50 轮前”的对比
        self._model_queue = []

    # --------- 工具函数（向量化辅助） ---------
    @staticmethod
    def _flatten_params(params):
        return torch.cat([p.view(-1) for p in params])

    @staticmethod
    def _unflatten_like(vec, like_params):
        # 将一维向量按照 like_params 的形状拆回（此处用于求形状，不回写模型）
        outputs = []
        offset = 0
        for p in like_params:
            num = p.numel()
            outputs.append(vec[offset:offset + num].view_as(p))
            offset += num
        return outputs

    def _num_attackers(self):
        # 与原实现类似：num_adv 若为比例则换算成整数
        adv = getattr(self.args, "num_adv", 1)
        if isinstance(adv, float) and adv < 1:
            adv = max(1, int(math.ceil(adv * self.args.num_clients)))
        return int(adv) if adv else 1

    # --------- 逻辑函数：与 PoisonedFL/byzantine.py 对应 ---------
    @staticmethod
    def _compute_lambda(all_updates, model_re, n_attackers):
        distances = []
        n_benign, d = all_updates.shape
        for update in all_updates:
            distance = (all_updates - update).norm(dim=1)
            distances.append(distance)
        distances = torch.stack(distances)
        distances = torch.sort(distances, dim=1).values
        scores = distances[:, : n_benign - 1 - n_attackers].sum(dim=1)
        min_score = scores.min()
        term_1 = min_score / ((n_benign - n_attackers - 1) * math.sqrt(d))
        max_wre_dist = ((all_updates - model_re).norm(dim=1)).max() / math.sqrt(d)
        return term_1 + max_wre_dist

    @staticmethod
    def _score(gradient, v, nbyz):
        num_neighbours = v.shape[0] - 2 - nbyz
        sorted_distance = torch.square(v - gradient).sum(dim=1).sort().values
        return sorted_distance[1 : 1 + num_neighbours].sum().item()

    def _poisonedfl(self, v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.0):
        # 1) k_95/k_99 动态计算（二项分布均值±Zσ近似），避免模型尺寸变动时抛异常
        n_dims = fixed_rand.numel()
        mu = 0.5 * n_dims
        sigma = math.sqrt(n_dims * 0.5 * 0.5)
        k_95 = int(round(mu + 1.645 * sigma))
        k_99 = int(round(mu + 2.326 * sigma))

        sf = scaling_factor

        # 2) 需历史信息才执行
        if isinstance(history, torch.Tensor):
            # 当前模型参数扁平化
            current_model_list = [p.detach().clone() for p in net.parameters()]
            current_model = self._flatten_params(current_model_list).view(-1, 1)
            last50_flat = self._flatten_params(last_50_model).view(-1, 1)

            history_norm = history.norm()
            last_grad_norm = last_grad.norm()
            # scale: || history - last_grad * (||history|| / ||last_grad||) ||
            scale = torch.norm(
                history - last_grad.unsqueeze(-1) * history_norm / (last_grad_norm + 1e-9),
                dim=1,
            )
            deviation = scale * fixed_rand / (scale.norm() + 1e-9)

            if e % 50 == 0:
                total_update = current_model - last50_flat
                total_update = torch.where(total_update == 0, current_model, total_update)
                current_sign = torch.sign(total_update)
                aligned_dim_cnt = (current_sign == fixed_rand.view(-1, 1)).sum()
                if aligned_dim_cnt < k_99 and scaling_factor * 0.7 >= 0.5:
                    sf = scaling_factor * 0.7
                else:
                    sf = scaling_factor
                lamda_succ = sf * history_norm
            else:
                sf = scaling_factor
                lamda_succ = sf * history_norm

            mal_update = lamda_succ * deviation  # shape: (dim,)

            for i in range(nfake):
                v[i] = mal_update  # 保持形状 (dim,)

        return v, sf

    @staticmethod
    def _random_attack(v, scaling_factor=100000.0):
        for i in range(v.shape[0]):
            v[i] = scaling_factor * torch.randn_like(v[0])
        return v, scaling_factor

    def _init_attack(self, v, net, nfake, init_model, scaling_factor=100000.0):
        current_model_list = [p.detach().clone() for p in net.parameters()]
        current_model = self._flatten_params(current_model_list).view(-1, 1)
        init_model_flat = self._flatten_params(init_model).view(-1, 1)
        direction = init_model_flat - current_model
        for i in range(nfake):
            v[i] = scaling_factor * direction.squeeze()
        return v, scaling_factor

    # --------- 攻击入口 ---------
    def non_omniscient(self) -> torch.Tensor:
        """
        入口：复现原 PoisonedFL 的 poisonedfl 分支（固定 random vector + 历史缩放）。
        """
        # self.update 可能是 torch.Tensor，也可能是 numpy.ndarray；统一转 torch.Tensor，并放到模型所在设备
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        benign_update = torch.as_tensor(self.update, device=model_device).flatten()
        device = benign_update.device

        # 初始化状态
        if self.fixed_rand is None:
            fr = torch.sign(torch.randn_like(benign_update))
            fr[fr == 0] = 1.0
            self.fixed_rand = fr.to(device)
        if self.init_model is None:
            self.init_model = [p.detach().clone() for p in self.model.parameters()]
        if not self._model_queue:
            self._model_queue.append([p.detach().clone() for p in self.model.parameters()])
        if self.history is None:
            # history 期望形状与 deviation 计算一致，转成 (dim,1)
            self.history = benign_update.view(-1, 1)
        if self.last_grad is None:
            self.last_grad = benign_update.clone()

        nfake = self._num_attackers()
        v = benign_update.new_zeros((nfake, benign_update.numel()))

        v_out, sf = self._poisonedfl(
            v,
            self.model,
            getattr(self.args, "learning_rate", None) or self.optimizer.param_groups[0]["lr"],
            nfake,
            self.history,
            self.fixed_rand,
            self.init_model,
            self._model_queue[0],
            self.last_grad,
            self.global_epoch,
            self.scaling_factor,
        )

        # 仅返回当前客户端的投毒更新（对其他攻击者可复用 v_out 中的副本）
        poisoned_vec = v_out[0].view_as(benign_update)

        # 更新内部状态以便下一轮
        self.last_grad = benign_update.detach()
        self.history = poisoned_vec.detach().view(-1, 1)
        # 更新模型快照队列，保留最近 50 轮前的快照位置
        self._model_queue.append([p.detach().clone() for p in self.model.parameters()])
        if len(self._model_queue) > 50:
            self._model_queue.pop(0)

        return poisoned_vec
