# -*- coding: utf-8 -*-

import torch
import numpy as np

from attackers import attacker_registry
from attackers.pbases.mpbase import MPBase
from fl.client import Client
from global_utils import actor


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class PoisonedFL(MPBase, Client):
    """
    PyTorch reimplementation of the PoisonedFL attack.
    The attacker aligns a scaled fixed random direction with the residual
    global update while adaptively shrinking the scaling factor every 50 rounds
    when the aggregated direction drifts away from the fixed sign pattern.
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        # 娌跨敤 MXNet 鍙傝€冨疄鐜扮殑榛樿鏀惧ぇ绯绘暟锛屾柟渚夸笌鍘熺粨鏋滃榻愶紱鑴氭湰鍙傛暟鍙鐩栦互鍋氭秷铻嶃€?
        self.default_attack_params = {"scaling_factor": 10.0}  
        self.update_and_set_attr()

        self.current_scaling_factor = float(self.scaling_factor)
        # 鍥哄畾闅忔満鏂瑰悜锛埪?锛夛紝鍙湪棣栨璋冪敤鏃剁敓鎴愶紝淇濇寔鏀诲嚮鏂瑰悜鍏ㄧ▼涓€鑷翠互绋冲畾鎵板姩銆?
        self.fixed_rand = None
        # 璁板綍鍒濆/涓婁竴杞?鏈€杩?50 杞殑鍏ㄥ眬妯″瀷鍚戦噺锛岀敤浜庤绠楁畫宸笌婕傜Щ瀵归綈搴︺€?
        self.init_model_vec = None
        self.prev_global_vec = None
        self.last_50_global_vec = None
        # 缂撳瓨涓婁竴杞殑锛堟綔鍦級鎭舵剰姊害锛岀己澶辨椂鍥為€€涓鸿壇鎬ф洿鏂帮紝閬垮厤鏃犲巻鍙蹭俊鎭椂鐨勯渿鑽°€?
        self.last_grad_vec = None

    def omniscient(self, clients):
        attackers = [
            client for client in clients
            if client.category == "attacker"
        ]
        if not attackers:
            return None

        # 褰撳墠搴熸挱鐨勫叏灞€妯″瀷鍚戦噺锛屼綔涓烘湰杞绠?residual/婕傜Щ鐨勫熀鍑嗐€?
        device = self.args.device
        if torch.is_tensor(self.global_weights_vec):
            current_global_vec = self.global_weights_vec.detach().flatten().to(device)
        else:
            current_global_vec = torch.as_tensor(
                self.global_weights_vec, dtype=torch.float32, device=device
            ).flatten()

        # 棣栨杩涘叆鏃跺垵濮嬪寲鍥哄畾鏂瑰悜涓庡熀鍑嗗揩鐓с€?
        if self.fixed_rand is None:
            # 涓?MXNet 鐗堟湰涓€鑷达細闅忔満绗﹀彿鍚戦噺锛宻ign 淇濊瘉鍙湁 卤1銆?
            self.fixed_rand = torch.sign(torch.randn_like(current_global_vec))
            # 鏋佸皬姒傜巼鍑虹幇 0锛岀敤 where 杞垚 浠ヤ繚鎸佺鍙风ǔ瀹氥€?
            zero_mask = self.fixed_rand == 0
            if torch.any(zero_mask):
                self.fixed_rand = torch.where(
                    zero_mask, torch.ones_like(self.fixed_rand), self.fixed_rand
                )
            self.init_model_vec = current_global_vec.clone()
            self.last_50_global_vec = current_global_vec.clone()

            # XXX
            # self._log_message(f"fixed_rand[:100]={self.fixed_rand[:100].detach().cpu()}")

        # history 涓鸿繛缁袱杞叏灞€妯″瀷鐨勫樊鍊硷紝绛変环浜?MXNet 閲岀殑 current_model - last_model銆?
        history_vec = None
        if self.prev_global_vec is not None:
            history_vec = (current_global_vec - self.prev_global_vec).unsqueeze(1)

        # XXX锛氭祴璇?
        # print("prev_global_vec: ", self.prev_global_vec)
        # print("history_vec: ", history_vec)

        # 鏃犺杩斿洖涓庡惁锛岄兘鍏堟洿鏂?prev_global锛屼繚璇佷笅涓€杞湁鍙傜収銆?
        self.prev_global_vec = current_global_vec.clone()

        # 缂哄皯鍘嗗彶鎴栦笂涓€杞伓鎰忔搴︽椂锛屽厛杩斿洖鑹€ф洿鏂颁互缁存寔鏁板€肩ǔ瀹氥€?
        if history_vec is None or self.last_grad_vec is None:
            benign_updates = torch.stack(
                [
                    client.update.detach().to(device)
                    if torch.is_tensor(client.update)
                    else torch.as_tensor(client.update, device=device)
                    for client in attackers
                ],
                dim=0,
            )
            self.last_grad_vec = benign_updates.mean(dim=0)
            # 鍚屾 50 杞揩鐓э紝纭繚鑷€傚簲缂╂斁鍩轰簬缁熶竴鐨勫叏灞€鑺傚銆?
            if self.global_epoch % 50 == 0:
                self.last_50_global_vec = current_global_vec.clone()
            return benign_updates

        k_95, k_99 = self._get_thresholds(self.fixed_rand.numel())
        sf = float(self.current_scaling_factor)
        eps = 1e-9

        history_norm = torch.norm(history_vec)
        last_grad_norm = torch.norm(self.last_grad_vec)

        # 鍘绘帀涓婁竴杞搴︽柟鍚戝悗寰楀埌娈嬪樊锛屽啀鎸夊浐瀹氱鍙烽噸鏂板榻愶紝杩欏搴旀枃閲岀殑 fixed direction 鎶曞奖銆?
        residual = history_vec.squeeze(1) - self.last_grad_vec * (
            history_norm / (last_grad_norm + eps)
        )
        scale = torch.norm(residual.unsqueeze(1), dim=1)
        deviation = scale * self.fixed_rand / (torch.norm(scale) + eps)

        current_epoch = int(self.global_epoch)
        if current_epoch % 50 == 0:
            # 姣?50 杞鏌ヤ竴娆″浐瀹氭柟鍚戠殑瀵归綈搴︼紝鍋忕Щ杩囧ぇ鍒欐寚鏁拌“鍑?scaling_factor锛岄槻姝㈡搴︽紓绉汇€?
            total_update = current_global_vec - self.last_50_global_vec
            replaced = torch.where(total_update == 0, current_global_vec, total_update)
            current_sign = torch.sign(replaced)
            aligned_dim_cnt = int((current_sign == self.fixed_rand).sum().item())
            if aligned_dim_cnt < k_99 and sf * 0.7 >= 0.5:
                sf = sf * 0.7
            lamda_succ = sf * history_norm
        else:
            lamda_succ = sf * history_norm

        # 鎸夊浐瀹氭柟鍚戠敓鎴愭伓鎰忔洿鏂帮紝骞跺鍒跺埌鎵€鏈夋敾鍑昏€咃紝澶嶇敤浜嗚壇鎬?update 鐨勫舰鐘朵互鍏煎鑱氬悎銆?
        mal_update = lamda_succ * deviation
        malicious_updates = mal_update.detach().unsqueeze(0).repeat(len(attackers), 1)

        # if current_epoch % 20 == 0:
        #     # XXX:娴嬭瘯
        #     mal_str = np.array2string(
        #         mal_update[:100].detach().cpu().numpy(),
        #         separator=" ",
        #         max_line_width=1000,
        #         formatter={"float_kind": lambda x: f"{x:.2e}"},
        #     )
        #     self._log_message(f"mal_update={mal_str}")
        #     update = attackers[0].update
        #     if torch.is_tensor(update):
        #         attacker_update = update.detach().cpu().numpy()
        #     else:
        #         attacker_update = np.array(update, copy=True)
        #     benign_update_norm = float(np.linalg.norm(attacker_update))
        #     benign_str = np.array2string(
        #         attacker_update[:100],
        #         separator=" ",
        #         max_line_width=1000,
        #         formatter={"float_kind": lambda x: f"{x:.2e}"},
        #     )
        #     self._log_message(f"benign_update={benign_str}")
        #     mal_update_norm = float(mal_update.norm().item())
        #     ratio = mal_update_norm / (benign_update_norm + 1e-12)
        #     self._log_message(f"mal_benign_l2_ratio={ratio}")
        #     self._log_message(f"lamda_succ={lamda_succ}")

        # 鎸佷箙鍖栫姸鎬侊紝涓嬩竴杞户缁熀浜庡悓涓€鏂瑰悜涓庣缉鏀剧郴鏁拌凯浠ｃ€?
        self.current_scaling_factor = sf
        self.last_grad_vec = malicious_updates[0].detach().clone()
        if current_epoch % 50 == 0:
            self.last_50_global_vec = current_global_vec.clone()

        return malicious_updates

    def _log_message(self, msg):
        logger = getattr(self.args, "logger", None)
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
    @staticmethod
    def _get_thresholds(dim):
        thresholds = {
            1204682: (603244, 603618),
            139960: (70288, 70415),
            717924: (359659, 359948),
            145212: (72919, 73049),
            61706: (31057, 31142),
            62006: (31157, 31242),
            11183582: (5594543, 5595682),
        }
        if dim not in thresholds:
            raise NotImplementedError(
                f"Unsupported fixed_rand dimension {dim} for PoisonedFL thresholds."
            )
        return thresholds[dim]

