"""
联邦学习攻击超参数搜索入口。

设计目标:
    - 保持与 main.py 尽量一致的命令行参数、训练流程、日志风格与目录结构；
    - 将超参数搜索控制器放在主流程外层，最大程度复用现有工程；
    - 第一版优先支持 COMPASS，并预留插件式扩展点以支持其他攻击。
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import json
import logging
import math
import random
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from aggregators import all_aggregators
from attackers import data_poisoning_attacks, model_poisoning_attacks
from attackers.compass import COMPASS_DEFAULT_ATTACK_PARAMS
from fl.algorithms import all_algorithms
from fl.models import all_models
from global_args import read_yaml, single_preprocess


DATASET_SUCCESS_THRESHOLDS: Dict[str, float] = {
    "5GNIDD": 0.5,
}

HYPER_ARG_NAMES = {
    "max_search_trials",
    "search_patience",
    "success_acc_threshold",
    "success_consecutive_rounds",
    "success_eval_start_round",
    "search_step_decay",
}
HARD_SEARCH_TRIAL_LIMIT = 100


@dataclass(frozen=True)
class SearchParamSpec:
    name: str
    min_value: float
    max_value: float
    default_step: float
    min_step: float
    decimals: int = 4


@dataclass
class TrialResult:
    trial_idx: int
    params: Dict[str, float]
    output_path: Path
    success: bool
    max_streak: int
    min_test_acc: float
    final_test_acc: float
    success_window: Optional[Tuple[int, int]]
    epochs: List[int]
    test_accs: List[float]
    sf_values: List[float]
    matched_filter_scores: List[float]
    error: Optional[str] = None
    from_index: bool = False

    @property
    def matched_filter_mean(self) -> Optional[float]:
        if not self.matched_filter_scores:
            return None
        return sum(self.matched_filter_scores) / len(self.matched_filter_scores)

    @property
    def matched_filter_last(self) -> Optional[float]:
        if not self.matched_filter_scores:
            return None
        return self.matched_filter_scores[-1]


class AttackSearchPlugin:
    """
    攻击搜索插件基类。

    每种攻击可以定义:
        - 可搜索参数空间；
        - 邻域生成逻辑；
        - 附加日志信号解析与候选排序规则。
    """

    attack_name = "BASE"
    param_specs: Sequence[SearchParamSpec] = ()

    def default_search_params(self) -> Dict[str, float]:
        return {}

    def initialize_params(self, raw_params: Optional[Dict[str, object]]) -> Dict[str, object]:
        params = copy.deepcopy(raw_params or {})
        for spec in self.param_specs:
            params.setdefault(spec.name, self.default_search_params().get(spec.name))
        missing = [spec.name for spec in self.param_specs if params.get(spec.name) is None]
        if missing:
            raise ValueError(
                f"Attack {self.attack_name} requires initial values for searchable params: {missing}"
            )
        return self._normalize_params(params)

    def initial_steps(self) -> Dict[str, float]:
        return {spec.name: spec.default_step for spec in self.param_specs}

    def reduce_steps(self, steps: Dict[str, float], decay: float) -> Dict[str, float]:
        reduced = {}
        for spec in self.param_specs:
            reduced[spec.name] = max(spec.min_step, steps[spec.name] * decay)
            reduced[spec.name] = self._quantize(spec, reduced[spec.name])
        return reduced

    def can_reduce_steps(self, steps: Dict[str, float]) -> bool:
        for spec in self.param_specs:
            if steps[spec.name] > spec.min_step + 10 ** (-(spec.decimals + 2)):
                return True
        return False

    def should_expand_search(
        self,
        current_result: TrialResult,
        best_result: Optional[TrialResult],
        current_steps: Dict[str, float],
    ) -> bool:
        return False

    def expand_steps(self, steps: Dict[str, float]) -> Dict[str, float]:
        expanded = {}
        for spec in self.param_specs:
            expanded_value = max(steps[spec.name] * 2.0, spec.default_step)
            max_reasonable_step = max(spec.min_step, (spec.max_value - spec.min_value) / 3.0)
            expanded[spec.name] = self._quantize(spec, min(expanded_value, max_reasonable_step))
        return expanded

    def make_key(self, params: Dict[str, object]) -> Tuple[Tuple[str, object], ...]:
        normalized = self._normalize_params(params)
        items = []
        for key in sorted(normalized):
            value = normalized[key]
            if isinstance(value, float):
                value = round(value, 8)
            items.append((key, value))
        return tuple(items)

    def generate_neighbors(
        self,
        current_params: Dict[str, object],
        steps: Dict[str, float],
        tried_keys: set,
    ) -> List[Dict[str, object]]:
        neighbors: List[Dict[str, object]] = []
        for spec in self.param_specs:
            step = steps[spec.name]
            if step <= 0:
                continue
            for direction in (1.0, -1.0):
                candidate = copy.deepcopy(current_params)
                current_value = float(candidate[spec.name])
                next_value = self._quantize(spec, current_value + direction * step)
                if next_value < spec.min_value or next_value > spec.max_value:
                    continue
                if math.isclose(next_value, current_value, rel_tol=0.0, abs_tol=10 ** (-(spec.decimals + 1))):
                    continue
                candidate[spec.name] = next_value
                candidate = self._normalize_params(candidate)
                candidate_key = make_index_key(self, candidate)
                if candidate_key not in tried_keys:
                    neighbors.append(candidate)
        return neighbors

    def should_force_breakthrough(
        self,
        current_result: TrialResult,
        best_result: Optional[TrialResult],
        current_steps: Dict[str, float],
    ) -> bool:
        return False

    def compare_results(self, left: TrialResult, right: TrialResult) -> int:
        left_score = self.score_result(left)
        right_score = self.score_result(right)
        if left_score > right_score:
            return 1
        if left_score < right_score:
            return -1
        return 0

    def score_result(self, result: TrialResult) -> Tuple[object, ...]:
        return (
            1 if result.success else 0,
            result.max_streak,
            -result.min_test_acc,
        )

    def metrics_summary(self, result: TrialResult) -> str:
        return ""

    def sample_untried_params(
        self,
        tried_keys: set,
        trial_results: Optional[Sequence[TrialResult]] = None,
        best_result: Optional[TrialResult] = None,
        current_steps: Optional[Dict[str, float]] = None,
        max_attempts: int = 512,
    ) -> Optional[Dict[str, object]]:
        """
        在全局搜索空间中随机采样一个尚未尝试过的参数点。

        当局部邻域已经找不到可用候选点时，用它做一次“重启”，避免搜索过早结束。
        """
        if not self.param_specs:
            return None

        for _ in range(max_attempts):
            candidate: Dict[str, object] = {}
            for spec in self.param_specs:
                scale = 10 ** spec.decimals
                min_int = int(round(spec.min_value * scale))
                max_int = int(round(spec.max_value * scale))
                candidate[spec.name] = round(random.randint(min_int, max_int) / scale, spec.decimals)
            candidate = self._normalize_params(candidate)
            if make_index_key(self, candidate) not in tried_keys:
                return candidate
        return None

    def _normalize_params(self, params: Dict[str, object]) -> Dict[str, object]:
        normalized = copy.deepcopy(params)
        for spec in self.param_specs:
            if spec.name in normalized and normalized[spec.name] is not None:
                normalized[spec.name] = self._quantize(spec, float(normalized[spec.name]))
        return normalized

    @staticmethod
    def _quantize(spec: SearchParamSpec, value: float) -> float:
        return round(float(value), spec.decimals)


class CompassSearchPlugin(AttackSearchPlugin):
    attack_name = "COMPASS"
    param_specs = (
        SearchParamSpec("scaling_factor", min_value=1.0, max_value=100.0, default_step=5, min_step=1, decimals=4),
        SearchParamSpec("important_magnitude", min_value=3.0, max_value=100.0, default_step=1.0, min_step=0.2, decimals=4),
        SearchParamSpec("unimportant_magnitude", min_value=0.8, max_value=1.3, default_step=0.2, min_step=0.1, decimals=4),
    )

    def initialize_params(self, raw_params: Optional[Dict[str, object]]) -> Dict[str, object]:
        params = {
            spec.name: COMPASS_DEFAULT_ATTACK_PARAMS.get(spec.name)
            for spec in self.param_specs
        }
        params.update(copy.deepcopy(raw_params or {}))
        missing = [spec.name for spec in self.param_specs if params.get(spec.name) is None]
        if missing:
            raise ValueError(
                f"Attack {self.attack_name} requires initial values for searchable params: {missing}"
            )
        return self._normalize_params(params)

    @staticmethod
    def _finite_or(value: float, default: float) -> float:
        return value if math.isfinite(value) else default

    @staticmethod
    def _tail_mean(values: Sequence[float], window: int) -> Optional[float]:
        if not values:
            return None
        tail = values[-window:]
        return sum(tail) / len(tail)

    @classmethod
    def _matched_quality(cls, result: TrialResult) -> float:
        """
        Compress noisy matched-filter signals into a stable [0, 1]-ish score.

        The mean captures sustained aggregation entry, the tail mean captures
        recent behavior, and the last value keeps the search sensitive to a
        late breakthrough without letting one noisy round dominate completely.
        """
        if not result.matched_filter_scores:
            return 0.0
        matched_mean = result.matched_filter_mean or 0.0
        matched_tail = cls._tail_mean(result.matched_filter_scores, 20) or matched_mean
        matched_last = result.matched_filter_last or matched_tail
        positive_ratio = sum(1 for score in result.matched_filter_scores if score > 0.0) / len(result.matched_filter_scores)
        raw_score = 0.45 * matched_mean + 0.35 * matched_tail + 0.20 * matched_last
        return 0.5 + 0.5 * math.tanh(raw_score) + 0.25 * positive_ratio

    @classmethod
    def _accuracy_damage(cls, result: TrialResult) -> float:
        """
        Reward both deep drops and stable low final accuracy.

        min_test_acc alone can over-value a one-round collapse, while
        final/tail accuracy prevents the search from chasing unstable flukes.
        """
        min_acc = cls._finite_or(result.min_test_acc, 1.0)
        final_acc = cls._finite_or(result.final_test_acc, 1.0)
        tail_acc = cls._tail_mean(result.test_accs, 20)
        tail_acc = cls._finite_or(tail_acc if tail_acc is not None else final_acc, final_acc)
        return 0.45 * (1.0 - min_acc) + 0.35 * (1.0 - tail_acc) + 0.20 * (1.0 - final_acc)

    @classmethod
    def _objective_score(cls, result: TrialResult) -> float:
        attack_damage = cls._accuracy_damage(result)
        matched_quality = cls._matched_quality(result)
        streak_bonus = min(result.max_streak, 20) / 20.0
        return 0.55 * attack_damage + 0.35 * matched_quality + 0.10 * streak_bonus

    @classmethod
    def _dominates(cls, left: TrialResult, right: TrialResult) -> bool:
        left_mean = left.matched_filter_mean if left.matched_filter_mean is not None else float("-inf")
        right_mean = right.matched_filter_mean if right.matched_filter_mean is not None else float("-inf")
        left_tail = cls._tail_mean(left.matched_filter_scores, 20)
        right_tail = cls._tail_mean(right.matched_filter_scores, 20)
        left_tail = left_tail if left_tail is not None else float("-inf")
        right_tail = right_tail if right_tail is not None else float("-inf")

        no_worse = (
            left.min_test_acc <= right.min_test_acc
            and left.final_test_acc <= right.final_test_acc
            and left_mean >= right_mean
            and left_tail >= right_tail
            and left.max_streak >= right.max_streak
        )
        strictly_better = (
            left.min_test_acc < right.min_test_acc
            or left.final_test_acc < right.final_test_acc
            or left_mean > right_mean
            or left_tail > right_tail
            or left.max_streak > right.max_streak
        )
        return no_worse and strictly_better

    @classmethod
    def _pareto_front(cls, results: Sequence[TrialResult], limit: int = 8) -> List[TrialResult]:
        usable = [result for result in results if not result.error and result.params]
        front = []
        for candidate in usable:
            if any(cls._dominates(other, candidate) for other in usable if other is not candidate):
                continue
            front.append(candidate)
        front.sort(key=lambda result: cls._objective_score(result), reverse=True)
        return front[:limit]

    @staticmethod
    def _matched_signal_is_weak(result: Optional[TrialResult]) -> bool:
        if result is None:
            return False
        matched_mean = result.matched_filter_mean
        matched_last = result.matched_filter_last
        if matched_mean is None or matched_last is None:
            return False
        return matched_mean < 0.02 and matched_last < 0.02

    def should_expand_search(
        self,
        current_result: TrialResult,
        best_result: Optional[TrialResult],
        current_steps: Dict[str, float],
    ) -> bool:
        candidate = best_result if best_result is not None else current_result
        if not self._matched_signal_is_weak(candidate):
            return False
        for spec in self.param_specs:
            if current_steps[spec.name] + 10 ** (-(spec.decimals + 2)) < spec.default_step:
                return True
        return False

    def should_force_breakthrough(
        self,
        current_result: TrialResult,
        best_result: Optional[TrialResult],
        current_steps: Dict[str, float],
    ) -> bool:
        candidate = best_result if best_result is not None else current_result
        if candidate.success:
            return False
        if candidate.max_streak >= 18:
            return False
        scaling_step = current_steps.get("scaling_factor", 0.0)
        important_step = current_steps.get("important_magnitude", 0.0)
        return scaling_step <= 10.0 and important_step <= 2.0

    def expand_steps(self, steps: Dict[str, float]) -> Dict[str, float]:
        expanded = {}
        for spec in self.param_specs:
            growth = 3.0 if spec.name in {"scaling_factor", "important_magnitude"} else 2.0
            floor = spec.default_step * (2.0 if spec.name in {"scaling_factor", "important_magnitude"} else 1.0)
            expanded_value = max(steps[spec.name] * growth, floor)
            max_reasonable_step = max(spec.min_step, (spec.max_value - spec.min_value) / 2.0)
            expanded[spec.name] = self._quantize(spec, min(expanded_value, max_reasonable_step))
        return expanded

    def score_result(self, result: TrialResult) -> Tuple[object, ...]:
        matched_mean = result.matched_filter_mean
        matched_last = result.matched_filter_last
        matched_tail = self._tail_mean(result.matched_filter_scores, 20)
        return (
            1 if result.success else 0,
            self._objective_score(result),
            self._accuracy_damage(result),
            self._matched_quality(result),
            -result.min_test_acc,
            -result.final_test_acc,
            result.max_streak,
            matched_tail if matched_tail is not None else float("-inf"),
            matched_mean if matched_mean is not None else float("-inf"),
            matched_last if matched_last is not None else float("-inf"),
        )

    def metrics_summary(self, result: TrialResult) -> str:
        if not result.matched_filter_scores:
            return ""
        mean_score = result.matched_filter_mean
        last_score = result.matched_filter_last
        tail_score = self._tail_mean(result.matched_filter_scores, 20)
        suffix = f"  sf_last: {result.sf_values[-1]:.4f}" if result.sf_values else ""
        return (
            f"  objective_score: {self._objective_score(result):.4f}"
            f"  matched_filter_mean: {mean_score:.4f}"
            f"  matched_filter_tail20: {tail_score:.4f}"
            f"  matched_filter_last: {last_score:.4f}{suffix}"
        )

    def generate_neighbors(
        self,
        current_params: Dict[str, object],
        steps: Dict[str, float],
        tried_keys: set,
    ) -> List[Dict[str, object]]:
        neighbors = super().generate_neighbors(current_params, steps, tried_keys)

        # Add coupled moves because COMPASS parameters often only help when
        # attack strength and masking magnitude move together.
        coupled_specs = [
            ("scaling_factor", "important_magnitude", 1.0, 1.0),
            ("scaling_factor", "important_magnitude", 1.0, -1.0),
            ("scaling_factor", "unimportant_magnitude", 1.0, -1.0),
            ("important_magnitude", "unimportant_magnitude", 1.0, -1.0),
        ]
        specs_by_name = {spec.name: spec for spec in self.param_specs}
        for first, second, first_direction, second_direction in coupled_specs:
            candidate = copy.deepcopy(current_params)
            first_spec = specs_by_name[first]
            second_spec = specs_by_name[second]
            first_value = self._quantize(first_spec, float(candidate[first]) + first_direction * steps[first])
            second_value = self._quantize(second_spec, float(candidate[second]) + second_direction * steps[second])
            if not (first_spec.min_value <= first_value <= first_spec.max_value):
                continue
            if not (second_spec.min_value <= second_value <= second_spec.max_value):
                continue
            candidate[first] = first_value
            candidate[second] = second_value
            candidate = self._normalize_params(candidate)
            if make_index_key(self, candidate) not in tried_keys:
                neighbors.append(candidate)

        unique_neighbors = []
        seen = set()
        for candidate in neighbors:
            key = make_index_key(self, candidate)
            if key in seen:
                continue
            seen.add(key)
            unique_neighbors.append(candidate)
        return unique_neighbors

    def sample_untried_params(
        self,
        tried_keys: set,
        trial_results: Optional[Sequence[TrialResult]] = None,
        best_result: Optional[TrialResult] = None,
        current_steps: Optional[Dict[str, float]] = None,
        max_attempts: int = 512,
    ) -> Optional[Dict[str, object]]:
        """
        为 COMPASS 做启发式全局重启：
        优先围绕 Pareto 前沿和综合目标高的历史试验采样，避免只沿着
        单一 accuracy 或 matched_filter_score 指标陷入局部区域。
        """
        usable_results = [result for result in trial_results or [] if not result.error and result.params]
        pareto_anchors = self._pareto_front(usable_results, limit=8)
        scored_anchors = sorted(usable_results, key=self.score_result, reverse=True)[:8]

        anchors: List[TrialResult] = []
        for result in [*(pareto_anchors or []), *scored_anchors]:
            if result not in anchors:
                anchors.append(result)
        if best_result is not None and best_result not in anchors and not best_result.error:
            anchors.insert(0, best_result)

        if not anchors:
            return super().sample_untried_params(
                tried_keys=tried_keys,
                trial_results=trial_results,
                best_result=best_result,
                current_steps=current_steps,
                max_attempts=max_attempts,
            )

        step_map = current_steps or self.initial_steps()
        anchor_result = best_result if best_result is not None else (anchors[0] if anchors else None)
        weak_signal = self._matched_signal_is_weak(anchor_result)
        breakthrough_mode = self.should_force_breakthrough(
            current_result=anchor_result if anchor_result is not None else anchors[0],
            best_result=best_result,
            current_steps=step_map,
        ) if anchors else False
        for _ in range(max_attempts):
            anchor_pool = anchors[: min(6, len(anchors))]
            weights = [max(0.05, self._objective_score(anchor)) for anchor in anchor_pool]
            anchor = random.choices(anchor_pool, weights=weights, k=1)[0]
            candidate = copy.deepcopy(anchor.params)

            # 当 matched_filter_score 很弱时，局部细搜通常只会在无效区域原地打转。
            # 这时强制把搜索半径拉大，并优先把更关键的攻击强度参数往更激进方向推进。
            for spec in self.param_specs:
                base_value = float(candidate[spec.name])
                base_step = max(step_map.get(spec.name, spec.default_step), spec.min_step)
                matched_quality = self._matched_quality(anchor)
                accuracy_damage = self._accuracy_damage(anchor)

                if breakthrough_mode and spec.name == "scaling_factor":
                    jump_scale = random.choice([2.0, 3.0, 4.0, 5.0])
                    direction = random.choice([1.0, -1.0])
                    radius = jump_scale * max(base_step, spec.default_step)
                    jitter = random.uniform(-0.5 * base_step, 0.5 * base_step)
                    next_value = self._quantize(spec, base_value + direction * radius + jitter)
                    next_value = min(spec.max_value, max(spec.min_value, next_value))
                    if not math.isclose(next_value, base_value, rel_tol=0.0, abs_tol=10 ** (-(spec.decimals + 1))):
                        candidate[spec.name] = next_value
                    continue
                if breakthrough_mode and spec.name == "important_magnitude":
                    jump_scale = random.choice([1.5, 2.0, 3.0])
                    direction = random.choice([1.0, -1.0])
                    radius = jump_scale * max(base_step, spec.default_step)
                    jitter = random.uniform(-0.35 * base_step, 0.35 * base_step)
                    next_value = self._quantize(spec, base_value + direction * radius + jitter)
                    next_value = min(spec.max_value, max(spec.min_value, next_value))
                    if not math.isclose(next_value, base_value, rel_tol=0.0, abs_tol=10 ** (-(spec.decimals + 1))):
                        candidate[spec.name] = next_value
                    continue
                if weak_signal:
                    if spec.name in {"scaling_factor", "important_magnitude"}:
                        base_step = max(base_step, spec.default_step * 2.0)
                        direction = random.choices([1.0, -1.0], weights=[0.85, 0.15], k=1)[0]
                        radius = random.choice([2.0, 4.0, 6.0]) * base_step
                    else:
                        base_step = max(base_step, spec.default_step)
                        direction = random.choice([1.0, -1.0])
                        radius = random.choice([2.0, 3.0, 4.0]) * base_step
                else:
                    if spec.name in {"scaling_factor", "important_magnitude"}:
                        if matched_quality >= 0.65 and accuracy_damage < 0.45:
                            direction = random.choices([1.0, -1.0], weights=[0.75, 0.25], k=1)[0]
                        else:
                            direction = random.choices([1.0, -1.0], weights=[0.60, 0.40], k=1)[0]
                        radius = random.choice([1.0, 2.0, 3.0]) * base_step
                    else:
                        direction = random.choice([1.0, -1.0])
                        radius = random.choice([1.0, 2.0]) * base_step

                jitter = random.uniform(-0.25 * base_step, 0.25 * base_step)
                next_value = self._quantize(spec, base_value + direction * radius + jitter)
                if spec.min_value <= next_value <= spec.max_value:
                    candidate[spec.name] = next_value

            candidate = self._normalize_params(candidate)
            key = make_index_key(self, candidate)
            if key not in tried_keys:
                return candidate

        return super().sample_untried_params(
            tried_keys=tried_keys,
            trial_results=trial_results,
            best_result=best_result,
            current_steps=current_steps,
            max_attempts=max_attempts,
        )


PLUGIN_REGISTRY = {
    "COMPASS": CompassSearchPlugin(),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hyper-parameter search runner for FL poisoning attacks"
    )
    all_attacks = ["NoAttack"] + model_poisoning_attacks + data_poisoning_attacks

    parser.add_argument("-config", "--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("-b", "--benchmark", default=False, type=bool, help="Run all combinations of attacks and defenses")
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-seed", "--seed", type=int)
    parser.add_argument("-alg", "--algorithm", choices=all_algorithms)
    parser.add_argument("-opt", "--optimizer", choices=["SGD", "Adam"], help="optimizer for training")
    parser.add_argument("-lr_scheduler", "--lr_scheduler", type=str, help="lr_scheduler for training")
    parser.add_argument("-milestones", "--milestones", type=int, nargs="+", help="milestone for learning rate scheduler")
    parser.add_argument("-num_clients", "--num_clients", type=int, help="number of participating clients")
    parser.add_argument("-bs", "--batch_size", type=int, help="batch_size")
    parser.add_argument("-lr", "--learning_rate", type=float, help="initial learning rate")
    parser.add_argument("-le", "--local_epochs", type=int, help="local global_epoch")
    parser.add_argument("-model", "--model", choices=all_models)
    parser.add_argument("-data", "--dataset", choices=["MNIST", "FashionMNIST", "CIFAR10", "CINIC10", "CIFAR100", "CIFAR20", "CIFAR50", "EMNIST", "CHMNIST", "5GNIDD"])
    parser.add_argument("-dtb", "--distribution", choices=["iid", "class-imbalanced_iid", "non-iid", "pat", "imbalanced_pat"])
    parser.add_argument("-dirichlet_alpha", "--dirichlet_alpha", type=float, help="smaller alpha for drichlet distribution, stronger heterogeneity")
    parser.add_argument("-im_iid_gamma", "--im_iid_gamma", type=float, help="smaller alpha for class imbalanced distribution")
    parser.add_argument("-att", "--attack", choices=all_attacks, help="Attacks options")
    parser.add_argument("-attack_start_epoch", "--attack_start_epoch", type=int, help="the attack start epoch")
    parser.add_argument("-attparam", "--attparam", type=float, help="scale for omniscient model poisoning attack")
    parser.add_argument("-def", "--defense", choices=all_aggregators, help="Defenses options")
    parser.add_argument("-num_adv", "--num_adv", type=float, help="the proportion (float < 1) or number (int>1) of adversaries")
    parser.add_argument("-o", "--output", type=str, help="output file for results")
    parser.add_argument("-prate", "--poisoning_ratio", help="poisoning portion")
    parser.add_argument("--target_label", type=int, help="The No. of target label for backdoored images")
    parser.add_argument("--trigger_path", help="Trigger Path")
    parser.add_argument("--trigger_size", type=int, help="Trigger Size")
    parser.add_argument("-gidx", "--gpu_idx", type=int, nargs="+", help="Index of GPU")
    parser.add_argument("-defense_params", "--defense_params", type=str, help="Override defense parameters")
    parser.add_argument("-attack_params", "--attack_params", type=str, help="Override attack parameters")

    parser.add_argument(
        "--max_search_trials",
        type=int,
        default=0,
        help=(
            "Maximum number of hyper-parameter trials; <= 0 uses the hard default limit "
            f"of {HARD_SEARCH_TRIAL_LIMIT}"
        ),
    )
    parser.add_argument("--search_patience", type=int, default=2, help="How many local-search rounds without improvement before shrinking step size")
    parser.add_argument("--success_acc_threshold", type=float, default=0.18, help="Attack success threshold on test accuracy")
    parser.add_argument("--success_consecutive_rounds", type=int, default=20, help="When all rounds finish, require the final N rounds to stay below threshold")
    parser.add_argument("--success_eval_start_round", type=int, default=50, help="Deprecated: retained only for backward-compatible configs")
    parser.add_argument("--search_step_decay", type=float, default=0.5, help="Step decay factor after patience is exhausted")
    return parser


def read_hyper_args() -> Tuple[SimpleNamespace, argparse.Namespace]:
    parser = build_parser()
    cli_args = parser.parse_args()
    args = read_yaml(cli_args.config) if cli_args.config else SimpleNamespace()
    return args, cli_args


def parse_dict_arg(raw_value: Optional[str], arg_name: str) -> Optional[Dict[str, object]]:
    if raw_value is None:
        return None
    try:
        parsed = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Failed to parse {arg_name}: expected a Python dict string, got {raw_value!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Failed to parse {arg_name}: expected a dict, got {type(parsed).__name__}")
    return parsed


def get_default_named_params(args: SimpleNamespace, param_type: str, selected_name: Optional[str]) -> Optional[Dict[str, object]]:
    if selected_name is None:
        return None
    options = getattr(args, f"{param_type}s", None)
    if not options:
        return None
    for item in options:
        if item.get(param_type) == selected_name:
            params = item.get(f"{param_type}_params")
            return copy.deepcopy(params) if params is not None else None
    return None


def merge_cli_args(args: SimpleNamespace, cli_args: argparse.Namespace) -> SimpleNamespace:
    merged = copy.deepcopy(args)
    parsed_attack_params = parse_dict_arg(cli_args.attack_params, "--attack_params")
    parsed_defense_params = parse_dict_arg(cli_args.defense_params, "--defense_params")

    if not hasattr(merged, "attack_params"):
        merged.attack_params = get_default_named_params(merged, "attack", getattr(merged, "attack", None))
    if not hasattr(merged, "defense_params"):
        merged.defense_params = get_default_named_params(merged, "defense", getattr(merged, "defense", None))

    for key, value in vars(cli_args).items():
        if key in {"config", "attack", "defense", "attack_params", "defense_params", *HYPER_ARG_NAMES}:
            continue
        if value is not None:
            setattr(merged, key, value)
            print(f"Warning: Overriding {key} with {value}")

    if cli_args.attack:
        merged.attack = cli_args.attack
    if cli_args.defense:
        merged.defense = cli_args.defense

    if parsed_attack_params is not None:
        merged.attack_params = parsed_attack_params
    elif cli_args.attack:
        merged.attack_params = get_default_named_params(merged, "attack", merged.attack)

    if parsed_defense_params is not None:
        merged.defense_params = parsed_defense_params
    elif cli_args.defense:
        merged.defense_params = get_default_named_params(merged, "defense", merged.defense)

    # Hyper-search-only arguments are not part of the YAML config, but later
    # summary/diagnostic code reads them from the run args object as well.
    # Copy them onto merged args so per-trial namespaces always carry them.
    for key in HYPER_ARG_NAMES:
        setattr(merged, key, getattr(cli_args, key))

    return merged


def cleanup_logger(logger_name: str) -> None:
    logger = logging.getLogger(logger_name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def parse_trial_log(log_path: Path) -> Tuple[List[int], List[float], List[float], List[float]]:
    import re

    content = log_path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)\s+.*?Test Acc:\s+(?P<test_acc>[-+eE0-9\.]+)",
        re.MULTILINE,
    )
    epochs: List[int] = []
    test_accs: List[float] = []
    for match in pattern.finditer(content):
        epochs.append(int(match.group("epoch")))
        test_accs.append(float(match.group("test_acc")))

    if not test_accs:
        raise ValueError(f"Unable to parse test accuracy from log: {log_path}")

    sf_values = [float(v) for v in re.findall(r"\bsf=([-+eE0-9\.]+)", content)]
    matched_scores = [
        float(v)
        for v in re.findall(r"\b(?:mutched_filter_score|matched_filter_score)=([-+eE0-9\.]+)", content)
    ]
    return epochs, test_accs, sf_values, matched_scores


def evaluate_attack_success(
    epochs: Sequence[int],
    test_accs: Sequence[float],
    threshold: float,
    consecutive_rounds: int,
    total_rounds: int,
) -> Tuple[bool, int, float, Optional[Tuple[int, int]]]:
    if not epochs:
        return False, 0, float("inf"), None

    min_test_acc = min(test_accs)
    final_epoch = epochs[-1]
    expected_final_epoch = max(0, total_rounds - 1)
    finished_all_rounds = final_epoch >= expected_final_epoch

    # Rule 1: if training stops before all planned rounds finish, treat it as
    # attack success because the run was interrupted early by instability.
    if not finished_all_rounds:
        return True, 1, min_test_acc, (final_epoch, final_epoch)

    trailing_streak = 0
    for acc in reversed(test_accs):
        if acc < threshold:
            trailing_streak += 1
        else:
            break

    success_window = None
    if trailing_streak >= consecutive_rounds:
        start_epoch = epochs[len(epochs) - trailing_streak]
        success_window = (start_epoch, final_epoch)

    return success_window is not None, trailing_streak, min_test_acc, success_window


def append_lines(file_path: Path, lines: Iterable[str]) -> None:
    with file_path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def format_params(params: Dict[str, object]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=True)


def format_trial_summary(result: TrialResult, args: SimpleNamespace, total_trials: int) -> List[str]:
    lines = [
        f"[HyperSearch] Trial {result.trial_idx}",
        f"[HyperSearch] Attack: {args.attack}  Defense: {args.defense}",
        f"[HyperSearch] Params: {format_params(result.params)}",
        f"[HyperSearch] Success: {result.success}",
        f"[HyperSearch] Final trailing streak below threshold: {result.max_streak}",
        f"[HyperSearch] Final test accuracy: {result.final_test_acc:.4f}",
    ]
    if result.from_index:
        lines.append("[HyperSearch] Result source: hyper-parameter index cache")
    if result.matched_filter_scores:
        lines.append(
            f"[HyperSearch] COMPASS matched_filter_score mean={result.matched_filter_mean:.4f} "
            f"last={result.matched_filter_last:.4f}"
        )
    if result.success_window is not None:
        lines.append(
            f"[HyperSearch] Success window: Epoch {result.success_window[0]} - Epoch {result.success_window[1]}"
        )
    if result.error:
        lines.append(f"[HyperSearch] Trial error: {result.error}")
    return lines


def format_success_description(
    result: TrialResult,
    threshold: float,
    consecutive_rounds: int,
    total_rounds: int,
) -> str:
    expected_final_epoch = max(0, total_rounds - 1)
    if result.epochs and result.epochs[-1] < expected_final_epoch:
        return (
            "[HyperSearch] Attack success: training ended before "
            f"the planned final round {expected_final_epoch}, so the run is "
            "counted as a successful attack."
        )
    return (
        "[HyperSearch] Attack success: the final "
        f"{consecutive_rounds} rounds all stayed below test accuracy {threshold}."
    )


def make_search_paths(base_output: Path, timestamp: str) -> Tuple[Path, Path]:
    summary_path = base_output.with_name(f"{base_output.stem}__hypersearch_summary__{timestamp}.txt")
    final_plot_path = base_output.with_name(f"{base_output.stem}__hypersearch_success__{timestamp}.png")
    return summary_path, final_plot_path


def move_output_under_defense_dir(base_output: Path, defense: str) -> Path:
    defense_dir = base_output.parent / defense
    defense_dir.mkdir(parents=True, exist_ok=True)
    return defense_dir / base_output.name


def make_search_index_path(base_output: Path) -> Path:
    return base_output.with_name(f"{base_output.stem}__hypersearch_index__.jsonl")


def make_index_key(plugin: AttackSearchPlugin, params: Dict[str, object]) -> str:
    normalized = plugin._normalize_params(params)
    return format_params(normalized)


def serialize_index_entry(plugin: AttackSearchPlugin, params: Dict[str, object]) -> Dict[str, object]:
    return {
        "params": plugin._normalize_params(params),
    }


def load_search_index(index_path: Path, plugin: AttackSearchPlugin) -> set:
    indexed_keys: set = set()
    if not index_path.exists():
        return indexed_keys
    with index_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                params = payload.get("params")
                if params is None:
                    continue
                indexed_keys.add(make_index_key(plugin, params))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return indexed_keys


def append_search_index(index_path: Path, plugin: AttackSearchPlugin, params: Dict[str, object]) -> None:
    payload = serialize_index_entry(plugin, params)
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        handle.write("\n")


def make_trial_output(base_output: Path, trial_idx: int, params: Dict[str, object]) -> Path:
    param_hash = hashlib.md5(format_params(params).encode("utf-8")).hexdigest()[:8]
    return base_output.with_name(f"{base_output.stem}__hyper_t{trial_idx:03d}_{param_hash}{base_output.suffix}")


def run_experiment_once(
    base_args: SimpleNamespace,
    params: Dict[str, object],
    trial_idx: int,
    total_trials: int,
    threshold: float,
    consecutive_rounds: int,
    search_summary_path: Path,
) -> TrialResult:
    experiment_main = importlib.import_module("main")
    trial_args = copy.deepcopy(base_args)
    trial_args.attack_params = copy.deepcopy(params)
    trial_output = make_trial_output(Path(base_args.output), trial_idx, params)
    trial_args.output = str(trial_output)

    cleanup_logger("main")
    original_plot_accuracy = experiment_main.plot_accuracy
    experiment_main.plot_accuracy = lambda *_args, **_kwargs: None

    try:
        experiment_main.fl_run(trial_args)
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        append_lines(
            trial_output,
            [
                f"[HyperSearch] Trial {trial_idx}/{total_trials}",
                f"[HyperSearch] Params: {format_params(params)}",
                f"[HyperSearch] Trial crashed during training.",
                f"[HyperSearch] Error: {error_message}",
                traceback.format_exc().rstrip(),
            ],
        )
        result = TrialResult(
            trial_idx=trial_idx,
            params=copy.deepcopy(params),
            output_path=trial_output,
            success=False,
            max_streak=0,
            min_test_acc=float("inf"),
            final_test_acc=float("nan"),
            success_window=None,
            epochs=[],
            test_accs=[],
            sf_values=[],
            matched_filter_scores=[],
            error=error_message,
        )
        append_lines(trial_output, format_trial_summary(result, trial_args, total_trials))
        append_lines(search_summary_path, format_trial_summary(result, trial_args, total_trials))
        return result
    finally:
        experiment_main.plot_accuracy = original_plot_accuracy
        cleanup_logger("main")

    try:
        epochs, test_accs, sf_values, matched_scores = parse_trial_log(trial_output)
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        result = TrialResult(
            trial_idx=trial_idx,
            params=copy.deepcopy(params),
            output_path=trial_output,
            success=False,
            max_streak=0,
            min_test_acc=float("inf"),
            final_test_acc=float("nan"),
            success_window=None,
            epochs=[],
            test_accs=[],
            sf_values=[],
            matched_filter_scores=[],
            error=error_message,
        )
        append_lines(trial_output, format_trial_summary(result, trial_args, total_trials))
        append_lines(search_summary_path, format_trial_summary(result, trial_args, total_trials))
        return result

    success, max_streak, min_test_acc, success_window = evaluate_attack_success(
        epochs, test_accs, threshold, consecutive_rounds, int(trial_args.epochs)
    )
    result = TrialResult(
        trial_idx=trial_idx,
        params=copy.deepcopy(params),
        output_path=trial_output,
        success=success,
        max_streak=max_streak,
        min_test_acc=min_test_acc,
        final_test_acc=test_accs[-1],
        success_window=success_window,
        epochs=epochs,
        test_accs=test_accs,
        sf_values=sf_values,
        matched_filter_scores=matched_scores,
    )
    append_lines(trial_output, format_trial_summary(result, trial_args, total_trials))
    append_lines(search_summary_path, format_trial_summary(result, trial_args, total_trials))
    return result


def plot_success_result(
    result: TrialResult,
    attack: str,
    defense: str,
    threshold: float,
    consecutive_rounds: int,
    plot_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax_acc = plt.subplots(figsize=(10, 6))
    acc_line = ax_acc.plot(
        result.epochs,
        result.test_accs,
        label="Test Accuracy",
        linewidth=2.0,
        color="tab:blue",
    )[0]
    threshold_line = ax_acc.axhline(
        y=threshold,
        color="tab:red",
        linestyle=":",
        linewidth=1.5,
        label=f"Accuracy Threshold={threshold}",
    )
    ax_acc.set_xlabel("Global Round")
    ax_acc.set_ylabel("Test Accuracy", color="tab:blue")
    ax_acc.tick_params(axis="y", labelcolor="tab:blue")
    ax_acc.grid(True, linestyle="--", alpha=0.4)

    legend_handles = [acc_line, threshold_line]

    if result.matched_filter_scores:
        matched_epochs = result.epochs[-len(result.matched_filter_scores):]
        ax_mf = ax_acc.twinx()
        matched_line = ax_mf.plot(
            matched_epochs,
            result.matched_filter_scores,
            label="Matched Filter Score",
            linestyle="--",
            linewidth=2.0,
            color="tab:orange",
        )[0]
        ax_mf.set_ylabel("Matched Filter Score", color="tab:orange")
        ax_mf.tick_params(axis="y", labelcolor="tab:orange")
        legend_handles.append(matched_line)

    fig.suptitle(f"{attack} vs {defense}: Test Accuracy and Matched Filter Score")
    ax_acc.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="best")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close()


def print_trial_status(result: TrialResult, args: SimpleNamespace) -> None:
    status = "SUCCESS" if result.success else "FAILED"
    msg = (
        f"[HyperSearch] Trial {result.trial_idx}: {status} | "
        f"attack={args.attack} defense={args.defense} | "
        f"params={format_params(result.params)} | "
        f"final_trailing_streak={result.max_streak} | final_test_acc={result.final_test_acc:.4f}"
    )
    if result.from_index:
        msg += " | source=index_cache"
    if result.matched_filter_scores:
        msg += f" | matched_filter_mean={result.matched_filter_mean:.4f}"
    if result.error:
        msg += f" | error={result.error}"
    print(msg)


def select_plugin(attack_name: str) -> AttackSearchPlugin:
    plugin = PLUGIN_REGISTRY.get(attack_name)
    if plugin is None:
        raise ValueError(
            f"No hyper-parameter search space is defined for attack {attack_name!r}. "
            "Please add a plugin entry in main_hyper.py first."
        )
    return plugin


def run_hyper_search(args: SimpleNamespace, cli_args: argparse.Namespace) -> int:
    if getattr(cli_args, "benchmark", False):
        raise ValueError("main_hyper.py currently supports single-run hyper search only; benchmark mode is not supported.")

    args = merge_cli_args(args, cli_args)
    single_preprocess(args)
    args.output = str(move_output_under_defense_dir(Path(args.output), args.defense))

    effective_threshold = DATASET_SUCCESS_THRESHOLDS.get(
        getattr(args, "dataset", ""), cli_args.success_acc_threshold
    )

    plugin = select_plugin(args.attack)
    search_params = plugin.initialize_params(getattr(args, "attack_params", None))
    args.attack_params = copy.deepcopy(search_params)

    base_output = Path(args.output)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    search_summary_path, final_plot_path = make_search_paths(base_output, timestamp)
    search_index_path = make_search_index_path(base_output)
    indexed_keys = load_search_index(search_index_path, plugin)
    append_lines(
        search_summary_path,
        [
            "[HyperSearch] Search started.",
            f"[HyperSearch] Attack: {args.attack}",
            f"[HyperSearch] Defense: {args.defense}",
            f"[HyperSearch] Base output: {base_output}",
            f"[HyperSearch] Search index: {search_index_path}",
            f"[HyperSearch] Indexed params loaded: {len(indexed_keys)}",
            f"[HyperSearch] Initial params: {format_params(search_params)}",
            f"[HyperSearch] Trial limit: {HARD_SEARCH_TRIAL_LIMIT}",
            "[HyperSearch] Success criterion: "
            f"(1) training stops before the planned {args.epochs} rounds finish; or "
            f"(2) after all {args.epochs} rounds finish, the final {cli_args.success_consecutive_rounds} "
            f"rounds all have test accuracy < {effective_threshold}.",
            "",
        ],
    )

    tried_keys = set(indexed_keys)
    trial_results: List[TrialResult] = []
    current_params = copy.deepcopy(search_params)
    current_steps = plugin.initial_steps()
    best_result: Optional[TrialResult] = None
    best_params = copy.deepcopy(current_params)
    no_improvement_rounds = 0
    trial_idx = 0
    requested_max_trials = int(cli_args.max_search_trials)
    max_trials = HARD_SEARCH_TRIAL_LIMIT if requested_max_trials <= 0 else min(requested_max_trials, HARD_SEARCH_TRIAL_LIMIT)
    trial_budget_label = str(max_trials)
    failure_reason: Optional[str] = None

    def reached_trial_limit() -> bool:
        return trial_idx >= max_trials

    def remaining_budget() -> Optional[int]:
        return max_trials - trial_idx

    def record_trial(params_to_run: Dict[str, object]) -> TrialResult:
        nonlocal trial_idx, best_result, best_params
        trial_idx += 1
        normalized_params = plugin._normalize_params(params_to_run)
        cache_key = make_index_key(plugin, normalized_params)
        print(
            f"[HyperSearch] Starting trial {trial_idx}/{trial_budget_label} | "
            f"params={format_params(normalized_params)}"
        )
        result = run_experiment_once(
            base_args=args,
            params=normalized_params,
            trial_idx=trial_idx,
            total_trials=max_trials,
            threshold=effective_threshold,
            consecutive_rounds=cli_args.success_consecutive_rounds,
            search_summary_path=search_summary_path,
        )
        tried_keys.add(cache_key)
        trial_results.append(result)
        append_search_index(search_index_path, plugin, normalized_params)
        print_trial_status(result, args)
        if best_result is None or plugin.compare_results(result, best_result) > 0:
            best_result = result
            best_params = copy.deepcopy(normalized_params)
        return result

    while make_index_key(plugin, current_params) in tried_keys:
        append_lines(
            search_summary_path,
            [
                f"[HyperSearch] Initial params already exist in search index. Skipping {format_params(current_params)}",
                "",
            ],
        )
        restart_params = plugin.sample_untried_params(
            tried_keys=tried_keys,
            trial_results=trial_results,
            best_result=best_result,
            current_steps=current_steps,
        )
        if restart_params is None:
            append_lines(
                search_summary_path,
                ["[HyperSearch] Initial params are already indexed and no new untried params were found. Stop search.", ""],
            )
            print("[HyperSearch] No untried params available in search index scope.")
            return 1
        current_params = restart_params

    current_result = record_trial(current_params)
    if current_result.success:
        best_result = current_result

    while not reached_trial_limit() and not current_result.success:
        neighbors = plugin.generate_neighbors(current_params, current_steps, tried_keys)
        if not neighbors:
            if plugin.should_expand_search(current_result, best_result, current_steps):
                current_steps = plugin.expand_steps(current_steps)
                no_improvement_rounds = 0
                append_lines(
                    search_summary_path,
                    [
                        "[HyperSearch] Weak matched_filter signal suggests the malicious update is still not entering aggregation.",
                        f"[HyperSearch] Expanding steps to {format_params(current_steps)} for farther exploration.",
                        "",
                    ],
                )
                continue
            if plugin.can_reduce_steps(current_steps):
                current_steps = plugin.reduce_steps(current_steps, cli_args.search_step_decay)
                no_improvement_rounds = 0
                append_lines(
                    search_summary_path,
                    [f"[HyperSearch] No valid neighbors left. Shrinking steps to {format_params(current_steps)}", ""],
                )
                continue
            restart_params = plugin.sample_untried_params(
                tried_keys=tried_keys,
                trial_results=trial_results,
                best_result=best_result,
                current_steps=current_steps,
            )
            if restart_params is None:
                append_lines(
                    search_summary_path,
                    ["[HyperSearch] No valid local neighbors and no untried global restart point found. Stop search.", ""],
                )
                break
            current_params = restart_params
            current_steps = plugin.initial_steps()
            no_improvement_rounds = 0
            append_lines(
                search_summary_path,
                [
                    "[HyperSearch] Local search exhausted. Restarting from a new globally sampled point "
                    "guided by Pareto and dual-objective historical trials.",
                    f"[HyperSearch] Restart params: {format_params(current_params)}",
                    "",
                ],
            )
            continue

        evaluated_neighbors: List[TrialResult] = []
        budget = remaining_budget()
        candidates_to_try = neighbors if budget is None else neighbors[:budget]
        for candidate in candidates_to_try:
            result = record_trial(candidate)
            evaluated_neighbors.append(result)
            if result.success:
                current_result = result
                current_params = copy.deepcopy(candidate)
                break

        if current_result.success:
            break

        if not evaluated_neighbors:
            break

        best_neighbor = max(evaluated_neighbors, key=plugin.score_result)
        if plugin.compare_results(best_neighbor, current_result) > 0:
            current_result = best_neighbor
            current_params = copy.deepcopy(best_neighbor.params)
            no_improvement_rounds = 0
            append_lines(
                search_summary_path,
                [f"[HyperSearch] Moving search center to trial {best_neighbor.trial_idx}: {format_params(current_params)}", ""],
            )
        else:
            no_improvement_rounds += 1
            append_lines(
                search_summary_path,
                [f"[HyperSearch] No improvement around current center. patience={no_improvement_rounds}/{cli_args.search_patience}", ""],
            )
            if no_improvement_rounds >= cli_args.search_patience:
                if plugin.should_force_breakthrough(current_result, best_result, current_steps):
                    current_steps = plugin.expand_steps(current_steps)
                    current_params = copy.deepcopy(best_params)
                    current_result = best_result if best_result is not None else current_result
                    no_improvement_rounds = 0
                    append_lines(
                        search_summary_path,
                        [
                            "[HyperSearch] Patience exhausted without attack success in the local region.",
                            f"[HyperSearch] Escalating to wider breakthrough exploration around the current region with larger steps {format_params(current_steps)}.",
                            "",
                        ],
                    )
                    continue
                if plugin.should_expand_search(current_result, best_result, current_steps):
                    current_steps = plugin.expand_steps(current_steps)
                    current_params = copy.deepcopy(best_params)
                    current_result = best_result if best_result is not None else current_result
                    no_improvement_rounds = 0
                    append_lines(
                        search_summary_path,
                        [
                            "[HyperSearch] Patience exhausted under weak matched_filter signal.",
                            f"[HyperSearch] Expanding steps to {format_params(current_steps)} instead of shrinking.",
                            "",
                        ],
                    )
                    continue
                if plugin.can_reduce_steps(current_steps):
                    current_steps = plugin.reduce_steps(current_steps, cli_args.search_step_decay)
                    current_params = copy.deepcopy(best_params)
                    current_result = best_result if best_result is not None else current_result
                    no_improvement_rounds = 0
                    append_lines(
                        search_summary_path,
                        [f"[HyperSearch] Patience exhausted. Shrinking steps to {format_params(current_steps)}", ""],
                    )
                else:
                    restart_params = plugin.sample_untried_params(
                        tried_keys=tried_keys,
                        trial_results=trial_results,
                        best_result=best_result,
                        current_steps=current_steps,
                    )
                    if restart_params is None:
                        append_lines(
                            search_summary_path,
                            ["[HyperSearch] Patience exhausted, step sizes cannot be reduced further, and no untried restart point remains. Stop search.", ""],
                        )
                        break
                    current_params = restart_params
                    current_steps = plugin.initial_steps()
                    current_result = best_result if best_result is not None else current_result
                    no_improvement_rounds = 0
                    append_lines(
                        search_summary_path,
                        [
                            "[HyperSearch] Patience exhausted. Restarting global search from a promising historical region "
                            "(for COMPASS this balances lower test accuracy with higher matched_filter_score).",
                            f"[HyperSearch] Restart params: {format_params(current_params)}",
                            "",
                        ],
                    )

    assert best_result is not None
    if not best_result.success and reached_trial_limit():
        failure_reason = f"Trial limit reached ({max_trials}) before attack success."

    if best_result.success:
        plot_message = f"[HyperSearch] Success plot saved to: {final_plot_path}"
        try:
            plot_success_result(
                best_result,
                attack=args.attack,
                defense=args.defense,
                threshold=effective_threshold,
                consecutive_rounds=cli_args.success_consecutive_rounds,
                plot_path=final_plot_path,
            )
        except Exception as exc:
            plot_message = f"[HyperSearch] Success plot was not generated: {type(exc).__name__}: {exc}"
        append_lines(
            best_result.output_path,
            [
                f"[HyperSearch] SUCCESS PARAMS: {format_params(best_result.params)}",
                format_success_description(
                    best_result,
                    threshold=effective_threshold,
                    consecutive_rounds=cli_args.success_consecutive_rounds,
                    total_rounds=int(args.epochs),
                ),
                f"[HyperSearch] Success window: Epoch {best_result.success_window[0]} - Epoch {best_result.success_window[1]}",
                plot_message,
            ],
        )
        append_lines(
            search_summary_path,
            [
                f"[HyperSearch] Search finished with success after {len(trial_results)} trials.",
                f"[HyperSearch] SUCCESS PARAMS: {format_params(best_result.params)}",
                f"[HyperSearch] Best final trailing streak below threshold: {best_result.max_streak}",
                f"[HyperSearch] Final test accuracy: {best_result.final_test_acc:.4f}",
                f"[HyperSearch] Success log: {best_result.output_path}",
                plot_message,
            ],
        )
        print(f"[HyperSearch] Search succeeded after {len(trial_results)} trials.")
        print(f"[HyperSearch] SUCCESS PARAMS: {format_params(best_result.params)}")
        print(plot_message)
        return 0

    closest_result = best_result
    append_lines(
        search_summary_path,
            [
                f"[HyperSearch] Search failed after {len(trial_results)} trials.",
                f"[HyperSearch] Failure reason: {failure_reason or 'No successful hyper-parameter setting was found.'}",
                f"[HyperSearch] Best params found: {format_params(closest_result.params)}",
                f"[HyperSearch] Best final trailing streak below threshold: {closest_result.max_streak}",
                f"[HyperSearch] Final test accuracy: {closest_result.final_test_acc:.4f}",
                f"[HyperSearch] Last tried params: {format_params(trial_results[-1].params)}",
                f"[HyperSearch] Closest log: {closest_result.output_path}",
            ],
        )
    print(f"[HyperSearch] Search failed after {len(trial_results)} trials.")
    print(f"[HyperSearch] Failure reason: {failure_reason or 'No successful hyper-parameter setting was found.'}")
    print(f"[HyperSearch] Best params found: {format_params(closest_result.params)}")
    print(f"[HyperSearch] Best final trailing streak below threshold: {closest_result.max_streak}")
    print(f"[HyperSearch] Final test accuracy: {closest_result.final_test_acc:.4f}")
    return 1


if __name__ == "__main__":
    args, cli_args = read_hyper_args()
    raise SystemExit(run_hyper_search(args, cli_args))
