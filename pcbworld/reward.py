"""Reward Formulation and PCB Quality Evaluation.

Calculates step-wise and episodic rewards for RL agent actions:
- Goal arrival bonus (+1000)
- Distance reduction progress (+10 * delta_dist)
- Collision / DRC violation penalty (-1000)
- Wirelength efficiency penalty (-0.1 * step_length)
- Directional bend penalty (-1.0 for bends > 45 deg)
- Via placement penalty (-5.0 per via)
- Congestion / corridor blocking penalty (-50.0)
"""

from __future__ import annotations

import math
from typing import Tuple, Dict, Any


class RewardCalculator:
    def __init__(
        self,
        connect_bonus: float = 1000.0,
        collision_penalty: float = 50.0,
        dist_scale: float = 15.0,
        align_scale: float = 5.0,
        length_penalty_scale: float = 0.02,
        bend_penalty: float = 0.5,
        via_penalty: float = 5.0,
        congestion_penalty_scale: float = 2.0,
    ):
        self.connect_bonus = connect_bonus
        self.collision_penalty = collision_penalty
        self.dist_scale = dist_scale
        self.align_scale = align_scale
        self.length_penalty_scale = length_penalty_scale
        self.bend_penalty = bend_penalty
        self.via_penalty = via_penalty
        self.congestion_penalty_scale = congestion_penalty_scale

    def compute_step_reward(
        self,
        prev_dist: float,
        curr_dist: float,
        step_len: float,
        heading_alignment: float,
        is_connected: bool,
        is_collided: bool,
        is_bend: bool,
        is_via: bool,
        congestion_overlap: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute the step reward and breakdown dict."""
        reward_breakdown = {}

        if is_connected:
            reward = self.connect_bonus
            reward_breakdown["connect"] = self.connect_bonus
            return reward, reward_breakdown

        if is_collided:
            reward = -self.collision_penalty
            reward_breakdown["collision"] = -self.collision_penalty
            return reward, reward_breakdown

        # 1. Distance progress toward target
        delta_dist = prev_dist - curr_dist
        dist_reward = delta_dist * self.dist_scale
        reward_breakdown["dist"] = dist_reward

        # 2. Heading alignment toward target pad
        align_reward = self.align_scale * heading_alignment
        reward_breakdown["align"] = align_reward

        # 3. Step length cost
        length_cost = -self.length_penalty_scale * step_len
        reward_breakdown["length"] = length_cost

        # 4. Bend penalty
        bend_cost = -self.bend_penalty if is_bend else 0.0
        reward_breakdown["bend"] = bend_cost

        # 5. Via penalty
        via_cost = -self.via_penalty if is_via else 0.0
        reward_breakdown["via"] = via_cost

        # 6. Congestion penalty
        cong_cost = -self.congestion_penalty_scale * congestion_overlap
        reward_breakdown["congestion"] = cong_cost

        total_reward = dist_reward + align_reward + length_cost + bend_cost + via_cost + cong_cost
        return total_reward, reward_breakdown
