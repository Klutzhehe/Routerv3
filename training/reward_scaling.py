"""Running reward scaling -- why the value loss was exploding.

`RewardCalculator` mixes a +1000 connect bonus with O(1-100)/step progress
terms (dist_scale=15 times a step of up to 8 cells), so episode returns swing
from roughly -750 on a failed net to several thousand on a routed one. The
value head has to fit that whole range with plain MSE, and until it does,
`0.5 * (pred - return)**2` sits in the thousands -- which, backpropagated
through the SAME trunk the policy head reads from and clipped to one global
grad norm of 0.5, spends nearly the entire gradient budget on the critic and
starves the actor. That is the flat `pi=0.000` / thrashing `V` loss pattern
measured in the failed run: not "not learning", but "learning only the
critic, and not even converging there."

This is the standard fix (OpenAI Baselines' `VecNormalize`, SB3's
`--norm_reward`): scale the reward fed to the *learner* by a running estimate
of the discounted return's std, so the value target stays near unit variance
regardless of how the raw reward table is weighted. Rewards used for logging
(`curr_ep_reward` in train.py) stay in raw units -- only what the buffer
stores and the value loss is computed against changes.
"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Welford/Chan's parallel algorithm, so batches of size 1 are exact."""

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: float) -> None:
        self._update_from_moments(x, 0.0, 1.0)

    def _update_from_moments(self, batch_mean: float, batch_var: float, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count


class RewardScaler:
    """Divides by the std of the running DISCOUNTED RETURN, not the raw
    reward -- the value function predicts returns, so that is the quantity
    whose scale actually determines the loss magnitude. `clip` bounds the
    rare +1000-class spike (net_done, or an early via/collision run) from
    dominating a single minibatch before the running estimate has adapted to
    it -- the same reason OpenAI Baselines clips normalized reward to +-10.
    """

    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8, clip: float = 10.0) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        self.ret_rms = RunningMeanStd()
        self._running_return = 0.0

    def scale(self, reward: float, done: bool) -> float:
        self._running_return = self._running_return * self.gamma + reward
        self.ret_rms.update(self._running_return)
        scaled = reward / (np.sqrt(self.ret_rms.var) + self.epsilon)
        scaled = float(np.clip(scaled, -self.clip, self.clip))
        if done:
            self._running_return = 0.0
        return scaled
