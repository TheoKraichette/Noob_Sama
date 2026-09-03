"""The phase 5 agent: a learned pilot and two explicit reflexes.

The network decides where to step (eight directions, walking or dashing),
when to use the hero's skill and when to press E: the twenty actions of
eos_env.MOVE_ACTIONS. Two reflexes it shares with the teacher handle what
a twin-stick player does with the other hand and what happens between
fights: the aim (aiming.py, the mouse) and the inventory (policies.
economy_action). Both are the same functions the teacher runs, so the
clone is measured on the pilot it learned, and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src import aiming, features2
from src.dqn import DQNAgent
from src.eos_env import MOVE_ACTIONS, Observation
from src.policies import economy_action, split_shot


class HybridAgent:
    def __init__(self, net: DQNAgent, *, lead: bool = True, priority: bool = True, epsilon: float = 0.0):
        self.net = net
        self.lead = lead
        self.priority = priority
        self.epsilon = epsilon
        self._target: dict | None = None
        self.rng = np.random.default_rng(0)

    @classmethod
    def load(cls, path: Path | str, **kw) -> "HybridAgent":
        net = DQNAgent.load(path, expect=(features2.N_FEATURES, features2.FEATURE_VERSION))
        return cls(net, **kw)

    def act(self, obs: Observation, previous: Observation | None) -> tuple[str, tuple[float, float] | None, dict]:
        """(action, aim, info): info says which reflex spoke, if any."""
        if obs.state != "play":
            return "noop", None, {"source": "none"}
        target = aiming.pick_target(obs, self._target) if self.priority else obs.nearest_enemy
        self._target = target
        aim = aiming.aim_vector(obs, target, lead=self.lead) if target else None
        eco = economy_action(obs)
        if eco is not None:
            move, _ = split_shot(eco)
            return move, aim, {"source": "economy", "target": target}
        x = np.asarray(features2.featurize(obs, previous, aim, target), dtype=np.float32)
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            action = MOVE_ACTIONS[int(self.rng.integers(len(MOVE_ACTIONS)))]
        else:
            action = self.net.act(x)
        return action, aim, {"source": "network", "target": target}
