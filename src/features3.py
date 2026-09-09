"""Feature vector v3: v2 plus the simulation's verdict on every move.

The teacher chooses its step by simulating the next 0.6 s for each of the
seventeen moves the keyboard allows (mover.py). A clone fed only positions
and velocities has to rebuild that verdict from scratch, and 24 % of its
steps differ from the teacher's - the ones that cost hearts. Phase 3 met
the same wall and answered it by giving the network the danger field
rather than asking it to guess one. This vector does the same with the
simulation: for each move, whether it gets hit, how soon, and how much
room it keeps. The network still decides; the simulation is an input,
computed from the observation and the game's rules, the same one the
teacher reads.
"""

from __future__ import annotations

import math

from src import features2, mover
from src.eos_env import Observation


FEATURE_VERSION = "v3"

# fixed slot order: standing still, then each direction walking and dashing
MOVES = ["noop"]
for _name in mover.DIRECTIONS:
    MOVES.append(_name)
    MOVES.append("dash_" + _name)
N_PER_MOVE = 4


def verdicts(obs: Observation) -> list[float]:
    """For each of the 17 moves: available, safe, time before the hit, clearance."""
    out: list[float] = []
    by_action = {c.action: c for c in mover.candidates(obs, allow_dash=True)}
    for name in MOVES:
        c = by_action.get(name)
        if c is None:
            out.extend([0.0, 0.0, 0.0, 0.0])  # a dash that is not ready
            continue
        hit = 1.0 if c.hit_t is None else min(c.hit_t, mover.HORIZON) / mover.HORIZON
        clearance = 1.0 if math.isinf(c.clearance) else max(-1.0, min(1.0, c.clearance / 100.0))
        out.extend([1.0, 1.0 if c.safe else 0.0, hit, clearance])
    return out


def featurize(
    obs: Observation,
    previous: Observation | None = None,
    aim: tuple[float, float] | None = None,
    target: dict | None = None,
) -> list[float]:
    if obs.state != "play":
        return [0.0] * N_FEATURES
    return features2.featurize(obs, previous, aim, target) + verdicts(obs)


N_FEATURES = features2.N_FEATURES + len(MOVES) * N_PER_MOVE
