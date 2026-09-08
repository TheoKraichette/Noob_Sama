"""Watch the phase 5 teacher play in a visible browser, in real time.

The same rules the demonstrations are recorded from (policies.
heuristic_decision with its options), driving the game through the
keyboard and the mouse at the game's own pace. --turbo runs the same
policy at simulation speed instead, still drawing the frames.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eos_env import EclipseEnv
from src.policies import heuristic_decision


FLAGS = {"lead": "lead", "priority": "priority", "fire": "always_fire",
         "mover": "predictive", "build": "build", "bosses": "bosses"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=9000)
    parser.add_argument("--step-seconds", type=float, default=0.15)
    parser.add_argument("--teacher", default="lead,priority,mover,bosses,build",
                        help="Options of the teacher, comma separated (see audit_deaths.py).")
    parser.add_argument("--seed", type=int, default=-1, help="Dungeon seed of episode 1 (-1: the game's clock).")
    parser.add_argument("--turbo", action="store_true", help="Simulation speed instead of real time.")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = {flag: False for flag in FLAGS.values()}
    for token in filter(None, args.teacher.split(",")):
        options[FLAGS[token]] = True
    with EclipseEnv(headless=args.headless, turbo=args.turbo, step_seconds=args.step_seconds) as env:
        for episode in range(1, args.episodes + 1):
            seed = args.seed + episode - 1 if args.seed >= 0 else None
            obs = env.reset(seed=seed)
            started = time.time()
            done = False
            steps = 0
            last_room = None
            while not done and steps < args.max_steps:
                action, aim = heuristic_decision(obs, **options)
                obs, _reward, done, _ = env.step(action, aim)
                steps += 1
                key = (obs.floor, obs.rooms)
                if key != last_room:
                    last_room = key
                    print(f"  floor {obs.floor} rooms {obs.rooms} hp {obs.hp}/{obs.max_hp} "
                          f"gold {obs.gold} items {obs.item_count} kills {obs.kills} step {steps}", flush=True)
            print(f"episode={episode} seed={seed} {obs.state} score={env.get_score():.1f} "
                  f"floor={obs.floor} rooms={obs.rooms} kills={obs.kills} steps={steps} "
                  f"({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
