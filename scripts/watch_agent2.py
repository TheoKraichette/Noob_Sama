"""Reload the phase 5 agent from its JSON file and let it play.

The proof that the saved agent replays: nothing here trains, the network
is read back from disk by src/agent2.py and drives the browser through
the same environment the baseline used. With --turbo the game runs at
simulation speed; without it, a window shows the run in real time.

Evaluation honesty: every episode starts on floor 1 with an empty
inventory and the save wiped (the curriculum of eos_env.reset is a
training device and is refused here).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent2 import HybridAgent
from src.eos_env import EclipseEnv


DEFAULT_AGENT = ROOT / "models" / "bc2_agent.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=DEFAULT_AGENT)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=9000)
    parser.add_argument("--step-seconds", type=float, default=0.15)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--turbo", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=-1, help="Dungeon seed of episode 1 (-1: the game's clock).")
    parser.add_argument("--output", type=Path, default=None, help="CSV, one row per episode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = HybridAgent.load(args.agent, epsilon=args.epsilon)
    print(f"agent={args.agent} features={agent.net.n_features} ({agent.net.feature_version}) "
          f"actions={len(agent.net.actions)} heads={agent.net.heads}")
    rows = []
    with EclipseEnv(headless=args.headless, turbo=args.turbo, step_seconds=args.step_seconds) as env:
        for episode in range(1, args.episodes + 1):
            seed = args.seed + episode - 1 if args.seed >= 0 else None
            obs = env.reset(seed=seed)
            if obs.floor != 1 or obs.item_count or obs.gold:
                raise RuntimeError("evaluation must start on floor 1 with an empty inventory")
            previous = None
            done = False
            steps = 0
            started = time.time()
            while not done and steps < args.max_steps:
                action, aim, _info = agent.act(obs, previous)
                previous = obs
                obs, _reward, done, _ = env.step(action, aim)
                steps += 1
            row = {
                "episode": episode, "seed": seed if seed is not None else "", "steps": steps,
                "done": done, "won": obs.state == "victory", "score": round(env.get_score(), 3),
                "floor": obs.floor, "floors": obs.floors, "rooms": obs.rooms, "kills": obs.kills,
                "time": round(obs.time, 1),
            }
            rows.append(row)
            print(f"episode={episode} score={row['score']:.1f} floor={obs.floor} rooms={obs.rooms} "
                  f"kills={obs.kills} won={row['won']} steps={steps} ({time.time() - started:.0f}s)", flush=True)
    if args.output and rows:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"output={args.output}")
    scores = [r["score"] for r in rows]
    print(f"episodes={len(rows)} avg_score={sum(scores) / len(scores):.2f} "
          f"rooms={sum(r['rooms'] for r in rows)} floors={sum(r['floors'] for r in rows)} "
          f"victories={sum(r['won'] for r in rows)}")


if __name__ == "__main__":
    main()
