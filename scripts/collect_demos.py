"""Record demonstrations as (features, action) pairs.

Three things changed for the full-run budget:

- **Curriculum** (`--floors`): training episodes may start on a deeper floor,
  so the dataset covers the guardians a normal run reaches once in a while.
  Evaluation never does this — see the assertion in scripts/watch_dqn.py.
- **Volume**: a 9 000-step episode produces eight times more pairs than the
  old 900-step one. Pairs are stored as float16 and travel states are
  subsampled, which also rebalances the set toward the combat that matters.
- **Workers** (`--workers`): several browsers in parallel, because wall-clock
  is now the limiting factor.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents import CONCRETE_ACTIONS, project_action
from src.dqn import DQNAgent
from src.eos_env import ACTIONS, EclipseEnv
from src.features import featurize
from src.policies import heuristic_action


DEFAULT_OUTPUT = ROOT / "data" / "demos.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--max-steps", type=int, default=9000)
    parser.add_argument("--step-seconds", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="DAgger mode: this saved agent PLAYS the episodes while the "
        "heuristic only provides the recorded labels, so the dataset "
        "covers the states the clone actually reaches.",
    )
    parser.add_argument(
        "--floors",
        type=int,
        nargs="+",
        default=[1],
        help="Starting floors cycled over the episodes (training only). "
        "For example '1 1 1 1 1 1 2 3 4 5' spends 60%% of the episodes on "
        "full runs and the rest covering deeper guardians.",
    )
    parser.add_argument(
        "--travel-keep",
        type=float,
        default=0.34,
        help="Fraction of the quiet travel states to keep. Combat states "
        "are always kept.",
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def _is_combat(obs) -> bool:
    """A state worth keeping in full: something is threatening the player."""
    return bool(obs.enemy_count or obs.shots or obs.hazards or obs.spawning)


def collect_shard(job: dict) -> dict:
    """Play a slice of the episodes in one browser and return its pairs."""
    args = job["args"]
    rng = random.Random(job["seed"])
    action_index = {name: i for i, name in enumerate(ACTIONS)}
    policy = DQNAgent.load(args["policy"]) if args["policy"] else None

    states: list[list[float]] = []
    actions: list[int] = []
    scores: list[float] = []
    floors_total = 0
    kept = dropped = 0

    with EclipseEnv(
        headless=True, turbo=True, step_seconds=args["step_seconds"]
    ) as env:
        for episode, start_floor in job["episodes"]:
            # One dungeon per (collection seed, episode): a DAgger round run
            # with another --seed covers other dungeons than the base set.
            obs = env.reset(
                start_floor=start_floor, seed=args["seed"] * 1000 + episode
            )
            previous = None
            done = False
            steps = 0

            while not done and steps < args["max_steps"]:
                label = project_action(heuristic_action(obs), CONCRETE_ACTIONS, rng)
                # Quiet corridors are highly redundant; the fights are not.
                if _is_combat(obs) or rng.random() < args["travel_keep"]:
                    states.append(featurize(obs, previous))
                    actions.append(action_index[label])
                    kept += 1
                else:
                    dropped += 1

                if policy is not None:
                    action = policy.act(featurize(obs, previous), epsilon=0.02)
                    if action not in ACTIONS:
                        action = label
                else:
                    action = label
                previous = obs
                obs, _reward, done, _info = env.step(action)
                steps += 1

            scores.append(env.get_score())
            floors_total += obs.floors
            print(
                f"[w{job['worker']}] episode={episode} floor={start_floor}->"
                f"{obs.floor} pairs={len(states)} score={env.get_score():.1f}",
                flush=True,
            )

    return {
        "states": np.asarray(states, dtype=np.float16),
        "actions": np.asarray(actions, dtype=np.int32),
        "scores": scores,
        "floors_total": floors_total,
        "kept": kept,
        "dropped": dropped,
    }


def main() -> None:
    args = parse_args()
    plan = [
        (episode, args.floors[(episode - 1) % len(args.floors)])
        for episode in range(1, args.episodes + 1)
    ]
    shared = {
        "seed": args.seed,
        "max_steps": args.max_steps,
        "step_seconds": args.step_seconds,
        "travel_keep": args.travel_keep,
        "policy": str(args.policy) if args.policy else None,
    }
    workers = max(1, args.workers)
    jobs = [
        {
            "worker": w,
            "episodes": plan[w::workers],
            "seed": args.seed + w,
            "args": shared,
        }
        for w in range(workers)
    ]
    jobs = [job for job in jobs if job["episodes"]]

    if len(jobs) == 1:
        shards = [collect_shard(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            shards = list(pool.map(collect_shard, jobs))

    states = np.concatenate([shard["states"] for shard in shards])
    actions = np.concatenate([shard["actions"] for shard in shards])
    scores = [score for shard in shards for score in shard["scores"]]
    floors_total = sum(shard["floors_total"] for shard in shards)
    kept = sum(shard["kept"] for shard in shards)
    dropped = sum(shard["dropped"] for shard in shards)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, states=states, actions=actions)
    print()
    print(f"saved {len(states)} pairs to {args.output} ({states.dtype})")
    print(f"kept={kept} dropped_travel={dropped}")
    print(
        f"episodes={args.episodes} avg_score={statistics.mean(scores):.2f} "
        f"floors={floors_total}"
    )


if __name__ == "__main__":
    main()
