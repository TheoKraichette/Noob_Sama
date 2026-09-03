"""Record the phase 5 teacher as (features v2, movement action) pairs.

What differs from collect_demos.py: the teacher aims with the mouse, so a
pair holds the movement action (one of eos_env.MOVE_ACTIONS), the aim it
shot along, and the enemy it chose; the feature vector is features2.py's.
Each pair also records whether the inventory reflex (policies.
economy_action) was speaking at that state, the floor and whether a
guardian was on screen: the training script weighs the pairs with them.

`--policy` is the DAgger mode: the learned agent plays, the teacher labels.
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

from src import features2, policies
from src.agent2 import HybridAgent
from src.eos_env import MOVE_ACTIONS, EclipseEnv
from src.features import is_combat_state
from src.policies import economy_action, heuristic_decision


DEFAULT_OUTPUT = ROOT / "data" / "demos2.npz"
TEACHER_FLAGS = {"lead": "lead", "priority": "priority", "fire": "always_fire",
                 "mover": "predictive", "build": "build", "bosses": "bosses"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--max-steps", type=int, default=9000)
    parser.add_argument("--step-seconds", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--teacher",
        default="lead,priority,mover,build",
        help="Options of the teacher (see audit_deaths.py --policy teacher:...).",
    )
    parser.add_argument("--policy", type=Path, default=None, help="DAgger: this agent plays.")
    parser.add_argument("--floors", type=int, nargs="+", default=[1])
    parser.add_argument("--travel-keep", type=float, default=0.34)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def teacher_options(spec: str) -> dict[str, bool]:
    options = {flag: False for flag in TEACHER_FLAGS.values()}
    for token in filter(None, spec.split(",")):
        options[TEACHER_FLAGS[token]] = True
    return options


def collect_shard(job: dict) -> dict:
    args = job["args"]
    rng = random.Random(job["seed"])
    flags = teacher_options(args["teacher"])
    index = {name: i for i, name in enumerate(MOVE_ACTIONS)}
    player = HybridAgent.load(args["policy"], epsilon=0.02) if args["policy"] else None

    states, actions, aims, sources, floors, bosses = [], [], [], [], [], []
    scores: list[float] = []
    kept = dropped = 0

    with EclipseEnv(headless=True, turbo=True, step_seconds=args["step_seconds"]) as env:
        for episode, start_floor in job["episodes"]:
            obs = env.reset(start_floor=start_floor, seed=args["seed"] * 1000 + episode)
            previous = None
            done = False
            steps = 0
            while not done and steps < args["max_steps"]:
                label, aim = heuristic_decision(obs, **flags)
                if label not in index:
                    label = "noop"
                target = policies.last_target()
                if is_combat_state(obs) or rng.random() < args["travel_keep"]:
                    states.append(features2.featurize(obs, previous, aim, target))
                    actions.append(index[label])
                    aims.append(aim if aim is not None else (0.0, 0.0))
                    sources.append(1 if economy_action(obs) is not None else 0)
                    floors.append(obs.floor)
                    bosses.append(1 if any(e.get("boss") for e in obs.enemies) else 0)
                    kept += 1
                else:
                    dropped += 1
                if player is not None:
                    action, play_aim, _ = player.act(obs, previous)
                else:
                    action, play_aim = label, aim
                previous = obs
                obs, _reward, done, _info = env.step(action, play_aim)
                steps += 1
            scores.append(env.get_score())
            print(
                f"[w{job['worker']}] episode={episode} floor={start_floor}->{obs.floor} "
                f"rooms={obs.rooms} pairs={len(states)} score={env.get_score():.1f}",
                flush=True,
            )

    return {
        "states": np.asarray(states, dtype=np.float16),
        "actions": np.asarray(actions, dtype=np.int32),
        "aims": np.asarray(aims, dtype=np.float16),
        "sources": np.asarray(sources, dtype=np.int8),
        "floors": np.asarray(floors, dtype=np.int8),
        "bosses": np.asarray(bosses, dtype=np.int8),
        "scores": scores,
        "kept": kept,
        "dropped": dropped,
    }


def main() -> None:
    args = parse_args()
    plan = [(ep, args.floors[(ep - 1) % len(args.floors)]) for ep in range(1, args.episodes + 1)]
    shared = {
        "seed": args.seed, "max_steps": args.max_steps, "step_seconds": args.step_seconds,
        "travel_keep": args.travel_keep, "teacher": args.teacher,
        "policy": str(args.policy) if args.policy else None,
    }
    workers = max(1, args.workers)
    jobs = [{"worker": w, "episodes": plan[w::workers], "seed": args.seed + w, "args": shared}
            for w in range(workers)]
    jobs = [job for job in jobs if job["episodes"]]
    if len(jobs) == 1:
        shards = [collect_shard(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            shards = list(pool.map(collect_shard, jobs))

    merged = {k: np.concatenate([s[k] for s in shards]) for k in
              ("states", "actions", "aims", "sources", "floors", "bosses")}
    scores = [x for s in shards for x in s["scores"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, feature_version=features2.FEATURE_VERSION, **merged)
    kept = sum(s["kept"] for s in shards)
    dropped = sum(s["dropped"] for s in shards)
    counts = np.bincount(merged["actions"], minlength=len(MOVE_ACTIONS))
    print()
    print(f"saved {len(merged['states'])} pairs x {merged['states'].shape[1]} features to {args.output}")
    print(f"kept={kept} dropped_travel={dropped} economy_states={int(merged['sources'].sum())}")
    print("actions: " + " ".join(f"{n}:{c}" for n, c in zip(MOVE_ACTIONS, counts)))
    print(f"episodes={args.episodes} avg_score={statistics.mean(scores):.2f}")


if __name__ == "__main__":
    main()
