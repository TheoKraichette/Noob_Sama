"""Iterated self-imitation: learn from the agent's own best runs.

Cloning converges on the teacher and stops there — measured in V19, where the
clone matched its teacher exactly. This goes past it without any hand-written
rule: the agent plays, its best episodes are kept, and it retrains on what it
did in those episodes. What varies between a good run and a bad one is the
agent's own choices, so reinforcing the good ones moves the policy up.

A slice of the original demonstrations is mixed back in at every round
(`--anchor`), otherwise the policy drifts away from the behaviours it never
samples any more and collapses.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dqn import DQNAgent
from src.eos_env import ACTIONS, EclipseEnv
from src.features import featurize, is_combat_state


DEFAULT_AGENT = ROOT / "models" / "bc_agent.json"
DEFAULT_OUTPUT = ROOT / "models" / "si_agent.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=DEFAULT_AGENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=9000)
    parser.add_argument("--step-seconds", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Exploration during the rollouts. Some noise is what makes the "
        "episodes differ from one another, which is what there is to rank.",
    )
    parser.add_argument(
        "--keep-frac",
        type=float,
        default=0.25,
        help="Fraction of each round's episodes kept for retraining.",
    )
    parser.add_argument(
        "--anchor",
        type=Path,
        nargs="*",
        default=None,
        help="Demonstration files mixed back in to prevent drift.",
    )
    parser.add_argument(
        "--anchor-frac",
        type=float,
        default=1.0,
        help="Anchor pairs per pair of fresh elite data. At 0.4 the policy "
        "over-fitted its own 30 000-pair elite set and lost competence.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--travel-keep", type=float, default=0.34)
    parser.add_argument(
        "--log",
        type=Path,
        default=ROOT / "reports" / "training_selfimit.csv",
    )
    return parser.parse_args()


def play_shard(job: dict) -> list[dict]:
    """Play a slice of the round's episodes and return one record each."""
    cfg = job["cfg"]
    agent = DQNAgent.load(cfg["agent"], seed=job["seed"])
    rng = np.random.default_rng(job["seed"])
    action_index = {name: i for i, name in enumerate(ACTIONS)}
    records: list[dict] = []

    with EclipseEnv(
        headless=True, turbo=True, step_seconds=cfg["step_seconds"]
    ) as env:
        for episode in job["episodes"]:
            obs = env.reset()
            previous = None
            done = False
            steps = 0
            states: list[list[float]] = []
            actions: list[int] = []

            while not done and steps < cfg["max_steps"]:
                features = featurize(obs, previous)
                action = agent.act(features, epsilon=cfg["epsilon"])
                if action not in ACTIONS:
                    action = "noop"
                if is_combat_state(obs) or rng.random() < cfg["travel_keep"]:
                    states.append(features)
                    actions.append(action_index[action])
                previous = obs
                obs, _reward, done, _info = env.step(action)
                steps += 1

            records.append(
                {
                    "episode": episode,
                    "states": np.asarray(states, dtype=np.float16),
                    "actions": np.asarray(actions, dtype=np.int32),
                    "score": env.get_score(),
                    "floors": obs.floors,
                    "floor": obs.floor,
                    "rooms": obs.rooms,
                    "won": obs.state == "victory",
                    "steps": steps,
                }
            )
            print(
                f"[w{job['worker']}] episode={episode} floor={obs.floor} "
                f"rooms={obs.rooms} score={env.get_score():.1f} "
                f"won={obs.state == 'victory'}",
                flush=True,
            )
    return records


def rollout(cfg: dict, episodes: list[int], workers: int, seed: int) -> list[dict]:
    jobs = [
        {"worker": w, "episodes": episodes[w::workers], "seed": seed + w, "cfg": cfg}
        for w in range(workers)
    ]
    jobs = [job for job in jobs if job["episodes"]]
    if len(jobs) == 1:
        return play_shard(jobs[0])
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        return [record for shard in pool.map(play_shard, jobs) for record in shard]


def load_anchor(paths: list[Path] | None) -> tuple[np.ndarray, np.ndarray] | None:
    if not paths:
        return None
    states = np.concatenate([np.load(p)["states"] for p in paths])
    actions = np.concatenate([np.load(p)["actions"] for p in paths])
    print(f"anchor: {len(actions)} demonstration pairs", flush=True)
    return states, actions


def main() -> None:
    args = parse_args()
    agent = DQNAgent.load(args.agent)
    print(
        f"start from {args.agent.name} "
        f"features={agent.n_features} actions={len(agent.actions)}",
        flush=True,
    )
    anchor = load_anchor(args.anchor)
    rng = np.random.default_rng(args.seed)
    workers = max(1, args.workers)

    # The champion is only replaced by a round that actually plays better.
    working = ROOT / "checkpoints" / "selfimit_working.json"
    agent.save(working)
    best_score = None
    best_params = [p.copy() for p in agent.params]
    rows: list[dict[str, object]] = []

    for round_no in range(1, args.rounds + 1):
        cfg = {
            "agent": str(working),
            "max_steps": args.max_steps,
            "step_seconds": args.step_seconds,
            "epsilon": args.epsilon,
            "travel_keep": args.travel_keep,
        }
        records = rollout(
            cfg, list(range(1, args.episodes + 1)), workers, args.seed + round_no * 100
        )
        scores = [r["score"] for r in records]
        mean_score = statistics.mean(scores)
        wins = sum(int(r["won"]) for r in records)
        # Selection signal, for the same reason as the ranking above: rooms
        # and floors cannot be farmed by idling, the score can.
        progress = statistics.mean(r["floors"] * 10 + r["rooms"] for r in records)
        print()
        print(
            f"round {round_no}: avg={mean_score:.2f} max={max(scores):.2f} "
            f"progress={progress:.2f} wins={wins} "
            f"floors={sorted({r['floor'] for r in records})}",
            flush=True,
        )

        improved = best_score is None or progress > best_score
        if improved:
            best_score = progress
            best_params = [p.copy() for p in agent.params]
            agent.save(
                args.output,
                extra={
                    "algo": "self_imitation",
                    "round": round_no,
                    "rollout_score": round(mean_score, 3),
                    "rollout_progress": round(progress, 3),
                },
            )
            print(f"  new best -> {args.output}", flush=True)
        else:
            # Hill-climbing with rejection. Without this the loop walks
            # wherever the last round pushed it, and one bad round compounds
            # into the next: measured at progress 12.3 -> 8.9 -> 8.2 over
            # three rounds. A round that does not play better is undone, and
            # the next attempt starts again from the champion.
            agent.params = [p.copy() for p in best_params]
            agent._adam_m = [np.zeros_like(p) for p in agent.params]
            agent._adam_v = [np.zeros_like(p) for p in agent.params]
            agent._adam_t = 0
            print(f"  round rejected, back to the champion ({best_score:.2f})", flush=True)

        # Rank by progress first, and only then by score. The comparison
        # score pays 0.1 per second survived, so an episode that freezes for
        # its whole budget scores higher than a decent run that died — rank
        # on score alone and the policy learns to stand still.
        records.sort(
            key=lambda r: (r["floors"], r["won"], r["rooms"], r["score"]),
            reverse=True,
        )
        keep = max(1, int(len(records) * args.keep_frac))
        elite = records[:keep]
        cut = elite[-1]["score"]
        states = np.concatenate([r["states"] for r in elite])
        actions = np.concatenate([r["actions"] for r in elite])
        print(
            f"  keeping {keep}/{len(records)} episodes "
            f"(score cut {cut:.1f}), {len(actions)} pairs",
            flush=True,
        )

        if anchor is not None and args.anchor_frac > 0:
            n_anchor = int(len(actions) * args.anchor_frac)
            idx = rng.choice(len(anchor[1]), size=min(n_anchor, len(anchor[1])), replace=False)
            states = np.concatenate([states, anchor[0][idx]])
            actions = np.concatenate([actions, anchor[1][idx]])
            print(f"  + {len(idx)} anchor pairs -> {len(actions)} total", flush=True)

        agent.lr = args.lr
        order = np.arange(len(actions))
        for epoch in range(1, args.epochs + 1):
            rng.shuffle(order)
            losses = []
            for start in range(0, len(order), args.batch_size):
                batch = order[start : start + args.batch_size]
                losses.append(
                    agent.train_step_bc(
                        states[batch].astype(np.float32), actions[batch]
                    )
                )
            print(f"  epoch={epoch} loss={np.mean(losses):.4f}", flush=True)

        agent.episodes_trained += len(records)
        agent.save(working)
        rows.append(
            {
                "round": round_no,
                "episodes": len(records),
                "avg_score": round(mean_score, 3),
                "max_score": round(max(scores), 3),
                "progress": round(progress, 3),
                "rooms": sum(r["rooms"] for r in records),
                "wins": wins,
                "kept": keep,
                "score_cut": round(cut, 3),
                "pairs": int(len(actions)),
            }
        )

    if rows:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nlog={args.log}")
    print(f"best rollout progress {best_score:.2f} saved in {args.output}")


if __name__ == "__main__":
    main()
