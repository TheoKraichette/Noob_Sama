"""Where does the run actually lose its hearts?

One row per point of life lost, with the culprit the game itself names
(`killerName`), the floor, the room and the boss attack in progress. Every
change made to the policy has to justify itself by a variation in this table,
not by an impression watching the browser.

The floor-entry table it also prints (items, gold, life on arrival) is what
calibrates the training curriculum in `src/eos_env.py`.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent2 import HybridAgent
from src.dqn import DQNAgent
from src.eos_env import ACTIONS, EclipseEnv
from src.features import featurize
from src.policies import heuristic_action, heuristic_decision


DEFAULT_OUTPUT = ROOT / "reports" / "death_audit.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        default="heuristic",
        help="'heuristic' (keyboard aim, eight rays), 'teacher' (the same "
        "rules with the mouse aim; options after a colon, e.g. "
        "'teacher:lead,priority,fire'), 'hybrid:<model.json>' (the phase 5 "
        "agent: learned pilot, shared aim and inventory reflexes) or the "
        "path to a saved agent (JSON).",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=9000)
    parser.add_argument("--step-seconds", type=float, default=0.15)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--start-floor", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Episode i plays the dungeon seeded with seed + i. Two audits "
        "run with the same seed face the same forty dungeons, so their "
        "difference is the policies' and not the layouts'. Negative "
        "disables seeding (the game's own clock seed).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Browsers in parallel. Twenty episodes leave a standard error "
        "of about 25 points on the mean score, which hides any change "
        "smaller than itself; running more of them is the only honest way "
        "to see one.",
    )
    return parser.parse_args()


def teacher_options(spec: str) -> dict[str, bool]:
    """'teacher:lead,priority' -> the keyword flags of heuristic_decision."""
    names = {
        "lead": "lead", "priority": "priority", "fire": "always_fire",
        "mover": "predictive", "build": "build", "bosses": "bosses",
    }
    options = {flag: False for flag in names.values()}
    _, _, tail = spec.partition(":")
    for token in filter(None, tail.split(",")):
        if token not in names:
            raise SystemExit(f"unknown teacher option {token!r} (choose from {sorted(names)})")
        options[names[token]] = True
    return options


def _boss_attack(obs) -> tuple[str | None, str | None]:
    """Name of the boss on screen and the attack it is running, if any."""
    for enemy in obs.enemies:
        if enemy.get("boss"):
            return enemy.get("type"), enemy.get("attack")
    return None, None


def audit_shard(job: dict) -> dict:
    """Play a slice of the episodes in one browser and return its tables."""
    cfg = job["cfg"]
    agent = None
    teacher = None
    hybrid = None
    if cfg["policy"] == "heuristic":
        pass
    elif cfg["policy"].startswith("teacher"):
        teacher = teacher_options(cfg["policy"])
    elif cfg["policy"].startswith("hybrid:"):
        hybrid = HybridAgent.load(cfg["policy"][7:], epsilon=cfg["epsilon"])
    else:
        agent = DQNAgent.load(Path(cfg["policy"]))

    hits: list[dict[str, object]] = []
    arrivals: list[dict[str, object]] = []
    episodes: list[dict[str, object]] = []

    with EclipseEnv(
        headless=True, turbo=True, step_seconds=cfg["step_seconds"]
    ) as env:
        for episode in job["episodes"]:
            seed = cfg["seed"] + episode if cfg["seed"] >= 0 else None
            obs = env.reset(start_floor=cfg["start_floor"], seed=seed)
            previous = None
            done = False
            steps = 0
            floor_seen = obs.floor
            arrivals.append(
                {
                    "episode": episode,
                    "floor": obs.floor,
                    "items": obs.item_count,
                    "gold": obs.gold,
                    "hp": obs.hp,
                    "max_hp": obs.max_hp,
                }
            )

            last_progress = 0
            worst_stall = 0
            stall_room = obs.room_type

            while not done and steps < cfg["max_steps"]:
                aim = None
                if teacher is not None:
                    action, aim = heuristic_decision(obs, **teacher)
                elif hybrid is not None:
                    action, aim, _ = hybrid.act(obs, previous)
                elif agent is None:
                    action = heuristic_action(obs)
                else:
                    action = agent.act(featurize(obs, previous), epsilon=cfg["epsilon"])
                if action not in ACTIONS:
                    action = "noop"
                before = obs
                previous = obs
                obs, _reward, done, _info = env.step(action, aim)
                steps += 1

                # A run that stops clearing rooms is burning its budget just
                # as surely as one that dies, and it is invisible in the
                # death table. Track the longest stretch without progress.
                if obs.rooms != before.rooms or obs.floor != before.floor:
                    last_progress = steps
                elif steps - last_progress > worst_stall:
                    worst_stall = steps - last_progress
                    stall_room = before.room_type
                    # Snapshot the first deep stall: is the PLAYER wedged, or
                    # is an enemy simply unreachable? The answers differ.
                    if worst_stall == 2000:
                        enemy = obs.nearest_enemy
                        stall_detail = {
                            "episode": episode,
                            "room_type": obs.room_type,
                            "enemies": obs.enemy_count,
                            "doors_open": obs.doors_open,
                            "stuck_steps": obs.stuck_steps,
                            "hp": f"{obs.hp}/{obs.max_hp}",
                            "pos": f"{obs.x:.0f},{obs.y:.0f}",
                            "pickups": obs.pickup_count,
                            "enemy": enemy.get("type") if enemy else None,
                            "enemy_hp": f"{enemy['hp']:.0f}/{enemy['max_hp']:.0f}"
                            if enemy
                            else None,
                            "enemy_dist": round(enemy["distance"]) if enemy else None,
                        }
                        print(f"  STALL {stall_detail}", flush=True)

                # The fatal blow counts too: the game names its author before
                # switching to the game-over screen, and a table that stops
                # one hit short of every death undercounts the runs' end.
                # A drop of max life (the sacrifice altar) clamps the bar
                # without anyone striking: not a hit, and the culprit the
                # game still names would be the previous one.
                if obs.hp < before.hp and obs.max_hp >= before.max_hp:
                    boss, attack = _boss_attack(before)
                    hp_left = max(0, obs.hp)
                    hits.append(
                        {
                            "episode": episode,
                            "step": steps,
                            "floor": before.floor,
                            "room_type": before.room_type,
                            "lost": before.hp - hp_left,
                            "hp_left": hp_left,
                            "fatal": obs.state == "gameover",
                            "killer": obs.killer,
                            "killer_was_boss": obs.killer_was_boss,
                            "boss": boss,
                            "boss_attack": attack,
                            "enemies": before.enemy_count,
                            "hazards": len(before.hazards),
                            "shots": len(before.shots),
                        }
                    )

                if obs.floor != floor_seen and obs.state == "play":
                    floor_seen = obs.floor
                    arrivals.append(
                        {
                            "episode": episode,
                            "floor": obs.floor,
                            "items": obs.item_count,
                            "gold": obs.gold,
                            "hp": obs.hp,
                            "max_hp": obs.max_hp,
                        }
                    )

            outcome = (
                "victory"
                if obs.state == "victory"
                else "death"
                if obs.state == "gameover"
                else "budget"
            )
            episodes.append(
                {
                    "episode": episode,
                    "seed": seed if seed is not None else "",
                    "steps": steps,
                    "outcome": outcome,
                    "score": round(env.get_score(), 3),
                    "floor": obs.floor,
                    "floors": obs.floors,
                    "rooms": obs.rooms,
                    "kills": obs.kills,
                    "items": obs.item_count,
                    "time": round(obs.time, 1),
                    "stall": worst_stall,
                    "stall_room": stall_room,
                    "end_room": obs.room_type,
                    "hp_lost": sum(
                        int(h["lost"]) for h in hits if h["episode"] == episode
                    ),
                }
            )
            print(
                f"[w{job['worker']}] episode={episode:02d} outcome={outcome} "
                f"floor={obs.floor} rooms={obs.rooms} kills={obs.kills} "
                f"items={obs.item_count} score={env.get_score():.1f} "
                f"hits={sum(1 for h in hits if h['episode'] == episode)} "
                f"stall={worst_stall}@{stall_room}",
                flush=True,
            )

    return {"hits": hits, "arrivals": arrivals, "episodes": episodes}


def main() -> None:
    args = parse_args()
    if args.policy.startswith("hybrid:"):
        probe = HybridAgent.load(args.policy[7:]).net
        print(f"policy={args.policy} features={probe.n_features} heads={probe.heads}", flush=True)
    elif args.policy != "heuristic" and not args.policy.startswith("teacher"):
        probe = DQNAgent.load(Path(args.policy))
        print(
            f"policy={args.policy} features={probe.n_features} "
            f"actions={len(probe.actions)}",
            flush=True,
        )
    else:
        print(f"policy={args.policy}", flush=True)

    cfg = {
        "policy": args.policy,
        "max_steps": args.max_steps,
        "step_seconds": args.step_seconds,
        "epsilon": args.epsilon,
        "start_floor": args.start_floor,
        "seed": args.seed,
    }
    plan = list(range(1, args.episodes + 1))
    workers = max(1, args.workers)
    jobs = [
        {"worker": w, "episodes": plan[w::workers], "cfg": cfg}
        for w in range(workers)
    ]
    jobs = [job for job in jobs if job["episodes"]]

    if len(jobs) == 1:
        shards = [audit_shard(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            shards = list(pool.map(audit_shard, jobs))

    hits = [row for shard in shards for row in shard["hits"]]
    arrivals = [row for shard in shards for row in shard["arrivals"]]
    episodes = sorted(
        (row for shard in shards for row in shard["episodes"]),
        key=lambda row: row["episode"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if hits:
        with args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(hits[0]))
            writer.writeheader()
            writer.writerows(hits)
    # One row per episode, with its seed: what scripts/compare_paired.py
    # reads to set two policies side by side on the same dungeons.
    episodes_path = episodes_output(args.output)
    if episodes:
        with episodes_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(episodes[0]))
            writer.writeheader()
            writer.writerows(episodes)

    _report(hits, arrivals, episodes, args.output)
    print(f"episodes={episodes_path}")


def life_per_room(episodes: list[dict], seed: int = 0) -> tuple[float, float, float]:
    """Pooled life lost per room, with a 95 % bootstrap interval.

    Summing over episodes before dividing weighs every room the same, so a
    run that dies in its second room does not count as much as one that
    clears forty. Resampling whole episodes keeps the dependence between a
    run's hits and its rooms.
    """
    import random

    rooms = [int(row["rooms"]) for row in episodes]
    lost = [int(row["hp_lost"]) for row in episodes]
    if not episodes:
        return 0.0, 0.0, 0.0
    point = sum(lost) / max(1, sum(rooms))
    rng = random.Random(seed)
    n = len(episodes)
    draws = []
    for _ in range(2000):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(sum(lost[i] for i in idx) / max(1, sum(rooms[i] for i in idx)))
    draws.sort()
    return point, draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def episodes_output(output: Path) -> Path:
    return output.with_name(output.stem + "_episodes.csv")


def _report(
    hits: list[dict],
    arrivals: list[dict],
    episodes: list[dict],
    output: Path,
) -> None:
    print()
    outcomes = Counter(row["outcome"] for row in episodes)
    floors = Counter(row["floor"] for row in episodes)
    scores = [float(row["score"]) for row in episodes]
    print(
        f"episodes={len(episodes)} "
        f"victories={outcomes['victory']} deaths={outcomes['death']} "
        f"budget_exhausted={outcomes['budget']}"
    )
    # The mean alone decides nothing: episode scores span an order of
    # magnitude, so a change is only visible once it exceeds the error on
    # the mean it is compared against.
    stderr = (
        statistics.stdev(scores) / len(scores) ** 0.5 if len(scores) > 1 else 0.0
    )
    print(
        f"score avg={statistics.mean(scores):.2f} +/-{stderr:.2f} "
        f"min={min(scores):.2f} max={max(scores):.2f}"
    )
    print(
        f"rooms={sum(int(r['rooms']) for r in episodes)} "
        f"floors={sum(int(r['floors']) for r in episodes)}"
    )
    print("floor reached: " + " ".join(f"{f}:{n}" for f, n in sorted(floors.items())))

    stalled = [row for row in episodes if int(row["stall"]) > 1500]
    print(
        f"stalls >1500 steps without clearing a room: {len(stalled)}/{len(episodes)} "
        f"(median worst stall {statistics.median(int(r['stall']) for r in episodes):.0f} steps)"
    )
    if stalled:
        rooms = Counter(row["stall_room"] for row in stalled)
        print("  stuck in: " + " ".join(f"{k}:{v}" for k, v in rooms.most_common()))

    if not hits:
        print("no life lost at all")
        return

    print()
    total_lost = sum(int(h["lost"]) for h in hits)
    total_rooms = sum(int(row["rooms"]) for row in episodes)
    # Going deeper naturally means taking more hits in absolute terms. What
    # says whether the policy got better at surviving is the rate.
    point, low, high = life_per_room(episodes)
    print(
        f"life lost: {total_lost} over {len(hits)} hits "
        f"({sum(1 for h in hits if h.get('fatal'))} fatal), "
        f"{total_rooms} rooms cleared -> {point:.3f} per room "
        f"[{low:.3f}, {high:.3f}] 95 %"
    )

    per_floor: dict[int, int] = defaultdict(int)
    for hit in hits:
        per_floor[int(hit["floor"])] += int(hit["lost"])
    print("  by floor:  " + " ".join(f"{f}:{n}" for f, n in sorted(per_floor.items())))

    per_killer: Counter = Counter()
    for hit in hits:
        per_killer[hit["killer"] or "?"] += int(hit["lost"])
    print("  by killer:")
    for name, lost in per_killer.most_common(12):
        print(f"    {lost:4d}  {name}")

    per_attack: Counter = Counter()
    for hit in hits:
        if hit["boss_attack"]:
            per_attack[f"{hit['boss']}/{hit['boss_attack']}"] += int(hit["lost"])
    if per_attack:
        print("  by boss attack:")
        for name, lost in per_attack.most_common(12):
            print(f"    {lost:4d}  {name}")

    print()
    print("floor entry (curriculum calibration):")
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in arrivals:
        grouped[int(row["floor"])].append(row)
    for floor in sorted(grouped):
        rows = grouped[floor]
        print(
            f"  floor {floor}: n={len(rows):3d} "
            f"items={statistics.median(r['items'] for r in rows):.0f} "
            f"gold={statistics.median(r['gold'] for r in rows):.0f} "
            f"hp={statistics.median(r['hp'] for r in rows):.0f}"
            f"/{statistics.median(r['max_hp'] for r in rows):.0f}"
        )

    print()
    print(f"output={output}")


if __name__ == "__main__":
    main()
