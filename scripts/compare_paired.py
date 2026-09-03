"""Set two policies side by side on the same dungeons.

`audit_deaths.py` writes one row per episode with the seed it played. When
two audits share their seeds, the difference between them can be measured
episode by episode, and the variance between dungeons — which alone spans an
order of magnitude in score — drops out of the comparison. The standard
error printed here is that of the paired difference, not of either mean.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path


COLUMNS = ("score", "rooms", "floors", "kills", "hp_lost")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path, help="*_episodes.csv of the reference")
    parser.add_argument("after", type=Path, help="*_episodes.csv of the candidate")
    return parser.parse_args()


def load(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    keyed = {row["seed"]: row for row in rows if row.get("seed")}
    if not keyed:
        sys.exit(f"{path}: no seeded episodes (run audit_deaths.py with --seed >= 0)")
    return keyed


def paired_rate_interval(
    before: dict[str, dict[str, str]],
    after: dict[str, dict[str, str]],
    seeds: list[str],
    draws: int = 2000,
) -> tuple[float, float]:
    """95 % bootstrap interval on the difference of pooled life per room.

    The two policies are resampled on the same dungeons at once, so the
    layouts' share of the variance drops out here as it does for the means.
    """
    import random

    rng = random.Random(0)
    n = len(seeds)
    diffs = []
    for _ in range(draws):
        picked = [seeds[rng.randrange(n)] for _ in range(n)]
        rate = []
        for table in (before, after):
            lost = sum(float(table[s]["hp_lost"]) for s in picked)
            rooms = sum(float(table[s]["rooms"]) for s in picked)
            rate.append(lost / max(1.0, rooms))
        diffs.append(rate[1] - rate[0])
    diffs.sort()
    return diffs[int(0.025 * draws)], diffs[int(0.975 * draws) - 1]


def main() -> None:
    args = parse_args()
    before = load(args.before)
    after = load(args.after)
    seeds = sorted(set(before) & set(after), key=int)
    if len(seeds) < 2:
        sys.exit("fewer than two shared seeds: nothing to pair")
    print(f"paired episodes: {len(seeds)} (seeds {seeds[0]}..{seeds[-1]})")
    print()
    print(f"{'metric':<10}{'before':>10}{'after':>10}{'diff':>10}{'stderr':>10}  verdict")
    for column in COLUMNS:
        a = [float(before[s][column]) for s in seeds]
        b = [float(after[s][column]) for s in seeds]
        diffs = [y - x for x, y in zip(a, b)]
        mean_diff = statistics.mean(diffs)
        stderr = statistics.stdev(diffs) / len(diffs) ** 0.5
        # Two standard errors: the usual bar for calling a difference real.
        verdict = "real" if abs(mean_diff) > 2 * stderr else "noise"
        print(
            f"{column:<10}{statistics.mean(a):>10.2f}{statistics.mean(b):>10.2f}"
            f"{mean_diff:>+10.2f}{stderr:>10.2f}  {verdict}"
        )

    rooms_a = sum(float(before[s]["rooms"]) for s in seeds)
    rooms_b = sum(float(after[s]["rooms"]) for s in seeds)
    lost_a = sum(float(before[s]["hp_lost"]) for s in seeds)
    lost_b = sum(float(after[s]["hp_lost"]) for s in seeds)
    low, high = paired_rate_interval(before, after, seeds)
    print()
    print(
        f"life lost per room: {lost_a / max(1, rooms_a):.3f} -> "
        f"{lost_b / max(1, rooms_b):.3f} "
        f"(diff {lost_b / max(1, rooms_b) - lost_a / max(1, rooms_a):+.3f}, "
        f"95 % [{low:+.3f}, {high:+.3f}] -> "
        f"{'real' if low > 0 or high < 0 else 'noise'})"
    )
    wins_a = sum(1 for s in seeds if before[s]["outcome"] == "victory")
    wins_b = sum(1 for s in seeds if after[s]["outcome"] == "victory")
    better = sum(1 for s in seeds if float(after[s]["score"]) > float(before[s]["score"]))
    print(f"victories: {wins_a} -> {wins_b}; candidate scores higher on {better}/{len(seeds)} dungeons")


if __name__ == "__main__":
    main()
