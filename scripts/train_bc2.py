"""Clone the phase 5 teacher's pilot from demos2 files.

Two heads (direction, modifier), weighted pairs and a smoothed label:

- class weight (freq_max / freq) ** 0.5, capped at 8: dashes, the skill
  and E are a few percent of the labels and would otherwise be learned
  as noise;
- state weight 3 in a guardian's room, 2 on floors 3 and up, 0.5 on the
  states the inventory reflex handles at play time;
- label smoothing 0.05 and a decoupled weight decay, because the teacher
  is not always sure either.

A checkpoint is written at every epoch; the best validation accuracy goes
to --output, and the game decides between the last few (audit_deaths.py
--policy hybrid:...).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import features2, features3
from src.dqn import DQNAgent

FEATURE_MODULES = {"v2": features2, "v3": features3}
from src.eos_env import MOVE_ACTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, nargs="+", default=[ROOT / "data" / "demos2.npz"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs=2, default=(384, 384))
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoothing", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-cap", type=float, default=8.0)
    parser.add_argument("--no-heads", action="store_true", help="One softmax over the 20 actions.")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "bc2_agent.json")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--init", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parts = {k: [] for k in ("states", "actions", "sources", "floors", "bosses")}
    version = None
    for path in args.demos:
        data = np.load(path)
        this = str(data["feature_version"]) if "feature_version" in data else "?"
        if version is None:
            version = this
        if this != version or this not in FEATURE_MODULES:
            raise SystemExit(f"{path}: features {this}, expected {version}")
        for k in parts:
            parts[k].append(data[k])
        print(f"loaded {len(data['actions'])} pairs from {path}")
    states = np.concatenate(parts["states"])
    actions = np.concatenate(parts["actions"])
    sources = np.concatenate(parts["sources"])
    floors = np.concatenate(parts["floors"])
    bosses = np.concatenate(parts["bosses"])
    n, n_features = states.shape
    module = FEATURE_MODULES[version]
    if n_features != module.N_FEATURES:
        raise SystemExit(f"demos have {n_features} features, {version} produces {module.N_FEATURES}")
    print(f"demos: {n} pairs, {n_features} features")

    counts = np.bincount(actions, minlength=len(MOVE_ACTIONS)).astype(np.float64)
    class_w = np.minimum(args.class_cap, np.sqrt(counts.max() / np.maximum(counts, 1.0)))
    weights = class_w[actions]
    weights *= np.where(bosses > 0, 3.0, 1.0)
    weights *= np.where(floors >= 3, 2.0, 1.0)
    weights *= np.where(sources > 0, 0.5, 1.0)
    weights = weights.astype(np.float32)
    print("class weights: " + " ".join(f"{a}:{w:.1f}" for a, w in zip(MOVE_ACTIONS, class_w)))

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    n_val = max(1, int(n * args.val_split))
    val_idx, train_idx = order[:n_val], order[n_val:]

    if args.init:
        agent = DQNAgent.load(args.init, seed=args.seed, expect=(n_features, version))
        agent.lr = args.lr
        agent.weight_decay = args.weight_decay
    else:
        agent = DQNAgent(
            n_features, list(MOVE_ACTIONS), hidden=tuple(args.hidden), lr=args.lr, seed=args.seed,
            heads=not args.no_heads, feature_version=version,
            weight_decay=args.weight_decay,
        )

    def accuracy(idx: np.ndarray) -> float:
        correct = 0
        for start in range(0, len(idx), 4096):
            batch = idx[start:start + 4096]
            correct += int((agent.predict(states[batch]) == actions[batch]).sum())
        return correct / len(idx)

    args.checkpoints.mkdir(parents=True, exist_ok=True)
    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_idx)
        losses = []
        for start in range(0, len(train_idx), args.batch_size):
            batch = train_idx[start:start + args.batch_size]
            losses.append(agent.train_step_bc(
                states[batch].astype(np.float32), actions[batch], weights[batch], args.smoothing,
            ))
        val_acc = accuracy(val_idx)
        print(f"epoch={epoch:02d} loss={np.mean(losses):.4f} val_acc={val_acc:.3f}", flush=True)
        agent.save(args.checkpoints / f"bc2_epoch{epoch:02d}.json",
                   extra={"algo": "bc", "val_accuracy": val_acc, "epoch": epoch})
        if val_acc > best_val:
            best_val = val_acc
            agent.save(args.output, extra={"algo": "bc", "val_accuracy": best_val, "epoch": epoch})
    print()
    print(f"best={args.output} (val accuracy {best_val:.3f})")


if __name__ == "__main__":
    main()
