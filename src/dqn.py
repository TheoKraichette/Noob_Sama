from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# The factored movement vocabulary (eos_env.MOVE_ACTIONS): a direction
# and a modifier. Index 0 of each = none / walk.
DIRECTIONS = ["none", "up", "down", "left", "right", "up_left", "up_right", "down_left", "down_right"]
MODIFIERS = ["walk", "dash", "skill", "interact"]


def _factor(action: str) -> tuple[int, int]:
    if action in ("skill", "interact"):
        return 0, MODIFIERS.index(action)
    if action.startswith("dash"):
        rest = action[5:] if action.startswith("dash_") else "none"
        return DIRECTIONS.index(rest), 1
    return DIRECTIONS.index("none" if action == "noop" else action), 0


def compose(direction: int, modifier: int) -> str:
    mod = MODIFIERS[modifier]
    d = DIRECTIONS[direction]
    if mod in ("skill", "interact"):
        return mod
    if mod == "dash":
        return "dash" if d == "none" else "dash_" + d
    return "noop" if d == "none" else d


def _softmax_ce(logits, target, w, smoothing):
    """Weighted, label-smoothed softmax cross-entropy and its gradient."""
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    n, k = p.shape
    rows = np.arange(n)
    onehot = np.full((n, k), smoothing / k, dtype=np.float32)
    onehot[rows, target] += 1.0 - smoothing
    loss = float(-(w * (onehot * np.log(p + 1e-12)).sum(axis=1)).sum())
    grad = (p - onehot) * w[:, None]
    return loss, grad


class ReplayBuffer:
    """Fixed-size circular buffer of transitions, stored as numpy arrays."""

    def __init__(self, capacity: int, n_features: int) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, n_features), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, n_features), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.index = 0
        self.size = 0

    def add(self, state, action: int, reward: float, next_state, done: bool) -> None:
        i = self.index
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


class DQNAgent:
    """Small MLP Q-network trained with Double DQN, Huber loss and Adam.

    Pure numpy on purpose: the network is tiny (two hidden layers), the
    save format stays a readable JSON like the tabular agent, and the
    project needs no deep-learning dependency.
    """

    def __init__(
        self,
        n_features: int,
        actions: list[str],
        *,
        hidden: tuple[int, ...] = (128, 128),
        gamma: float = 0.99,
        lr: float = 1e-3,
        seed: int | None = None,
        heads: bool = False,
        feature_version: str = "v1",
        weight_decay: float = 0.0,
    ) -> None:
        self.n_features = n_features
        self.actions = list(actions)
        self.hidden = tuple(hidden)
        self.gamma = gamma
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self.steps_trained = 0
        self.episodes_trained = 0
        self.feature_version = feature_version
        self.weight_decay = weight_decay
        # Two heads factor a movement action into its direction and its
        # modifier (walk, dash, skill, interact): "dash up-left" and
        # "up-left" stop being two competing classes on near-identical
        # states, and the rare modifiers get a head of their own.
        self.heads = bool(heads)
        if self.heads:
            self._head_index = [_factor(a) for a in self.actions]
            n_out = len(DIRECTIONS) + len(MODIFIERS)
        else:
            n_out = len(self.actions)

        sizes = [n_features, *self.hidden, n_out]
        self.params: list[np.ndarray] = []
        for fan_in, fan_out in zip(sizes, sizes[1:]):
            scale = np.sqrt(2.0 / fan_in)
            self.params.append(self.rng.normal(0.0, scale, (fan_in, fan_out)).astype(np.float32))
            self.params.append(np.zeros(fan_out, dtype=np.float32))
        self.target_params = [p.copy() for p in self.params]

        self._adam_t = 0
        self._adam_m = [np.zeros_like(p) for p in self.params]
        self._adam_v = [np.zeros_like(p) for p in self.params]

    # ----- inference -----

    def _forward(self, x: np.ndarray, params: list[np.ndarray]) -> list[np.ndarray]:
        """Return the activation of every layer (input included)."""
        activations = [x]
        n_layers = len(params) // 2
        for layer in range(n_layers):
            w, b = params[2 * layer], params[2 * layer + 1]
            x = x @ w + b
            if layer < n_layers - 1:
                x = np.maximum(x, 0.0)
            activations.append(x)
        return activations

    def q_values(self, features) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return self._forward(x, self.params)[-1][0]

    def act(self, features, *, epsilon: float = 0.0) -> str:
        if epsilon > 0 and self.rng.random() < epsilon:
            return self.actions[int(self.rng.integers(len(self.actions)))]
        out = self.q_values(features)
        if not self.heads:
            return self.actions[int(np.argmax(out))]
        nd = len(DIRECTIONS)
        return compose(int(np.argmax(out[:nd])), int(np.argmax(out[nd:])))

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Index into `actions` for a batch of states (heads decoded)."""
        out = self._forward(states.astype(np.float32), self.params)[-1]
        if not self.heads:
            return out.argmax(axis=1)
        nd = len(DIRECTIONS)
        lookup = {a: i for i, a in enumerate(self.actions)}
        d = out[:, :nd].argmax(axis=1)
        m = out[:, nd:].argmax(axis=1)
        return np.array([lookup.get(compose(int(a), int(b)), 0) for a, b in zip(d, m)])

    # ----- training -----

    def train_step(self, batch) -> float:
        states, actions, rewards, next_states, dones = batch
        n = len(actions)
        rows = np.arange(n)

        # Double DQN: the online net picks the next action, the target net
        # evaluates it (limits the max-operator over-estimation).
        next_online = self._forward(next_states, self.params)[-1]
        best_next = next_online.argmax(axis=1)
        next_target = self._forward(next_states, self.target_params)[-1]
        targets = rewards + self.gamma * (1.0 - dones) * next_target[rows, best_next]

        activations = self._forward(states, self.params)
        q = activations[-1]
        td = q[rows, actions] - targets

        # Gradient of the Huber loss (delta=1) at the taken actions only.
        grad = np.zeros_like(q)
        grad[rows, actions] = np.clip(td, -1.0, 1.0) / n
        self._backward(activations, grad)
        self.steps_trained += 1
        return float(np.mean(np.abs(td)))

    def train_step_bc(
        self,
        states: np.ndarray,
        action_idx: np.ndarray,
        weights: np.ndarray | None = None,
        smoothing: float = 0.0,
        head_weight: float = 2.0,
    ) -> float:
        """Behaviour-cloning step: softmax cross-entropy on demo actions.

        `weights` re-weighs each pair (rare classes, guardian rooms);
        `smoothing` is label smoothing against an ambiguous teacher. With
        two heads the loss is CE(direction) + head_weight * CE(modifier).
        The same output serves as logits here and as Q-values for TD
        fine-tuning, as in learning-from-demonstrations setups.
        """
        activations = self._forward(states, self.params)
        logits = activations[-1]
        n = len(action_idx)
        w = np.ones(n, dtype=np.float32) if weights is None else weights.astype(np.float32)
        w = w / w.sum()
        if self.heads:
            nd = len(DIRECTIONS)
            factors = np.array([self._head_index[i] for i in action_idx])
            loss_a, grad_a = _softmax_ce(logits[:, :nd], factors[:, 0], w, smoothing)
            loss_b, grad_b = _softmax_ce(logits[:, nd:], factors[:, 1], w, smoothing)
            grad = np.concatenate([grad_a, head_weight * grad_b], axis=1)
            loss = loss_a + head_weight * loss_b
        else:
            loss, grad = _softmax_ce(logits, action_idx, w, smoothing)
        self._backward(activations, grad)
        self.steps_trained += 1
        return float(loss)

    def _backward(self, activations: list[np.ndarray], grad_out: np.ndarray) -> None:
        grads: list[np.ndarray] = [None] * len(self.params)
        delta = grad_out
        n_layers = len(self.params) // 2
        for layer in range(n_layers - 1, -1, -1):
            a_in = activations[layer]
            grads[2 * layer] = a_in.T @ delta
            grads[2 * layer + 1] = delta.sum(axis=0)
            if layer > 0:
                delta = delta @ self.params[2 * layer].T
                delta = delta * (activations[layer] > 0)
        self._adam_update(grads)

    def _adam_update(self, grads: list[np.ndarray], beta1=0.9, beta2=0.999, eps=1e-8) -> None:
        self._adam_t += 1
        lr_t = self.lr * np.sqrt(1 - beta2**self._adam_t) / (1 - beta1**self._adam_t)
        for i, grad in enumerate(grads):
            self._adam_m[i] = beta1 * self._adam_m[i] + (1 - beta1) * grad
            self._adam_v[i] = beta2 * self._adam_v[i] + (1 - beta2) * grad * grad
            self.params[i] -= lr_t * self._adam_m[i] / (np.sqrt(self._adam_v[i]) + eps)
            if self.weight_decay and self.params[i].ndim == 2:
                self.params[i] -= self.lr * self.weight_decay * self.params[i]

    def sync_target(self) -> None:
        self.target_params = [p.copy() for p in self.params]

    # ----- persistence -----

    def save(self, path: Path | str, *, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "algo": "dqn",
            "actions": self.actions,
            "n_features": self.n_features,
            "hidden": list(self.hidden),
            "gamma": self.gamma,
            "lr": self.lr,
            "steps_trained": self.steps_trained,
            "episodes_trained": self.episodes_trained,
            "feature_version": self.feature_version,
            "heads": self.heads,
            "params": [np.round(p, 6).tolist() for p in self.params],
        }
        if extra:
            payload.update(extra)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        seed: int | None = None,
        expect: tuple[int, str] | None = None,
    ) -> "DQNAgent":
        """Reload a saved agent; `expect` = (n_features, version) of the
        environment it will play in, refused loudly when they differ."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = data.get("feature_version", "v1")
        if expect is not None and (data["n_features"], version) != tuple(expect):
            raise ValueError(
                f"{path}: model built on {data['n_features']} features ({version}), "
                f"environment produces {expect[0]} ({expect[1]})"
            )
        agent = cls(
            data["n_features"],
            data["actions"],
            hidden=tuple(data["hidden"]),
            gamma=data["gamma"],
            lr=data["lr"],
            seed=seed,
            heads=data.get("heads", False),
            feature_version=version,
        )
        agent.params = [np.asarray(p, dtype=np.float32) for p in data["params"]]
        agent.target_params = [p.copy() for p in agent.params]
        agent._adam_m = [np.zeros_like(p) for p in agent.params]
        agent._adam_v = [np.zeros_like(p) for p in agent.params]
        agent.steps_trained = data.get("steps_trained", 0)
        agent.episodes_trained = data.get("episodes_trained", 0)
        return agent
