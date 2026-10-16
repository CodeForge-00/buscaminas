"""Agentes de IA: red neuronal supervisada y aprendizaje por refuerzo (DQN)."""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .features import extract, frontier, window_of
from .game import LOST, WON, Minesweeper
from .nn import MLP, sigmoid
from .features import N_FEATURES

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
NN_PATH = os.path.join(MODELS, "nn.npz")
RL_PATH = os.path.join(MODELS, "rl.npz")

HIDDEN = (128, 64)
R_SAFE, R_MINE, R_WIN = 0.1, -1.0, 1.0


@dataclass
class Decision:
    cell: tuple
    kind: str                      # "prob" (menor = más seguro) | "q" (mayor = mejor)
    scores: np.ndarray | None      # (h, w) con nan en celdas ya abiertas
    lines: list = field(default_factory=list)
    window: np.ndarray | None = None   # (4, 5, 5) lo que ve la red en esa celda
    hidden: list = field(default_factory=list)
    feat: np.ndarray | None = None
    explore: bool = False


def step_reward(game: Minesweeper, r: int, c: int) -> float:
    game.reveal(r, c)
    if game.state == LOST:
        return R_MINE
    return R_SAFE + (R_WIN if game.state == WON else 0.0)


def opening_move(game: Minesweeper):
    return game.h // 2, game.w // 2


def _fmt(cell):
    return f"({cell[0] + 1},{cell[1] + 1})"


class NNAgent:
    """Red que estima P(mina) de cada celda tapada y abre la más segura."""
    name = "Red neuronal"
    kind = "prob"

    def __init__(self, path: str = NN_PATH):
        self.net = MLP.load(path)

    def decide(self, game: Minesweeper, **_) -> Decision:
        if not game.revealed.any():
            cell = opening_move(game)
            return Decision(cell, "prob", None, [
                "Tablero sin información: todavía no hay números que analizar.",
                f"Abro el centro {_fmt(cell)} (el primer click nunca es mina).",
            ])
        cov = game.covered
        feats = extract(cov, game.numbers, game.n_mines)
        p = sigmoid(self.net.forward(feats)[:, 0]).reshape(game.h, game.w)
        score = np.where(cov, p, np.inf) + np.random.random(p.shape) * 1e-6
        idx = int(np.argmin(score))
        cell = divmod(idx, game.w)
        self.net.forward(feats[idx:idx + 1])
        hidden = [h[0].copy() for h in self.net.hidden()]

        scores = np.where(cov, p, np.nan)
        order = np.argsort(score.ravel())[:3]
        n_cov = int(cov.sum())
        safe = int((p[cov] < 0.02).sum())
        lines = [
            f"Celdas tapadas: {n_cov} · minas: {game.n_mines} "
            f"(densidad media {game.n_mines / n_cov:.0%}).",
            f"La red estima P(mina) para cada celda; {safe} tienen < 2%.",
            f"Elijo {_fmt(cell)}: P(mina) = {p[cell]:.1%}, la más baja.",
        ]
        alts = [f"{_fmt(divmod(int(i), game.w))} {p.ravel()[i]:.1%}" for i in order[1:]]
        if alts:
            lines.append("Siguientes: " + " · ".join(alts))
        if p[cell] > 0.15:
            lines.append("No hay jugada segura: tengo que arriesgar.")
        return Decision(cell, "prob", scores, lines, window_of(feats[idx]), hidden, feats[idx])


class RLAgent:
    """DQN: Q(celda) = valor esperado de abrirla, con red compartida entre celdas."""
    name = "Refuerzo (DQN)"
    kind = "q"

    def __init__(self, path: str | None = None, seed: int = 0):
        self.net = MLP.load(path) if path else MLP([N_FEATURES, *HIDDEN, 1], seed)
        self.target = MLP(self.net.sizes)
        self.target.copy_from(self.net)
        self.buffer: deque = deque(maxlen=30000)
        self.epsilon = 0.0
        self.gamma = 0.9
        self.lr = 1e-3
        self.updates = 0
        self.episodes = 0
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------- decidir
    def decide(self, game: Minesweeper, epsilon: float | None = None, **_) -> Decision:
        eps = self.epsilon if epsilon is None else epsilon
        if not game.revealed.any():
            cell = opening_move(game)
            return Decision(cell, "q", None, [
                "Sin información todavía: abro el centro.",
                "(El valor Q aprendido solo aplica cuando hay números visibles.)",
            ])
        cov = game.covered
        feats = extract(cov, game.numbers, game.n_mines)
        q = self.net.forward(feats)[:, 0].reshape(game.h, game.w)
        scores = np.where(cov, q, np.nan)
        explore = self.rng.random() < eps
        if explore:
            front = frontier(cov)
            pool = np.argwhere(front if front.any() and self.rng.random() < 0.7 else cov)
            cell = tuple(pool[self.rng.integers(len(pool))])
            idx = cell[0] * game.w + cell[1]
        else:
            masked = np.where(cov, q, -np.inf) + self.rng.random(q.shape) * 1e-6
            idx = int(np.argmax(masked))
            cell = divmod(idx, game.w)
        self.net.forward(feats[idx:idx + 1])
        hidden = [h[0].copy() for h in self.net.hidden()]

        order = np.argsort(np.where(cov, -q, np.inf).ravel())[:3]
        lines = [
            f"La red asigna un valor Q a cada celda tapada ({int(cov.sum())}).",
            "Q = recompensa esperada de abrirla (+0.1 segura, −1 mina, +1 ganar).",
        ]
        if explore:
            lines.append(f"EXPLORACIÓN (ε={eps:.2f}): abro {_fmt(cell)} al azar "
                         f"para aprender, Q={q[cell]:+.3f}.")
        else:
            lines.append(f"Elijo {_fmt(cell)}: Q = {q[cell]:+.3f}, el máximo.")
            alts = [f"{_fmt(divmod(int(i), game.w))} {q.ravel()[i]:+.3f}" for i in order[1:]]
            if alts:
                lines.append("Siguientes: " + " · ".join(alts))
            if q[cell] < 0:
                lines.append("Todas las opciones tienen Q negativo: es una apuesta.")
        return Decision(cell, "q", scores, lines, window_of(feats[idx]), hidden,
                        feats[idx], explore)

    # ----------------------------------------------------------- aprender
    def observe(self, decision: Decision, reward: float, game: Minesweeper) -> None:
        """Guarda la transición (estado ya actualizado en `game`)."""
        if decision.feat is None:
            return
        done = game.over
        self.buffer.append((decision.feat, reward, done, game.covered.copy(),
                            game.numbers.astype(np.int8), game.n_mines))

    def train_step(self, batch_size: int = 64) -> float | None:
        if len(self.buffer) < max(batch_size, 500):
            return None
        picks = self.rng.integers(len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in picks]
        X = np.stack([b[0] for b in batch])
        y = np.array([b[1] for b in batch], dtype=np.float32)

        nxt = [(i, b) for i, b in enumerate(batch) if not b[2]]
        if nxt:
            F = [extract(b[3], b[4], b[5]) for _, b in nxt]
            q = self.target.forward(np.concatenate(F))[:, 0]
            pos = 0
            for (i, b), f in zip(nxt, F):
                qi = q[pos:pos + len(f)]
                pos += len(f)
                qi = np.where(b[3].ravel(), qi, -np.inf)
                y[i] += self.gamma * qi.max()

        pred = self.net.forward(X)[:, 0]
        err = pred - y
        loss = float(np.mean(np.where(np.abs(err) < 1, 0.5 * err ** 2, np.abs(err) - 0.5)))
        d = np.clip(err, -1, 1)[:, None] / batch_size
        self.net.backward_step(d.astype(np.float32), lr=self.lr)
        self.updates += 1
        if self.updates % 300 == 0:
            self.target.copy_from(self.net)
        return loss

    def save(self, path: str = RL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.net.save(path, episodes=self.episodes)


def load_agent(mode: str):
    """mode: 'nn' | 'rl'. Lanza FileNotFoundError si falta el modelo entrenado."""
    if mode == "nn":
        return NNAgent()
    if mode == "rl":
        agent = RLAgent(RL_PATH)
        return agent
    raise ValueError(mode)
