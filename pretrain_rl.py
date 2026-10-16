"""Inicializa el DQN con Q ≈ R_SAFE·(1-p) + R_MINE·p, usando P(mina) de la red neuronal.

Después se afina con train_rl.py --resume.
"""
import argparse

import numpy as np

from minas.agents import NN_PATH, R_MINE, R_SAFE, RL_PATH, NNAgent
from minas.features import extract, frontier
from minas.game import Minesweeper
from minas.nn import MLP, sigmoid
from minas.agents import step_reward
from train_nn import SIZES, snapshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    teacher = NNAgent()
    X = []
    for _ in range(a.boards):
        h, w = SIZES[rng.integers(len(SIZES))]
        g = Minesweeper(h, w, density=(0.08, 0.20), rng=rng)
        while not g.over:
            if g.revealed.any():
                f = extract(g.covered, g.numbers, g.n_mines)
                idx = np.flatnonzero(frontier(g.covered).ravel())
                if len(idx):
                    X.append(f[rng.choice(idx, size=min(len(idx), 25), replace=False)])
            step_reward(g, *teacher.decide(g).cell)
    X = np.concatenate(X)
    p = sigmoid(teacher.net.forward(X)[:, 0])
    Y = (R_SAFE * (1 - p) + R_MINE * p).astype(np.float32)
    print(f"{len(X)} ejemplos")
    net = MLP(teacher.net.sizes, a.seed)
    W = (1.0 / (p + 0.03) ** 2).astype(np.float32)
    W /= W.mean()
    for ep in range(a.epochs):
        perm = rng.permutation(len(X))
        tot = 0.0
        for i in range(0, len(X), 512):
            b = perm[i:i + 512]
            err = net.forward(X[b])[:, 0] - Y[b]
            tot += float((W[b] * err ** 2).sum())
            net.backward_step((2 * W[b] * err / len(b))[:, None].astype(np.float32),
                              lr=1e-3 * 0.5 * (1 + np.cos(np.pi * ep / a.epochs)))
        print(f"  época {ep + 1}/{a.epochs} mse={tot / len(X):.5f}", flush=True)
    net.save(RL_PATH, episodes=0)


if __name__ == "__main__":
    main()
