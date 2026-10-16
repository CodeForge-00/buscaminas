"""Entrena la red neuronal supervisada (P(mina) por celda).

Ronda 1: estados generados con un "oráculo" que abre celdas seguras al azar.
Ronda 2: estados generados por la propia red jugando (corrige su sesgo).
"""
import argparse
import os
import time

import numpy as np

from minas.agents import NN_PATH, NNAgent, step_reward
from minas.evaluate import evaluate
from minas.features import N_FEATURES, extract, frontier
from minas.game import Minesweeper
from minas.nn import MLP, sigmoid

SIZES = [(8, 8), (9, 9), (10, 10), (12, 12), (16, 16)]


def snapshot(game, out_x, out_y, rng, cap=40):
    cov = game.covered
    feats = extract(cov, game.numbers, game.n_mines)
    front = frontier(cov).ravel()
    idx = np.flatnonzero(front)
    rest = np.flatnonzero(cov.ravel() & ~front)
    if len(rest):
        idx = np.concatenate([idx, rng.choice(rest, size=min(len(rest), max(3, len(idx) // 4)), replace=False)])
    if len(idx) > cap:
        idx = rng.choice(idx, size=cap, replace=False)
    out_x.append(feats[idx])
    out_y.append(game.mines.ravel()[idx].astype(np.float32))


def gen_oracle(n_boards, rng):
    X, Y = [], []
    for _ in range(n_boards):
        h, w = SIZES[rng.integers(len(SIZES))]
        g = Minesweeper(h, w, density=(0.06, 0.24), rng=rng)
        g.reveal(int(rng.integers(h)), int(rng.integers(w)))
        while not g.over:
            if rng.random() < 0.2:
                snapshot(g, X, Y, rng)
            safe = g.covered & ~g.mines
            pool = safe & frontier(g.covered) if rng.random() < 0.8 else safe
            if not pool.any():
                pool = safe
            cells = np.argwhere(pool)
            g.reveal(*cells[rng.integers(len(cells))])
    return np.concatenate(X), np.concatenate(Y)


def gen_selfplay(agent, n_boards, rng):
    X, Y = [], []
    for _ in range(n_boards):
        h, w = SIZES[rng.integers(len(SIZES))]
        g = Minesweeper(h, w, density=(0.06, 0.24), rng=rng)
        while not g.over:
            if g.revealed.any():
                snapshot(g, X, Y, rng, cap=25)
            step_reward(g, *agent.decide(g).cell)
    return np.concatenate(X), np.concatenate(Y)


def fit(net, X, Y, epochs, lr, bs=512):
    n = len(X)
    for ep in range(epochs):
        perm = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            p = sigmoid(net.forward(X[b])[:, 0])
            y = Y[b]
            tot += float(-np.mean(y * np.log(p + 1e-7) + (1 - y) * np.log(1 - p + 1e-7))) * len(b)
            net.backward_step(((p - y) / len(b))[:, None].astype(np.float32), lr=lr)
        print(f"  época {ep + 1}/{epochs}  loss={tot / n:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=int, default=2500)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--selfplay-boards", type=int, default=1500)
    ap.add_argument("--rounds", type=int, default=1, help="rondas de autojuego")
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    net = MLP([N_FEATURES, *a.hidden, 1], a.seed)
    print("Ronda 1: datos del oráculo...", flush=True)
    X, Y = gen_oracle(a.boards, rng)
    print(f"  {len(X)} ejemplos, {Y.mean():.1%} minas ({time.time() - t0:.0f}s)")
    fit(net, X, Y, a.epochs, 2e-3)
    os.makedirs(os.path.dirname(NN_PATH), exist_ok=True)
    net.save(NN_PATH)
    agent = NNAgent()
    print("  evaluación 9x9:", evaluate(agent, 100))

    for r in range(a.rounds):
        print(f"Autojuego {r + 1}/{a.rounds}...", flush=True)
        X2, Y2 = gen_selfplay(agent, a.selfplay_boards, rng)
        print(f"  {len(X2)} ejemplos ({time.time() - t0:.0f}s)")
        X, Y = np.concatenate([X, X2]), np.concatenate([Y, Y2])
        fit(net, X, Y, max(4, a.epochs // 2), 1e-3 / (1 + r))
        net.save(NN_PATH)
        agent = NNAgent()
    for size in [(9, 9), (12, 12), (16, 16)]:
        wr, prog = evaluate(agent, 100, sizes=(size,))
        print(f"  {size[0]}x{size[1]}: gana {wr:.0%}, progreso medio {prog:.0%}")
    print(f"Guardado en {NN_PATH} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
