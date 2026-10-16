"""Entrena el agente DQN jugando partidas con minas y tamaños aleatorios."""
import argparse
import time

import numpy as np

from minas.agents import RL_PATH, RLAgent, step_reward
from minas.evaluate import evaluate
from minas.game import WON, Minesweeper

SIZES = [(8, 8), (9, 9), (10, 10), (12, 12), (16, 16)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--episodes", type=int, default=100000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    agent = RLAgent(RL_PATH if a.resume else None, a.seed)
    if a.lr:
        agent.lr = a.lr
    t0, deadline = time.time(), time.time() + a.minutes * 60
    wins = prog = steps = 0
    window = 300

    for ep in range(1, a.episodes + 1):
        frac = min((time.time() - t0) / (a.minutes * 60), 1.0)
        eps = max(0.05, 1.0 - frac / 0.6) if not a.resume else 0.05
        h, w = SIZES[rng.integers(len(SIZES))]
        g = Minesweeper(h, w, density=(0.08, 0.20), rng=rng)
        while not g.over:
            d = agent.decide(g, epsilon=eps)
            r = step_reward(g, *d.cell)
            agent.observe(d, r, g)
            steps += 1
            if steps % 4 == 0:
                agent.train_step()
        agent.episodes += 1
        wins += g.state == WON
        prog += g.revealed.sum() / (h * w - g.n_mines)
        if ep % window == 0:
            print(f"ep {ep:6d}  ε={eps:.2f}  victorias {wins / window:.0%}  "
                  f"progreso {prog / window:.0%}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
            wins = prog = 0
            agent.save()
        if time.time() > deadline:
            break

    agent.save()
    for size in [(9, 9), (12, 12)]:
        wr, p = evaluate(agent, 100, sizes=(size,), epsilon=0.0)
        print(f"Evaluación {size[0]}x{size[1]}: gana {wr:.0%}, progreso medio {p:.0%}")


if __name__ == "__main__":
    main()
