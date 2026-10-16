"""Evalúa un agente jugando partidas sin interfaz."""
from __future__ import annotations

import numpy as np

from .agents import step_reward
from .game import WON, Minesweeper


def play_game(agent, h, w, rng, **kw):
    game = Minesweeper(h, w, rng=rng)
    while not game.over:
        d = agent.decide(game, **kw)
        step_reward(game, *d.cell)
    safe_total = h * w - game.n_mines
    return game.state == WON, game.revealed.sum() / safe_total


def evaluate(agent, games=200, sizes=((9, 9),), seed=123, **kw):
    rng = np.random.default_rng(seed)
    wins, prog = 0, 0.0
    for i in range(games):
        h, w = sizes[i % len(sizes)]
        won, p = play_game(agent, h, w, rng, **kw)
        wins += won
        prog += p
    return wins / games, prog / games
