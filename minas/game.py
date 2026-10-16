"""Motor del buscaminas.

Cada partida sorteá una cantidad distinta de minas (según un rango de densidad)
y las coloca en posiciones aleatorias *después* del primer click, de modo que
el primer click nunca pierde.
"""
from __future__ import annotations

import numpy as np

DEFAULT_DENSITY = (0.10, 0.20)

READY, PLAYING, WON, LOST = "ready", "playing", "won", "lost"


def neighbor_sum(a: np.ndarray) -> np.ndarray:
    """Suma de los 8 vecinos de cada celda."""
    h, w = a.shape
    p = np.pad(a, 1)
    out = np.zeros(a.shape, dtype=np.int32)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            out += p[dy:dy + h, dx:dx + w]
    return out


class Minesweeper:
    def __init__(self, h: int, w: int, n_mines: int | None = None,
                 density: tuple[float, float] = DEFAULT_DENSITY,
                 rng: np.random.Generator | None = None):
        self.h, self.w = h, w
        self.rng = rng or np.random.default_rng()
        if n_mines is None:
            n_mines = int(round(h * w * self.rng.uniform(*density)))
        self.n_mines = max(1, min(n_mines, h * w - 1))
        self.mines = np.zeros((h, w), dtype=bool)
        self.counts = np.zeros((h, w), dtype=np.int32)
        self.revealed = np.zeros((h, w), dtype=bool)
        self.flags = np.zeros((h, w), dtype=bool)
        self.state = READY
        self.exploded: tuple[int, int] | None = None
        self.moves = 0

    # ------------------------------------------------------------ consultas
    @property
    def covered(self) -> np.ndarray:
        return ~self.revealed

    @property
    def numbers(self) -> np.ndarray:
        """Números visibles para el jugador (0 en celdas cubiertas)."""
        return np.where(self.revealed, self.counts, 0)

    @property
    def over(self) -> bool:
        return self.state in (WON, LOST)

    @property
    def flags_left(self) -> int:
        return self.n_mines - int(self.flags.sum())

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.h and 0 <= c < self.w

    def neighbors(self, r: int, c: int):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and self.in_bounds(r + dr, c + dc):
                    yield r + dr, c + dc

    # -------------------------------------------------------------- acciones
    def _place_mines(self, safe_r: int, safe_c: int) -> None:
        forbidden = {(safe_r, safe_c), *self.neighbors(safe_r, safe_c)}
        cells = [(r, c) for r in range(self.h) for c in range(self.w)]
        allowed = [x for x in cells if x not in forbidden]
        if len(allowed) < self.n_mines:  # tablero muy chico: solo protege la celda
            allowed = [x for x in cells if x != (safe_r, safe_c)]
        idx = self.rng.choice(len(allowed), size=self.n_mines, replace=False)
        for i in idx:
            self.mines[allowed[i]] = True
        self.counts = neighbor_sum(self.mines.astype(np.int32))

    def reveal(self, r: int, c: int) -> int:
        """Descubre una celda. Devuelve cuántas celdas nuevas se abrieron."""
        if self.over or not self.in_bounds(r, c) or self.revealed[r, c] or self.flags[r, c]:
            return 0
        if self.state == READY:
            self._place_mines(r, c)
            self.state = PLAYING
        self.moves += 1
        if self.mines[r, c]:
            self.revealed[r, c] = True
            self.exploded = (r, c)
            self.state = LOST
            return 1
        opened = 0
        stack = [(r, c)]
        while stack:
            y, x = stack.pop()
            if self.revealed[y, x] or self.flags[y, x]:
                continue
            self.revealed[y, x] = True
            opened += 1
            if self.counts[y, x] == 0:
                stack.extend(self.neighbors(y, x))
        if self.revealed.sum() == self.h * self.w - self.n_mines:
            self.state = WON
        return opened

    def toggle_flag(self, r: int, c: int) -> None:
        if not self.over and self.in_bounds(r, c) and not self.revealed[r, c]:
            self.flags[r, c] = not self.flags[r, c]

    def chord(self, r: int, c: int) -> int:
        """Click sobre un número ya abierto con tantas banderas como indica."""
        if self.over or not self.revealed[r, c] or self.counts[r, c] == 0:
            return 0
        nb = list(self.neighbors(r, c))
        if sum(self.flags[y, x] for y, x in nb) != self.counts[r, c]:
            return 0
        opened = 0
        for y, x in nb:
            if not self.flags[y, x]:
                opened += self.reveal(y, x)
        return opened
