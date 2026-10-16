"""Representación del tablero para las redes: una ventana 5x5 por celda.

Canales de la ventana (lo que "ve" la red alrededor de cada celda):
  0 cubierta       1 si la celda sigue tapada
  1 número         número visible / 8
  2 razón local    número / celdas tapadas vecinas (1.0 => todas son minas)
  3 fuera          1 si la posición cae fuera del tablero
  4 segura         1 si se deduce con certeza que no es mina
  5 mina           1 si se deduce con certeza que es mina
  6 riesgo         mayor fracción de minas pendientes que le exige algún número vecino
Más 2 valores globales: densidad de minas restantes y fracción cubierta.
Como solo es información local, las redes sirven para cualquier tamaño.
"""
from __future__ import annotations

import numpy as np

RADIUS = 2
K = 2 * RADIUS + 1
CHANNELS = ("cubierta", "número", "razón", "fuera", "segura", "mina", "riesgo")
N_FEATURES = len(CHANNELS) * K * K + 2


def deduce(covered: np.ndarray, numbers: np.ndarray):
    """Deducción por restricciones (regla simple + regla de subconjuntos).

    Devuelve (segura, mina, riesgo): arrays (h, w) con las celdas deducidas y,
    para el resto, la mayor fracción minas_pendientes/celdas_inciertas entre los
    números vecinos (0 si ninguno la restringe).
    """
    h, w = covered.shape
    safe = np.zeros((h, w), dtype=bool)
    mine = np.zeros((h, w), dtype=bool)
    cons = []
    for r, c in zip(*np.nonzero(~covered & (numbers > 0))):
        cells = frozenset((rr, cc) for rr in range(max(r - 1, 0), min(r + 2, h))
                          for cc in range(max(c - 1, 0), min(c + 2, w)) if covered[rr, cc])
        if cells:
            cons.append((cells, int(numbers[r, c])))
    changed = True
    while changed:
        changed = False
        reduced = []
        for cells, n in cons:
            n -= sum(mine[x] for x in cells)
            cells = frozenset(x for x in cells if not mine[x] and not safe[x])
            if not cells:
                continue
            reduced.append((cells, n))
            if n <= 0 or n >= len(cells):
                target = mine if n >= len(cells) else safe
                for x in cells:
                    if not target[x]:
                        target[x] = True
                        changed = True
        if changed:
            continue
        for a, na in reduced:
            for b, nb in reduced:
                if len(a) < len(b) and a < b:
                    diff, nd = b - a, nb - na
                    if nd <= 0 or nd >= len(diff):
                        target = mine if nd >= len(diff) else safe
                        for x in diff:
                            if not target[x]:
                                target[x] = True
                                changed = True
        cons = reduced
    risk = np.zeros((h, w), dtype=np.float32)
    for cells, n in cons:
        cells = [x for x in cells if not mine[x] and not safe[x]]
        if not cells:
            continue
        frac = max(n, 0) / len(cells)
        for x in cells:
            risk[x] = max(risk[x], frac)
    return safe, mine, risk


def extract(covered: np.ndarray, numbers: np.ndarray, n_mines: int) -> np.ndarray:
    """Devuelve (h*w, N_FEATURES) float32: una fila por celda, en orden C."""
    h, w = covered.shape
    cov = covered.astype(np.float32)
    num = numbers.astype(np.float32)

    cp = np.pad(cov, 1)
    cn = np.zeros((h, w), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            if dy != 1 or dx != 1:
                cn += cp[dy:dy + h, dx:dx + w]
    ratio = np.where(~covered & (cn > 0), num / np.maximum(cn, 1), 0).astype(np.float32)

    safe, mine, risk = deduce(covered, numbers)
    planes = [
        np.pad(cov, RADIUS),
        np.pad(num / 8.0, RADIUS) * np.pad((~covered).astype(np.float32), RADIUS),
        np.pad(ratio, RADIUS),
        np.pad(np.zeros((h, w), dtype=np.float32), RADIUS, constant_values=1.0),
        np.pad(safe.astype(np.float32), RADIUS),
        np.pad(mine.astype(np.float32), RADIUS),
        np.pad(risk, RADIUS),
    ]
    feats = np.empty((h, w, len(planes), K, K), dtype=np.float32)
    for ch, p in enumerate(planes):
        for dy in range(K):
            for dx in range(K):
                feats[:, :, ch, dy, dx] = p[dy:dy + h, dx:dx + w]
    feats = feats.reshape(h * w, -1)

    n_cov = max(int(covered.sum()), 1)
    glob = np.empty((h * w, 2), dtype=np.float32)
    glob[:, 0] = min(n_mines / n_cov, 1.0)
    glob[:, 1] = n_cov / (h * w)
    return np.concatenate([feats, glob], axis=1)


def window_of(feats_row: np.ndarray) -> np.ndarray:
    """(4, 5, 5) con los canales de la ventana de una celda (para dibujarlos)."""
    return feats_row[:len(CHANNELS) * K * K].reshape(len(CHANNELS), K, K)


def frontier(covered: np.ndarray) -> np.ndarray:
    """Celdas tapadas con al menos un vecino ya abierto."""
    h, w = covered.shape
    p = np.pad((~covered).astype(np.int32), 1)
    near = np.zeros((h, w), dtype=np.int32)
    for dy in range(3):
        for dx in range(3):
            if dy != 1 or dx != 1:
                near += p[dy:dy + h, dx:dx + w]
    return covered & (near > 0)
