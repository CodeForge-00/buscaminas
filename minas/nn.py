"""Perceptrón multicapa mínimo en numpy (ReLU + Adam)."""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, sizes, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W = [(rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]
        self._m = [np.zeros_like(p) for p in self.W + self.b]
        self._v = [np.zeros_like(p) for p in self.W + self.b]
        self._t = 0
        self._acts: list[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Salida lineal (N, out). Guarda las activaciones para backward()."""
        self._acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0)
            self._acts.append(x)
        return x

    def hidden(self) -> list[np.ndarray]:
        """Activaciones de las capas ocultas del último forward()."""
        return self._acts[1:-1]

    def backward_step(self, dout: np.ndarray, lr: float = 1e-3, clip: float = 5.0) -> None:
        """dout = dLoss/dSalida (ya promediada sobre el batch). Aplica Adam."""
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for i in reversed(range(len(self.W))):
            gW[i] = self._acts[i].T @ d
            gb[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (self._acts[i] > 0)
        params, grads = self.W + self.b, gW + gb
        self._t += 1
        for k, (p, g) in enumerate(zip(params, grads)):
            g = np.clip(g, -clip, clip)
            self._m[k] = 0.9 * self._m[k] + 0.1 * g
            self._v[k] = 0.999 * self._v[k] + 0.001 * g * g
            mh = self._m[k] / (1 - 0.9 ** self._t)
            vh = self._v[k] / (1 - 0.999 ** self._t)
            p -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)

    def copy_from(self, other: "MLP") -> None:
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def save(self, path: str, **extra) -> None:
        data = {f"W{i}": w for i, w in enumerate(self.W)}
        data.update({f"b{i}": b for i, b in enumerate(self.b)})
        np.savez(path, sizes=np.array(self.sizes), **data, **extra)

    @classmethod
    def load(cls, path: str) -> "MLP":
        z = np.load(path)
        net = cls(z["sizes"].tolist())
        net.W = [z[f"W{i}"] for i in range(len(net.W))]
        net.b = [z[f"b{i}"] for i in range(len(net.b))]
        return net


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
