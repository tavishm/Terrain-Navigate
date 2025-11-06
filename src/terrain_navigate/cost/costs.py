from __future__ import annotations
from typing import Tuple
from math import sqrt
import numpy as np

Coord = Tuple[int, int]


def heuristic_euclidean(goal: Coord):
    """Return a heuristic function h(n) = planar Euclidean distance to goal.
    Admissible if move cost >= planar distance.
    """

    def h(n: Coord) -> float:
        di = abs(goal[0] - n[0])
        dj = abs(goal[1] - n[1])
        # diagonal distance metric equivalent to Octile can be tighter, but this is fine
        return sqrt(float(di * di + dj * dj))

    return h


def slope_cost(H: np.ndarray, k: float = 1.0, p: float = 2.0):
    """Return a move cost function for a height grid H (m x n).

    Cost = base step distance * (1 + k * |tan(theta)|^p)
    where tan(theta) = dz / base_distance, dz = H[next]-H[curr].
    """
    from ..graph.grid import step_distance

    def c(a: Coord, b: Coord) -> float:
        base = step_distance(a, b)
        dz = float(H[b] - H[a])
        tan_theta = dz / max(base, 1e-8)
        return base * (1.0 + k * (abs(tan_theta) ** p))

    return c
