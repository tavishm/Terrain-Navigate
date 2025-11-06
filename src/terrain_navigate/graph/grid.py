from __future__ import annotations
from math import sqrt
from typing import Iterable, Tuple

Coord = Tuple[int, int]


def iter_neighbors(idx: Coord, shape: Tuple[int, int], connectivity: int = 8) -> Iterable[Coord]:
    i, j = idx
    m, n = shape
    steps4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    steps8 = steps4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    steps = steps8 if connectivity == 8 else steps4
    for di, dj in steps:
        ni, nj = i + di, j + dj
        if 0 <= ni < m and 0 <= nj < n:
            yield (ni, nj)


def step_distance(a: Coord, b: Coord) -> float:
    ai, aj = a
    bi, bj = b
    di = abs(ai - bi)
    dj = abs(aj - bj)
    if di + dj == 1:
        return 1.0
    # diagonal
    return sqrt(2.0)
