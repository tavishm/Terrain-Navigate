from __future__ import annotations
from heapq import heappush, heappop
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Coord = Tuple[int, int]


def reconstruct_path(came_from: Dict[Coord, Coord], current: Coord) -> List[Coord]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def a_star(
    start: Coord,
    goal: Coord,
    neighbors_fn: Callable[[Coord], Sequence[Coord]],
    heuristic_fn: Callable[[Coord], float],
    move_cost_fn: Callable[[Coord, Coord], float],
    max_iter: int | None = None,
) -> Tuple[List[Coord], float]:
    """Generic A* search.

    Returns (path, cost). Empty path if no route.
    """
    open_heap: List[Tuple[float, Coord]] = []
    heappush(open_heap, (0.0, start))
    g_score: Dict[Coord, float] = {start: 0.0}
    f_score: Dict[Coord, float] = {start: heuristic_fn(start)}
    came_from: Dict[Coord, Coord] = {}

    closed: set[Coord] = set()
    iterations = 0

    while open_heap:
        _, current = heappop(open_heap)
        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, g_score[current]
        if current in closed:
            continue
        closed.add(current)

        for nb in neighbors_fn(current):
            tentative_g = g_score[current] + move_cost_fn(current, nb)
            if tentative_g < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb] = tentative_g
                f = tentative_g + heuristic_fn(nb)
                f_score[nb] = f
                heappush(open_heap, (f, nb))

        iterations += 1
        if max_iter and iterations >= max_iter:
            break

    return [], float("inf")
