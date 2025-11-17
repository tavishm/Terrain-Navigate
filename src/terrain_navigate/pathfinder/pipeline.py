"""Bridge the legacy robotics pipeline configuration with Terrain-Navigate primitives."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt, tan, radians
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np

from .config import PipelineConfig
from ..cost.costs import heuristic_euclidean, slope_cost
from ..data.loader import load_synthetic_terrain
from ..graph.grid import iter_neighbors, step_distance
from ..planner.a_star import a_star

Coord = Tuple[int, int]
NeighborsFn = Callable[[Coord], Sequence[Coord]]
HeuristicFn = Callable[[Coord], float]
MoveCostFn = Callable[[Coord, Coord], float]


@dataclass(frozen=True)
class PlanResult:
    """Simple container with the computed path and its cumulative cost."""

    path: List[Coord]
    cost: float

    @property
    def success(self) -> bool:
        return bool(self.path)


def ensure_workspace(config: PipelineConfig) -> Path:
    """Create the workspace + cache directories if they do not already exist."""

    config.workspace.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    return config.workspace


def _neighbors_factory(heightmap: np.ndarray, connectivity: int) -> NeighborsFn:
    shape = heightmap.shape

    def neighbors(idx: Coord) -> Sequence[Coord]:
        return tuple(iter_neighbors(idx, shape, connectivity))

    return neighbors


def _heuristic_factory(goal: Coord, planner: str, tie_breaker: float) -> HeuristicFn:
    planner = planner.lower()
    if planner == "manhattan":
        def h(n: Coord) -> float:
            return tie_breaker * (abs(goal[0] - n[0]) + abs(goal[1] - n[1]))
        return h
    if planner == "octile":
        def h(n: Coord) -> float:
            di = abs(goal[0] - n[0])
            dj = abs(goal[1] - n[1])
            return tie_breaker * (max(di, dj) + (sqrt(2.0) - 1.0) * min(di, dj))
        return h
    base = heuristic_euclidean(goal)

    def h(n: Coord) -> float:
        return tie_breaker * base(n)

    return h


def _move_cost_factory(heightmap: np.ndarray, max_slope_deg: float) -> MoveCostFn:
    base_cost = slope_cost(heightmap, k=0.3, p=2.0)
    max_grade = tan(radians(max_slope_deg))

    def cost(a: Coord, b: Coord) -> float:
        dz = float(heightmap[b] - heightmap[a])
        grade = abs(dz) / max(step_distance(a, b), 1e-8)
        if grade > max_grade:
            return float("inf")
        return base_cost(a, b)

    return cost


def plan_route(
    config: PipelineConfig,
    start: Coord,
    goal: Coord,
    *,
    terrain: np.ndarray | None = None,
    connectivity: int = 8,
    max_iter: int | None = None,
) -> PlanResult:
    """Plan a route using the provided configuration and Terrain-Navigate planner."""

    ensure_workspace(config)
    heightmap = terrain if terrain is not None else load_synthetic_terrain()
    neighbors = _neighbors_factory(heightmap, connectivity)
    heuristic = _heuristic_factory(goal, config.planner.heuristic, config.planner.tie_breaker)
    move_cost = _move_cost_factory(heightmap, config.planner.max_slope_deg)
    path, cost = a_star(start, goal, neighbors, heuristic, move_cost, max_iter=max_iter)
    return PlanResult(path=path, cost=cost)
