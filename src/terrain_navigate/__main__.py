from __future__ import annotations
import argparse
import numpy as np
from .data.loader import load_synthetic_terrain
from .graph.grid import iter_neighbors
from .cost.costs import heuristic_euclidean, slope_cost
from .planner.a_star import a_star


def run_demo(size: int, k: float, p: float) -> None:
    shape = (size, size)
    H = load_synthetic_terrain(shape=shape)
    start = (0, 0)
    goal = (shape[0] - 1, shape[1] - 1)

    neighbors_fn = lambda c: iter_neighbors(c, shape, connectivity=8)
    h_fn = heuristic_euclidean(goal)
    move_cost_fn = slope_cost(H, k=k, p=p)

    path, cost = a_star(start, goal, neighbors_fn, h_fn, move_cost_fn)

    print(f"Terrain shape: {H.shape}")
    print(f"Path length: {len(path)}  Total cost: {cost:.2f}")
    # Simple ASCII visualization (small sizes only)
    if size <= 40 and path:
        grid = np.full(shape, fill_value='.', dtype=str)
        for cell in path:
            grid[cell] = '#'
        grid[start] = 'S'
        grid[goal] = 'G'
        ascii_map = '\n'.join(''.join(row) for row in grid)
        print(ascii_map)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo A* terrain planner")
    parser.add_argument('--size', type=int, default=60, help='Grid size (NxN)')
    parser.add_argument('--k', type=float, default=1.0, help='Slope penalty factor')
    parser.add_argument('--p', type=float, default=2.0, help='Slope penalty exponent')
    args = parser.parse_args()
    run_demo(args.size, args.k, args.p)


if __name__ == '__main__':  # pragma: no cover
    main()
