import sys
import numpy as np
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pathfinder.algorithms import AStar
from src.pathfinder.environments import GridEnvironment
from src.pathfinder.costs import PowerCost, EuclideanCost

def test_grid_environment():
    print("Testing GridEnvironment...")
    # Create a simple 10x10 grid
    Z = np.zeros((10, 10))
    # Add an obstacle (high cost)
    Z[4:6, 4:6] = 100.0
    
    env = GridEnvironment(
        Z=Z,
        global_x_min=0.0,
        global_x_max=10.0,
        global_y_min=0.0,
        global_y_max=10.0,
        resolution=1.0
    )
    
    start = np.array([1.0, 1.0, 0.0])
    goal = np.array([8.0, 8.0, 0.0])
    
    # Test get_nearest_node
    nearest = env.get_nearest_node(start, radius=1.5)
    print(f"Nearest node to {start}: {nearest}")
    assert np.allclose(nearest[:2], [1.0, 1.0], atol=0.5) 
    
    # Test get_neighbors
    neighbors = env.get_neighbors(nearest, radius=1.5)
    print(f"Found {len(neighbors)} neighbors")
    assert len(neighbors) > 0

def test_astar_grid():
    print("\nTesting A* on Grid...")
    # Create a simple 10x10 grid
    Z = np.zeros((10, 10))
    # Add an obstacle
    Z[5, :] = 100.0
    Z[5, 2] = 0.0 # Gap
    
    env = GridEnvironment(
        Z=Z,
        global_x_min=0.0,
        global_x_max=10.0,
        global_y_min=0.0,
        global_y_max=10.0,
        resolution=1.0
    )
    
    cost_fn = PowerCost(n=2, alpha=1.0)
    astar = AStar()
    
    start_raw = np.array([1.5, 1.5, 0.0])
    goal_raw = np.array([8.5, 8.5, 0.0])
    
    start = env.get_nearest_node(start_raw, radius=1.5)
    goal = env.get_nearest_node(goal_raw, radius=1.5)
    
    print(f"Snapped start: {start}")
    print(f"Snapped goal: {goal}")
    
    path, cost = astar.find_path(start, goal, env, cost_fn, node_radius=1.5)
    
    if path is not None:
        print(f"Path found! Cost: {cost}")
        print(f"Path length: {len(path)}")
        # Check if path goes through the gap
        gap_used = False
        for p in path:
            if 4.0 < p[0] < 6.0 and 1.0 < p[1] < 3.0:
                gap_used = True
        print(f"Gap used: {gap_used}")
    else:
        print("No path found.")

if __name__ == "__main__":
    test_grid_environment()
    test_astar_grid()
