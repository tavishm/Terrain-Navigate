import json
import copy

def create_modular_notebook():
    with open('A_star_v3.ipynb', 'r') as f:
        nb = json.load(f)
    
    new_nb = copy.deepcopy(nb)
    
    # Find the cell with the A* implementation (Cell index 1, execution_count 2)
    # It starts with "# A* pathfinding over the gridded elevation surface"
    
    target_cell_idx = -1
    for i, cell in enumerate(new_nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "# A* pathfinding over the gridded elevation surface" in source:
                target_cell_idx = i
                break
    
    if target_cell_idx != -1:
        # Replace the cell content
        new_source = [
            "# Modular A* pathfinding using src.pathfinder package\n",
            "import sys\n",
            "from pathlib import Path\n",
            "# Add current directory to sys.path to ensure src is found\n",
            "if str(Path.cwd()) not in sys.path:\n",
            "    sys.path.append(str(Path.cwd()))\n",
            "\n",
            "from src.pathfinder.algorithms import AStar\n",
            "from src.pathfinder.environments import GridEnvironment\n",
            "from src.pathfinder.costs import PowerCost\n",
            "\n",
            "radius = 1.0\n",
            "n = 2400\n",
            "alpha = 1e-5\n",
            "\n",
            "resolution = 0.5  # meters between grid cells\n",
            "global_x_min = 515000.0\n",
            "global_y_min = 4247000.0\n",
            "\n",
            "num_x, num_y = Z.shape\n",
            "global_x_max = global_x_min + (num_x - 1) * resolution\n",
            "global_y_max = global_y_min + (num_y - 1) * resolution\n",
            "\n",
            "# Initialize Environment\n",
            "env = GridEnvironment(\n",
            "    Z=Z,\n",
            "    global_x_min=global_x_min,\n",
            "    global_x_max=global_x_max,\n",
            "    global_y_min=global_y_min,\n",
            "    global_y_max=global_y_max,\n",
            "    resolution=resolution\n",
            ")\n",
            "\n",
            "# Initialize Cost Function\n",
            "cost_fn = PowerCost(n=n, alpha=alpha)\n",
            "\n",
            "# Initialize PathFinder\n",
            "astar_solver = AStar()\n",
            "\n",
            "# Wrapper function to maintain compatibility with existing notebook calls if needed\n",
            "# But better to update the calls. Let's see where 'astar' is called.\n",
            "# It seems 'astar' is called later. We can define a wrapper or update the call.\n",
            "# Let's define a wrapper 'astar' function that matches the old signature for compatibility\n",
            "def astar(start, goal, node_radius):\n",
            "    path, cost = astar_solver.find_path(start, goal, env, cost_fn, node_radius)\n",
            "    if path is None:\n",
            "        raise ValueError(\"Goal not reachable with current parameters\")\n",
            "    return path, cost\n",
            "\n",
            "def nearest_point(point, search_radius):\n",
            "    return env.get_nearest_node(point, search_radius)\n"
        ]
        new_nb['cells'][target_cell_idx]['source'] = new_source
        new_nb['cells'][target_cell_idx]['execution_count'] = None
        new_nb['cells'][target_cell_idx]['outputs'] = []

    with open('A_star_v3_modular.ipynb', 'w') as f:
        json.dump(new_nb, f, indent=1)
    
    print("Created A_star_v3_modular.ipynb")

if __name__ == "__main__":
    create_modular_notebook()
