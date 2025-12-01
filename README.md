# Terrain Navigate

A modular Python library for 3D pathfinding on terrain maps, specifically designed for MDRS and URC challenges but generic enough for other applications.

## Features

- **Modular Architecture**: Easily extensible components for PathFinder, Environment, and CostFunction.
- **A* Algorithm**: Robust implementation of A* for 3D terrain.
- **Efficient Environments**: Supports both Spatial Index (v1) and Grid-based (v2/v3) environments.
- **Customizable Costs**: Includes Euclidean and Power-based cost functions.
- **Data Loading**: Tools to fetch and process USGS LAS data.

## Installation

```bash
pip install -e .
```

## Usage

See the `examples` directory for usage demonstrations.

### Basic Example

```python
from src.pathfinder import AStar, GridEnvironment, PowerCost

# Initialize environment and cost function
env = GridEnvironment(Z=Z, ...)
cost_fn = PowerCost(n=2400, alpha=1e-5)

# Find path
astar = AStar()
path, cost = astar.find_path(start, goal, env, cost_fn, node_radius=1.0)
```

## Structure

- `src/pathfinder`: Core package containing the library code.
- `examples`: Example scripts and notebooks (including MDRS specific examples).
- `tests`: Unit tests.

## Author

Tavish
