# MDRS Pathfinding Examples

This directory contains the workflow for real-world terrain pathfinding.

## Workflow

1.  **Download Data**: Run `python download_data.py`.
    *   Fetches metadata from USGS.
    *   Filters for the MDRS region.
    *   Generates a `data/candidates.txt` list of LAS file URLs.
    *   *Action*: Download the .las files listed in `candidates.txt` and place them in `data/`.
2.  **Process Data**: Run `python preprocess_data.py`.
    *   Reads `.las` files from `data/`.
    *   Voxelizes them into a 2D heightmap grid.
    *   Saves `data/processed_map.npy`.
3.  **Run Pathfinding**: Open the notebooks:
    *   `A_star_v1.ipynb`: Spatial Index (Point Cloud) demo.
    *   `A_star_v2.ipynb`: Grid + Euclidean Cost demo.
    *   `A_star_v3.ipynb`: Grid + Power Cost demo (Best for Rovers).
