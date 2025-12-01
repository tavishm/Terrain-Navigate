# MDRS Pathfinding Examples

This directory contains a complete workflow for performing pathfinding on MDRS (Mars Desert Research Station) terrain data using the `terrain_navigate` library.

## Workflow

1.  **Download Data**: Run `python 01_download_data.py` to fetch the file list and identify relevant LAS files for the MDRS region.
    *   *Note*: This script currently generates a list of URLs (`data/candidates.txt`). You would typically download these files using `wget` or `curl`.
2.  **Process Data**: Run `python 02_process_data.py` to process the raw data into a grid format (`data/processed_map.npy`).
    *   *Note*: For this demo, this script generates a synthetic terrain map to ensure it runs out-of-the-box without large downloads.
3.  **Run Pathfinding**: Open `03_pathfinding_demo.ipynb` in Jupyter to visualize the map, configure the planner, and generate a path.

## Files

- `01_download_data.py`: Robust script to fetch USGS metadata and filter for MDRS coordinates.
- `02_process_data.py`: Script to convert point clouds to elevation grids (simulated here).
- `03_pathfinding_demo.ipynb`: Interactive notebook for path planning and visualization.
- `data/`: Directory where downloaded and processed data is stored (ignored by git).

## Requirements

Ensure you have installed the package:
```bash
pip install -e ../../
```
