# Terrain Navigate Examples

This directory contains examples demonstrating how to use the `terrain_navigate` library.

## Workflow

1.  **Download Data**: Run `python download_data.py` to fetch metadata and identify relevant LAS files for the MDRS region.
2.  **Process Data**: Run `python preprocess_data.py` to process the raw data into a grid format (`data/processed_map.npy`).
3.  **Run Pathfinding**: Open the notebooks to visualize the map and generate paths.

## Notebooks

*   **`A_star_v1.ipynb`**: Demonstrates the **Spatial Index** environment (KDTree-based) for point clouds.
*   **`A_star_v2.ipynb`**: Demonstrates the **Grid** environment with **Euclidean Cost** (standard 3D distance).
*   **`A_star_v3.ipynb`**: Demonstrates the **Grid** environment with **Power Cost** (MDRS-style slope penalty).

## Scripts

- `download_data.py`: Robust script to fetch USGS metadata and filter for MDRS coordinates.
- `preprocess_data.py`: Script to convert point clouds to elevation grids (simulated here).
- `data/`: Directory where downloaded and processed data is stored (ignored by git).
