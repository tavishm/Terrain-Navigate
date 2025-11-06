# Terrain-Navigate

Compact terrain path planning prototype. Ships a working A* demo on synthetic terrain; later swap in real LAZ/point-cloud data.

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m terrain_navigate --size 60 --k 1.0 --p 2.0
```

- `--size` sets the grid size (NxN).
- `--k`, `--p` control slope penalty strength.

## Layout

```
src/terrain_navigate/
  __main__.py          # demo entrypoint
  data/loader.py       # synthetic terrain (for now)
  graph/grid.py        # neighbors + step distances
  cost/costs.py        # slope-based move cost + heuristic
  planner/a_star.py    # generic A*
```

## Next
- Add LAZ reader (laspy/PDAL) and grid it.
- Add simple PNG plot of path over height.
- Tiny tests for neighbors + admissibility.
