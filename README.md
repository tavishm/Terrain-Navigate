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
  __main__.py             # demo entrypoint
  data/loader.py          # synthetic terrain (for now)
  graph/grid.py           # neighbors + step distances
  cost/costs.py           # slope-based move cost + heuristic
  planner/a_star.py       # generic A*
  pathfinder/             # robotics pipeline configs + helpers
```

## Robotics path-finder compatibility

Code from the legacy `Robotics_Path_Finder` repository now ships inside the
`terrain_navigate.pathfinder` namespace. Use it to keep existing configuration
dataclasses while reusing the modern planner primitives.

```python
from pathlib import Path
from terrain_navigate.pathfinder import (
    PipelineConfig,
    PlannerTuning,
    TileSelection,
    plan_route,
)

config = PipelineConfig(
    workspace=Path("./workspace"),
    tiles=TileSelection(
        tile_ids=["USGS_LPC_UT_Southern_QL1_2018_12STF9497"],
        source_url="https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/",
    ),
    planner=PlannerTuning(heuristic="octile", max_slope_deg=35.0, tie_breaker=1.05),
)

result = plan_route(config, start=(0, 0), goal=(150, 150))
assert result.success, "No feasible route"
print("cost:", result.cost)
```

Need to parse an original USGS directory listing? Feed the text blob into
`terrain_navigate.pathfinder.parse_listing` to obtain structured `TileInfo`
records with timestamps and byte sizes.

## Next
- Add LAZ reader (laspy/PDAL) and grid it.
- Add simple PNG plot of path over height.
- Tiny tests for neighbors + admissibility.
