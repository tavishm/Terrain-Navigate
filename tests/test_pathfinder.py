from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from terrain_navigate.pathfinder import (
    PipelineConfig,
    PlannerTuning,
    TileSelection,
    parse_listing,
    plan_route,
)


def test_pipeline_cache_dir(tmp_path):
    config = PipelineConfig(
        workspace=tmp_path,
        tiles=TileSelection(tile_ids=["tile-A"], source_url="https://example.com"),
    )
    assert config.cache_dir == tmp_path / "cache"


def test_parse_listing_extracts_tile_metadata():
    listing = """
    USGS_TILE_0001.laz> 03-Jun-2019 17:08            16158635
    Parent Directory
    USGS_TILE_0002.laz> 04-Jun-2019 09:10            123
    """
    entries = parse_listing(listing)
    assert len(entries) == 2
    assert entries[0].file_name == "USGS_TILE_0001.laz"
    assert entries[0].size_bytes == 16158635
    assert entries[0].modified_at == datetime(2019, 6, 3, 17, 8)


def test_plan_route_succeeds_on_flat_grid(tmp_path):
    config = PipelineConfig(
        workspace=tmp_path / "ws",
        tiles=TileSelection(tile_ids=["tile-A"], source_url="https://example.com"),
        planner=PlannerTuning(heuristic="octile", max_slope_deg=45.0, tie_breaker=1.0),
    )
    terrain = np.zeros((20, 20), dtype=np.float32)
    start, goal = (0, 0), (19, 19)
    result = plan_route(config, start=start, goal=goal, terrain=terrain, connectivity=8)
    assert result.success
    assert result.path[0] == start
    assert result.path[-1] == goal
    assert result.cost > 0
