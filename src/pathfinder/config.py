"""Configuration dataclasses for the robotics path finder prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class TileSelection:
    """Describes a set of LiDAR tiles that define the area of interest."""

    tile_ids: Sequence[str]
    source_url: str


@dataclass(frozen=True)
class PlannerTuning:
    """Parameters that influence search behaviour and terrain penalties."""

    heuristic: str = "manhattan"
    max_slope_deg: float = 30.0
    tie_breaker: float = 1.0


@dataclass(frozen=True)
class PipelineConfig:
    """Aggregates configuration for acquisition, preprocessing, and planning."""

    workspace: Path
    tiles: TileSelection
    planner: PlannerTuning = field(default_factory=PlannerTuning)

    @property
    def cache_dir(self) -> Path:
        """Return the directory used for cached downloads."""

        return self.workspace / "cache"
