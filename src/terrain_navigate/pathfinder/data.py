"""Utility helpers for dealing with LiDAR tile listings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List

LISTING_DATETIME_FORMAT = "%d-%b-%Y %H:%M"


@dataclass(frozen=True)
class TileInfo:
    """Minimal metadata about a LiDAR tile entry."""

    file_name: str
    modified_at: datetime
    size_bytes: int


def parse_listing(listing: str) -> List[TileInfo]:
    """Parse a plain-text directory listing into ``TileInfo`` objects."""

    entries: List[TileInfo] = []
    for raw_line in listing.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Parent Directory"):
            continue

        if ">" in line:
            name_part, _, rest = line.partition(">")
        else:
            tokens = line.split()
            if len(tokens) < 4:
                continue
            name_part, rest = tokens[0], " ".join(tokens[1:])

        tokens = rest.split()
        if len(tokens) < 3:
            continue
        date_str, time_str = tokens[0], tokens[1]
        size_str = tokens[-1]
        try:
            modified = datetime.strptime(f"{date_str} {time_str}", LISTING_DATETIME_FORMAT)
            size = int(size_str)
        except ValueError:
            continue

        file_name = name_part.rstrip(".").strip()
        entries.append(TileInfo(file_name=file_name, modified_at=modified, size_bytes=size))
    return entries
