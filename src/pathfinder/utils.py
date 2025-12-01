import numpy as np
from pathlib import Path
from typing import Optional

_packed_dtype = np.dtype([("ix", "<i4"), ("iy", "<i4")])

def _pack_coords(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    data = np.empty(ix.shape[0], dtype=_packed_dtype)
    data["ix"] = ix
    data["iy"] = iy
    return data.view(np.int64)

def _pack_single(ix: int, iy: int) -> int:
    data = np.empty(1, dtype=_packed_dtype)
    data["ix"] = ix
    data["iy"] = iy
    return int(data.view(np.int64)[0])

def _build_lookup(cell_coords: np.ndarray) -> dict:
    packed = _pack_coords(cell_coords[:, 0], cell_coords[:, 1])
    return {int(key): int(i) for i, key in enumerate(packed)}

def build_spatial_index(points: np.ndarray, cell_size: float) -> dict:
    xy = np.floor(points[:, :2] / cell_size).astype(np.int32)
    order = np.lexsort((xy[:, 1], xy[:, 0]))
    order_dtype = np.uint32 if points.shape[0] < np.iinfo(np.uint32).max else np.int64
    order = order.astype(order_dtype, copy=False)
    sorted_xy = xy[order]
    if sorted_xy.size == 0:
        raise ValueError("No points provided for spatial index")
    diff = np.any(np.diff(sorted_xy, axis=0), axis=1)
    boundaries = np.flatnonzero(diff) + 1
    boundaries = np.concatenate(([0], boundaries, [sorted_xy.shape[0]]))
    cell_offsets = boundaries[:-1].astype(np.int64)
    cell_counts = np.diff(boundaries).astype(np.int32)
    cell_coords = sorted_xy[cell_offsets].astype(np.int32)
    lookup = _build_lookup(cell_coords)
    del xy, sorted_xy
    index = {
        "cell_size": float(cell_size),
        "order": order,
        "cell_coords": cell_coords,
        "cell_offsets": cell_offsets,
        "cell_counts": cell_counts,
        "lookup": lookup,
    }
    return index

def save_spatial_index(index: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cell_size=index["cell_size"],
        order=index["order"],
        cell_coords=index["cell_coords"],
        cell_offsets=index["cell_offsets"],
        cell_counts=index["cell_counts"],
    )

def load_spatial_index(path: Path) -> dict:
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    index = {
        "cell_size": float(data["cell_size"]),
        "order": data["order"],
        "cell_coords": data["cell_coords"],
        "cell_offsets": data["cell_offsets"],
        "cell_counts": data["cell_counts"],
    }
    index["lookup"] = _build_lookup(index["cell_coords"])
    return index

def get_or_build_index(points: np.ndarray, cell_size: float, path: Path) -> dict:
    path = Path(path)
    if path.exists():
        return load_spatial_index(path)
    index = build_spatial_index(points, cell_size)
    save_spatial_index(index, path)
    return index
