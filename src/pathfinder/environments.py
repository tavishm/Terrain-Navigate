import numpy as np
from typing import Optional
from .base import Environment
from .utils import _pack_single

class SpatialIndexEnvironment(Environment):
    """
    Environment implementation using a spatial index (v1 logic).
    """
    def __init__(self, global_map: np.ndarray, spatial_index: dict):
        self.global_map = global_map
        self.spatial_index = spatial_index
        self.cell_size = spatial_index["cell_size"]
        self.order = spatial_index["order"]
        self.offsets = spatial_index["cell_offsets"]
        self.counts = spatial_index["cell_counts"]
        self.lookup = spatial_index["lookup"]

    def get_neighbors(self, node: np.ndarray, radius: float) -> np.ndarray:
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        
        center = np.asarray(node[:2], dtype=np.float64)
        base = np.floor(center / self.cell_size).astype(np.int32)
        candidates = []
        radius_sq = float(radius) * float(radius)
        
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = _pack_single(int(base[0] + dx), int(base[1] + dy))
                bucket = self.lookup.get(key)
                if bucket is None:
                    continue
                start = self.offsets[bucket]
                end = start + self.counts[bucket]
                chunk = self.global_map[self.order[start:end]]
                diff = chunk[:, :2] - center
                mask = (diff[:, 0] ** 2 + diff[:, 1] ** 2) <= radius_sq
                if np.any(mask):
                    candidates.append(chunk[mask])
        
        if candidates:
            return np.vstack(candidates)
        return np.empty((0, self.global_map.shape[1]), dtype=self.global_map.dtype)

    def get_nearest_node(self, point: np.ndarray, radius: float) -> np.ndarray:
        if radius <= 0:
            raise ValueError("Radius must be positive.")

        # For nearest node, we can reuse get_neighbors to find candidates
        candidates = self.get_neighbors(point, radius)
        if candidates.size == 0:
            raise ValueError("No nearby points found; increase radius or choose a different point")
        
        # Ensure point has 3 dimensions for distance calculation if candidates are 3D
        if point.shape[0] < 3 and candidates.shape[1] >= 3:
             point = np.pad(point, (0, 3 - point.shape[0]), mode="constant")
             
        idx = int(np.argmin(np.linalg.norm(candidates[:, :3] - point[:3], axis=1)))
        return candidates[idx].copy()


class GridEnvironment(Environment):
    """
    Environment implementation using a grid (v2/v3 logic).
    """
    def __init__(
        self, 
        Z: np.ndarray, 
        global_x_min: float, 
        global_x_max: float, 
        global_y_min: float, 
        global_y_max: float, 
        resolution: float
    ):
        self.Z = Z
        self.global_x_min = global_x_min
        self.global_x_max = global_x_max
        self.global_y_min = global_y_min
        self.global_y_max = global_y_max
        self.resolution = resolution
        self.num_x = Z.shape[0]
        self.num_y = Z.shape[1]

    def get_neighbors(self, node: np.ndarray, radius: float) -> np.ndarray:
        if radius <= 0:
            raise ValueError("Radius must be positive.")

        point_xy = np.asarray(node[:2], dtype=np.float64)
        x_min = max(point_xy[0] - radius, self.global_x_min)
        x_max = min(point_xy[0] + radius, self.global_x_max)
        y_min = max(point_xy[1] - radius, self.global_y_min)
        y_max = min(point_xy[1] + radius, self.global_y_max)

        i_min = max(int(np.floor((x_min - self.global_x_min) / self.resolution)), 0)
        i_max = min(int(np.ceil((x_max - self.global_x_min) / self.resolution)), self.num_x - 1)
        j_min = max(int(np.floor((y_min - self.global_y_min) / self.resolution)), 0)
        j_max = min(int(np.ceil((y_max - self.global_y_min) / self.resolution)), self.num_y - 1)

        if i_min > i_max or j_min > j_max:
            return np.empty((0, 3), dtype=np.float64)

        x_idx = np.arange(i_min, i_max + 1)
        y_idx = np.arange(j_min, j_max + 1)
        x_coords = self.global_x_min + x_idx * self.resolution
        y_coords = self.global_y_min + y_idx * self.resolution

        x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="ij")
        z_values = self.Z[np.ix_(x_idx, y_idx)]

        radial_mask = ((x_grid - point_xy[0]) ** 2 + (y_grid - point_xy[1]) ** 2) <= radius ** 2
        if not np.any(radial_mask):
            return np.empty((0, 3), dtype=np.float64)

        valid_mask = radial_mask & np.isfinite(z_values)
        if not np.any(valid_mask):
            return np.empty((0, 3), dtype=np.float64)

        candidates = np.stack((x_grid[valid_mask], y_grid[valid_mask], z_values[valid_mask]), axis=-1)
        return candidates.astype(np.float64, copy=False)

    def get_nearest_node(self, point: np.ndarray, radius: float) -> np.ndarray:
        if radius <= 0:
            raise ValueError("Radius must be positive.")
            
        candidates = self.get_neighbors(point, radius)
        if candidates.size == 0:
             raise ValueError("No nearby points found; increase radius or choose a different point")
        
        if point.shape[0] < 3:
            point = np.pad(point, (0, 3 - point.shape[0]), mode="constant")
            
        idx = int(np.argmin(np.linalg.norm(candidates - point, axis=1)))
        return candidates[idx].copy()
