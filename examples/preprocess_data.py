import sys
import numpy as np
from pathlib import Path
import warnings

# Try to import laspy
try:
    import laspy
except ImportError:
    laspy = None

# Ensure src is in python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def process_las_files(data_dir: Path, resolution: float = 1.0) -> np.ndarray:
    """
    Reads .las/.laz files from data_dir and rasterizes them into a grid (heightmap).
    """
    las_files = list(data_dir.glob("*.las")) + list(data_dir.glob("*.laz"))
    
    if not las_files:
        return None

    print(f"Found {len(las_files)} point cloud files.")
    
    # 1. Determine bounds
    print("Determining bounds...")
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    
    for f in las_files:
        with laspy.open(f) as fh:
            header = fh.header
            min_x = min(min_x, header.x_min)
            min_y = min(min_y, header.y_min)
            max_x = max(max_x, header.x_max)
            max_y = max(max_y, header.y_max)
            
    print(f"Bounds: ({min_x:.2f}, {min_y:.2f}) to ({max_x:.2f}, {max_y:.2f})")
    
    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))
    
    print(f"Grid size: {width} x {height}")
    
    # 2. Create Grid
    grid = np.full((height, width), -np.inf, dtype=np.float32)
    
    # 3. Rasterize
    print("Rasterizing points...")
    for f in las_files:
        print(f"Processing {f.name}...")
        las = laspy.read(f)
        
        ix = ((las.x - min_x) / resolution).astype(int)
        iy = ((las.y - min_y) / resolution).astype(int)
        
        ix = np.clip(ix, 0, width - 1)
        iy = np.clip(iy, 0, height - 1)
        
        flat_indices = iy * width + ix
        grid_flat = grid.ravel()
        np.maximum.at(grid_flat, flat_indices, las.z)
        grid = grid_flat.reshape((height, width))

    # Fill holes
    valid_mask = grid > -10000
    if np.any(valid_mask):
        min_val = grid[valid_mask].min()
        grid[~valid_mask] = min_val
    else:
        grid[:] = 0

    return grid

def main():
    print("=== Data Processor (Voxelization/Grid Generation) ===")
    
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    output_file = data_dir / "processed_map.npy"
    
    if output_file.exists():
        print(f"Processed map already exists at: {output_file}")
        print("Delete it to regenerate.")
        # return 

    if laspy:
        grid = process_las_files(data_dir)
        if grid is None:
            print("Error: No .las/.laz files found in 'data/' directory.")
            print("Please run 'download_data.py' and download the files.")
            return
    else:
        print("Error: `laspy` not installed. Please install it with: pip install laspy[lazrs]")
        return
    
    print(f"Map shape: {grid.shape}")
    print(f"Elevation range: {grid.min():.2f}m to {grid.max():.2f}m")
    
    np.save(output_file, grid)
    print(f"Saved processed map to: {output_file}")

if __name__ == "__main__":
    main()
