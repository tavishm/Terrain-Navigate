import sys
import numpy as np
from pathlib import Path

# Ensure src is in python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def main():
    print("=== Data Processor (Voxelization/Grid Generation) ===")
    
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    output_file = data_dir / "processed_map.npy"
    
    if output_file.exists():
        print(f"Processed map already exists at: {output_file}")
        print("Delete it to regenerate.")
        return

    print("Generating synthetic terrain data (simulating LAS processing)...")
    # In a real scenario, this would read the LAS files downloaded in step 1
    # and rasterize them into a grid.
    
    # Create a 100x100 grid representing a 100m x 100m area
    # Z represents elevation
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(x, y)
    
    # Generate some terrain features (hills)
    Z = np.sin(X/10) * np.cos(Y/10) * 5 + 10
    
    # Add an obstacle (crater/rock)
    Z[40:60, 40:60] += 20.0 
    
    print(f"Map shape: {Z.shape}")
    print(f"Elevation range: {Z.min():.2f}m to {Z.max():.2f}m")
    
    np.save(output_file, Z)
    print(f"Saved processed map to: {output_file}")

if __name__ == "__main__":
    main()
