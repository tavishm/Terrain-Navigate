import sys
import time
from pathlib import Path

# Ensure src is in python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.terrain_navigate.resources import st_o
from src.terrain_navigate.loader import parse_file_list, generate_links, fetch_all_xmls, find_candidates

def main():
    print("=== MDRS Data Downloader ===")
    
    # 1. Setup paths
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    print(f"Data directory: {data_dir}")

    # 2. Parse file list (from resources)
    print("\n[1/4] Parsing file list...")
    ids = parse_file_list(st_o)
    print(f"Found {len(ids)} total files in repository.")

    # 3. Generate download links
    print("\n[2/4] Generating metadata links...")
    links = generate_links(ids)
    
    # 4. Fetch XML metadata (threaded)
    print("\n[3/4] Fetching XML metadata (this may take a moment)...")
    start_time = time.time()
    xml_data = fetch_all_xmls(links)
    duration = time.time() - start_time
    print(f"Fetched {len(xml_data)} XML files in {duration:.2f}s")

    # 5. Filter for MDRS area
    print("\n[4/4] Filtering for MDRS region...")
    # MDRS Coordinates (Approximate)
    ex_coords = {
        "NW": (38.433521, -110.826259),
        "NE": (38.433521, -110.757541),
        "SE": (38.379469, -110.757566),
        "SW": (38.379469, -110.826234)
    }
    
    candidate_indices = find_candidates(xml_data, ex_coords)
    
    if not candidate_indices:
        print("No matching files found for MDRS coordinates.")
        return

    print(f"\nFound {len(candidate_indices)} candidate files covering MDRS:")
    
    # Save candidate info
    candidates_file = data_dir / "candidates.txt"
    with open(candidates_file, "w") as f:
        for idx in candidate_indices:
            file_id = ids[idx]
            url = links[idx].replace("meta.xml", ".las") # Construct LAS url
            print(f" - {file_id}")
            f.write(f"{file_id},{url}\n")
            
    print(f"\nCandidate list saved to: {candidates_file}")
    print("Note: Actual LAS file downloading is not implemented in this demo to save bandwidth.")
    print("You can use the URLs in 'candidates.txt' to download the point clouds.")

if __name__ == "__main__":
    main()
