import sys
from pathlib import Path

# Add the project root to sys.path to allow imports from src
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.terrain_navigate.resources import st_o
from src.terrain_navigate.loader import parse_file_list, generate_links, fetch_all_xmls, find_candidates

def main():
    print("Parsing file list...")
    ids = parse_file_list(st_o)
    print(f"Found {len(ids)} IDs")
    
    print("Generating links...")
    links = generate_links(ids)
    # print(links) # Optional: print links
    
    print("Fetching XML metadata...")
    xml_data = fetch_all_xmls(links)
    print(f"Successfully loaded {len(xml_data)} XML files")
    
    # Define bounds (MDRS area)
    ex_coords = {
        "NW": (38.433521, -110.826259),
        "NE": (38.433521, -110.757541),
        "SE": (38.379469, -110.757566),
        "SW": (38.379469, -110.826234)
    }
    
    print("Filtering candidates...")
    candidate_indices = find_candidates(xml_data, ex_coords)
    
    print(f"Found {len(candidate_indices)} candidates:")
    for idx in candidate_indices:
        print(f"Candidate index: {idx}")
        # In a real scenario, we would download the LAS files here
        # download_las(links[idx].replace("meta.xml", ".las"))

if __name__ == "__main__":
    main()
