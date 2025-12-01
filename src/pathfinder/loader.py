import requests
import xml.etree.ElementTree as ET
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

def parse_file_list(file_list_str: str) -> List[str]:
    """
    Parses the file list string to extract IDs.
    """
    ids = []
    for line in file_list_str.split("\n"):
        if not line.strip():
            continue
        # Logic from original script: replace prefix and split by suffix
        # USGS_LPC_UT_Southern_QL1_2018_12SXH3268_LAS_201..>
        clean_line = line.replace("USGS_LPC_UT_Southern_QL1_2018_", "").split("_LAS_201..>")[0]
        ids.append(clean_line)
    return ids

def generate_links(ids: List[str]) -> List[str]:
    """
    Generates download links from IDs.
    """
    base_url = "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/USGS_LPC_UT_Southern_QL1_2018_LAS_2019/metadata/USGS_LPC_UT_Southern_QL1_2018_"
    suffix = "_LAS_2019_meta.xml"
    return [f"{base_url}{idx}{suffix}" for idx in ids]

def fetch_xml(link: str, timeout: int = 10, retries: int = 3) -> Optional[ET.Element]:
    """
    Fetches and parses an XML file from a URL with retries.
    """
    for attempt in range(retries):
        try:
            response = requests.get(link, timeout=timeout)
            if response.status_code == 200:
                return ET.fromstring(response.content)
            else:
                print(f"Failed to retrieve {link} (Status: {response.status_code})")
        except requests.RequestException as e:
            print(f"Error processing {link} (Attempt {attempt + 1}/{retries}): {e}")
            
    return None

def fetch_all_xmls(links: List[str], max_workers: int = 6) -> List[ET.Element]:
    """
    Fetches multiple XMLs in parallel.
    """
    xml_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_xml, link): link for link in links}
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is not None:
                xml_data.append(result)
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(links)} files")
    return xml_data

def find_candidates(xml_data: List[ET.Element], bounds: Dict[str, Tuple[float, float]]) -> List[int]:
    """
    Finds candidate files that overlap with the given bounds.
    
    Args:
        xml_data: List of parsed XML elements.
        bounds: Dictionary with 'NE', 'NW', 'SE', 'SW' coordinates.
        
    Returns:
        List of indices of matching XML files.
    """
    candidates = []
    for i, xml in enumerate(xml_data):
        try:
            # Assuming the structure matches the original script
            # xml[0][4][0] -> bounding box? 
            # The original script used: xml_data[i][0][4][0][0].text for west
            # We need to be robust if structure varies, but sticking to original logic for now.
            
            # Note: The original script accessed xml_data[i] which was a list? 
            # In fetch_all_xmls we return a list of Elements.
            # Let's assume the structure is consistent.
            
            west = float(xml[0][4][0][0].text)
            east = float(xml[0][4][0][1].text)
            north = float(xml[0][4][0][2].text)
            south = float(xml[0][4][0][3].text)
            
            # Check overlap
            # (west < ex_coords["NE"][1] and east > ex_coords["NW"][1]) and (south < ex_coords["NW"][0] and north > ex_coords["SW"][0])
            # ex_coords["NE"][1] is max_lon (approx)
            # ex_coords["NW"][1] is min_lon
            # ex_coords["NW"][0] is max_lat
            # ex_coords["SW"][0] is min_lat
            
            # Bounds format from original:
            # "NW": (38.433521, -110.826259) -> (lat, lon)
            # "NE": (38.433521, -110.757541)
            # "SE": (38.379469, -110.757566)
            # "SW": (38.379469, -110.826234)
            
            max_lon = bounds["NE"][1]
            min_lon = bounds["NW"][1]
            max_lat = bounds["NW"][0]
            min_lat = bounds["SW"][0]
            
            if (west < max_lon and east > min_lon) and (south < max_lat and north > min_lat):
                candidates.append(i)
                
        except (IndexError, AttributeError, ValueError) as e:
            print(f"Error parsing XML at index {i}: {e}")
            continue
            
    return candidates
