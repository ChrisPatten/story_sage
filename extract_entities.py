"""Entity Extraction Script for StorySage.

This script extracts named entities from story chunks using the StorySageEntityExtractor.
It processes text chunks from a specified series and updates a central entities.json file
with the extracted entities.

Example usage:
    $ python extract_entities.py
    
Example results:
    {
        "harry_potter": {
            "characters": ["Harry Potter", "Hermione Granger", ...],
            "locations": ["Hogwarts", "Diagon Alley", ...],
            "organizations": ["Gryffindor", "Ministry of Magic", ...]
        }
    }

Dependencies:
    - config.yml: Contains Chroma DB configuration
    - series.yml: Contains series metadata
    - ./entities/entities.json: Storage for extracted entities
    - ./chunks/{series}/semantic_chunks/*.json: Source text chunks
"""

import yaml
import json
import glob
import re
import os
from story_sage.utils.local_entity_extractor import StorySageEntityExtractor
from story_sage.utils.embedding import load_chunk_from_disk
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process text files into semantic chunks.')
parser.add_argument('--series_name', type=str, help='Name of the series to process')
parser.add_argument('--group_only', action='store_true', help='Only do grouping from intermediate extracted characters')
args = parser.parse_args()

series_metadata_name = args.series_name


# Load configuration settings
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

chroma_path = config['CHROMA_PATH']
chroma_collection = config['CHROMA_COLLECTION']
series_path = config['SERIES_PATH']
entities_path = config['ENTITIES_PATH']

# Load existing entities or create new if not exists
if not os.path.exists(entities_path):
    entities = {}
else:
    with open(entities_path, 'r') as file:
        entities = json.load(file)

# Load series metadata and map series names to IDs
with open(series_path, 'r') as file:
    series_list = yaml.safe_load(file)

# Find the target series information based on series_id
target_series_info = next(series for series in series_list if series['series_metadata_name'] == series_metadata_name)

def get_all_chunks(chunk_glob: str) -> dict:
    """Load and combine all text chunks from matching JSON files.
    
    Args:
        chunk_glob (str): Glob pattern to match chunk files (e.g., './chunks/*/semantic_chunks/*.json')
        
    Returns:
        list: Combined list of text chunks from all matching files
        
    Example:
        chunks = get_all_chunks('./chunks/harry_potter/semantic_chunks/*.json')
    """
    chunks = []
    for file in glob.glob(chunk_glob):
        chunks.extend(load_chunk_from_disk(file))

    return chunks


# Initialize the entity extractor with target series metadata
# 'mps' device specifies Metal Performance Shaders for Mac acceleration
extractor = StorySageEntityExtractor(series=target_series_info, device='mps')

if not args.group_only:
    print(f'Extracting entities for series: {series_metadata_name}')
    # Construct path to chunk files and load them
    path_to_chunks = f'./chunks/{series_metadata_name}/semantic_chunks/*.json'
    chunks = get_all_chunks(path_to_chunks)

    # Extract and group entities from all chunks
    grouped_entities = extractor.get_grouped_entities(chunks)
else:
    print(f'Grouping extracted entities for series: {series_metadata_name}')
    path_to_entities = f'all_strings_{series_metadata_name}.json'

    grouped_entities = extractor.group_intermediate_entities(path_to_intermediate = path_to_entities)

# Update the entities dictionary with new extractions
entities[target_series_info['series_metadata_name']] = grouped_entities.to_dict()

# Save updated entities back to JSON file
with open(entities_path, 'w') as file:
    file.write(json.dumps(entities, indent=4))