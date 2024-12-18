"""
This script processes entity data for a story series using the StorySageEntityProcessor class.

It reads entity files for specified series, aggregates and processes the entities,
groups similar names, removes duplicates, and generates a consolidated `entities.json` file.

The script utilizes the `StorySageEntityProcessor` class to handle entity processing tasks,
such as zipping entities, collecting unique values, grouping similar names, and creating
result dictionaries with unique IDs.
"""

from story_sage.utils import StorySageEntityProcessor
import glob
import json
import os
import yaml

# Load config.yml to get the OpenAI API key
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Initialize the entity processor (API key can be specified if required)
processor = StorySageEntityProcessor(api_key=config['OPENAI_API_KEY'])

# List of series IDs to process
SERIES_IDS_TO_PROCESS = ['2']

# Dictionary to hold all entities for the series
all_entities_dict = {'series': {}}

# Load series information from a YAML file
with open('series.yml', 'r') as f:
    series_list = yaml.safe_load(f)

# Iterate over all entity JSON files in the specified directory
for entity_file in glob.glob('./entities/*/*.json'):
    with open(entity_file, 'r') as f:
        entities = json.load(f)
        # Extract the series metadata name from the file path
        series_metadata_name = os.path.basename(os.path.dirname(entity_file))
        if '_' not in series_metadata_name:
            raise ValueError(f"Invalid series_metadata_name '{series_metadata_name}' in file '{entity_file}'. It must include an underscore.")
        # Find the corresponding series information from the series list
        series_info_obj = next(series for series in series_list if series['series_metadata_name'] == series_metadata_name)
        series_length = len(series_info_obj['books'])
        series_id_str = str(series_info_obj['series_id'])
        # Skip series that are not in the list to process
        if series_id_str not in SERIES_IDS_TO_PROCESS:
            continue
        # Initialize the series entry if it doesn't exist
        if series_id_str not in all_entities_dict['series']:
            all_entities_dict['series'][series_id_str] = {
                'series_metadata_name': series_metadata_name,
                'series_id': series_info_obj['series_id'],
                'series_name': series_info_obj['series_name'],
                'series_entities': {'all_entities': {}}
            }
        # Retrieve existing entities for the series
        series_entities = all_entities_dict['series'][series_id_str]['series_entities']['all_entities']
        # Zip new entities into the series entities
        all_entities = processor.zip_entities(series_entities, entities)
        all_entities_dict['series'][series_id_str]['series_entities']['all_entities'] = all_entities

# Process each series in the aggregated entities dictionary
for series_id, series_info in all_entities_dict['series'].items():
    # Collect unique people and entity names
    series_entities = series_info['series_entities']['all_entities']
    people, entities = processor.collect_unique_values(series_entities)
    series_info['series_entities']['people'] = people
    series_info['series_entities']['entities'] = entities
    # Remove the 'all_entities' key as it's no longer needed
    del series_info['series_entities']['all_entities']

# Group similar names and remove duplicates for people and entities
for series_id, series_info in all_entities_dict['series'].items():
    # Group similar people names using NLP
    grouped_people = processor.group_similar_names(series_info['series_entities']['people'])
    series_info['series_entities']['people'] = processor.remove_duplicate_elements(grouped_people)
    
    # Group similar entity names using NLP
    grouped_entities = processor.group_similar_names(series_info['series_entities']['entities'])
    series_info['series_entities']['entities'] = processor.remove_duplicate_elements(grouped_entities)

# Create result dictionaries mapping IDs to names
for series_id, series_info in all_entities_dict['series'].items():
    people = series_info['series_entities']['people']
    entities = series_info['series_entities']['entities']
    # Generate the final result dictionary for the series
    series_entities = processor.create_result_dict(people, entities, series_id)
    series_info['series_entities'] = series_entities

# Merge new entities into the existing 'entities.json' file
if os.path.exists('entities.json'):
    with open('entities.json', 'r') as json_file:
        existing_data = json.load(json_file)
else:
    existing_data = {'series': {}}

# Update existing data with new series information
for series_id, series_info in all_entities_dict['series'].items():
    if series_id in existing_data['series']:
        # Extend existing lists with new data
        existing_data['series'][series_id]['series_entities']['people'].extend(series_info['series_entities']['people'])
        existing_data['series'][series_id]['series_entities']['entities'].extend(series_info['series_entities']['entities'])
    else:
        # Add new series information
        existing_data['series'][series_id] = series_info

# Write the updated data back to 'entities.json'
with open('entities.json', 'w') as json_file:
    json.dump(existing_data, json_file, indent=4)

"""
Example Usage:

To run the script, execute the following command in the terminal:
$ python prepare_entities.py

Example Results:

- The script processes entity files located in './entities/*/*.json'.
- It aggregates entities for the specified series IDs listed in SERIES_IDS_TO_PROCESS.
- Similar names are grouped, and duplicates are removed.
- A consolidated 'entities.json' file is created or updated with the processed data.

Note:

- Ensure that the 'series.yml' file exists and contains accurate series information.
- The 'StorySageEntityProcessor' class must be correctly imported from 'story_sage.utils'.
- An OpenAI API key may be required for NLP tasks; configure it appropriately.
"""