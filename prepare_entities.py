"""
prepare_entities.py

This script processes entity data for a story series using the StorySageEntityProcessor class.

It performs the following tasks:
- Reads entity JSON files for specified series from the './entities/<series_metadata_name>/' directory.
- Collects and cleans all 'people' and other entity types from each file.
- Merges these lists across all books in the series, ensuring uniqueness.
- Converts all entity names to lowercase and removes any non-alphabetic characters except spaces.
- Groups similar names using NLP techniques and removes duplicates.
- Generates a consolidated 'entities.json' file with unique IDs mapping to entity names.

Example Usage:
    $ python prepare_entities.py

Example Results:
    - Processes entity files located in './entities/harry_potter/*.json'.
    - Aggregates entities for the series IDs listed in SERIES_IDS_TO_PROCESS.
    - Outputs a consolidated 'entities.json' file with processed entities.

Notes:
    - Ensure that 'series.yml' contains accurate series information.
    - The 'StorySageEntityProcessor' class must be correctly imported from 'story_sage.utils'.
    - An OpenAI API key is required for NLP tasks; it should be configured in 'config.yml'.
"""

from story_sage.utils import StorySageEntityProcessor
import glob
import json
import os
import yaml

# Load OpenAI API key from 'config.yml'
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Initialize the entity processor with the API key
processor = StorySageEntityProcessor(api_key=config['OPENAI_API_KEY'])

# List of series IDs to process
SERIES_IDS_TO_PROCESS = ['2']  # Modify this list to include the series IDs you want to process

# Dictionary to hold all entities for the series
all_entities_dict = {'series': {}}

# Load series information from 'series.yml'
with open('series.yml', 'r') as f:
    series_list = yaml.safe_load(f)

# Iterate over all entity JSON files in the specified directory
for entity_file in glob.glob('./entities/harry_potter/*.json'):
    # Each file is a book with chapters containing extracted entities
    with open(entity_file, 'r') as f:
        all_chapter_entities = json.load(f)
        # Extract the series metadata name from the file path
        series_metadata_name = os.path.basename(os.path.dirname(entity_file))
        if '_' not in series_metadata_name:
            raise ValueError(
                f"Invalid series_metadata_name '{series_metadata_name}' in file '{entity_file}'. It must include an underscore."
            )
        # Find the corresponding series information from the series list
        series_info_obj = next(
            series for series in series_list if series['series_metadata_name'] == series_metadata_name
        )
        series_id_str = str(series_info_obj['series_id'])
        # Skip series that are not in the list to process
        if series_id_str not in SERIES_IDS_TO_PROCESS:
            continue

        """
        all_chapter_entities is in this format:
        [
            [
                {
                    "people": ["Harry Potter", "Ron Weasley"],
                    "places": ["Hogwarts"],
                    ...
                },
                {
                    "completion_tokens": 192,
                    ...
                }
            ],
            ...
        ]
        """
        # Get just the entities from the chapters (skip the tokens info)
        book_entities = [chapter[0] for chapter in all_chapter_entities]

        # Process entities for the entire book
        new_book_entities = {'people': set(), 'entities': set()}
        # Add entities from each chapter into a master list
        for chapter_entities in book_entities:
            # Clean and add people entities
            cleaned_people = processor.get_cleaned_entities_list(chapter_entities.get('people', []))
            new_book_entities['people'].update(cleaned_people)
            # Clean and add other entities
            for entity_type in ['places', 'groups', 'animals', 'objects']:
                entities = chapter_entities.get(entity_type, [])
                cleaned_entities = processor.get_cleaned_entities_list(entities)
                new_book_entities['entities'].update(cleaned_entities)

        # Initialize the series entry if it doesn't exist
        if series_id_str not in all_entities_dict['series']:
            # Create a new entry for the series with the current book's entities
            all_entities_dict['series'][series_id_str] = {
                'series_metadata_name': series_metadata_name,
                'series_id': series_info_obj['series_id'],
                'series_name': series_info_obj['series_name'],
                'series_entities': {
                    'people': new_book_entities['people'],
                    'entities': new_book_entities['entities']
                }
            }
        else:
            # Merge the book entities into the existing series entities
            try:
                all_entities_dict['series'][series_id_str]['series_entities'] = processor.zip_entities(
                    all_entities_dict['series'][series_id_str]['series_entities'],
                    new_book_entities
                )
            except Exception as e:
                raise e

# Merge new entities into the existing 'entities.json' file
if os.path.exists('entities.json'):
    with open('entities.json', 'r') as json_file:
        existing_data = json.load(json_file)
else:
    existing_data = {'series': {}}

for series_id, series_info in existing_data['series'].items():
    if series_id in all_entities_dict['series']:
        # Merge the existing entities with the new entities for the series
        all_entities_dict['series'][series_id] = processor.zip_entities(
            series_info,
            all_entities_dict['series'][series_id]
        )

# Get only unique people and entities for the series
for series_id, series_info in all_entities_dict['series'].items():
    if 'people' not in series_info['series_entities']:
        series_info['series_entities']['people'] = []
    if 'entities' not in series_info['series_entities']:
        series_info['series_entities']['entities'] = []
    # Ensure uniqueness by converting to lists (from sets)
    series_info['series_entities']['people'] = processor.get_unique_entities(
        series_info['series_entities']['people']
    )
    series_info['series_entities']['entities'] = processor.get_unique_entities(
        series_info['series_entities']['entities']
    )

# Make sure no entities already exist in people
for series_id, series_info in all_entities_dict['series'].items():
    # Remove any entities that are already classified under people
    series_info['series_entities']['entities'] = [
        entity for entity in series_info['series_entities']['entities']
        if entity not in series_info['series_entities']['people']
    ]

# Group similar names and remove duplicates for people and entities
for series_id, series_info in all_entities_dict['series'].items():
    # Group similar people names using NLP
    grouped_people = processor.group_similar_names(series_info['series_entities']['people'])
    # Remove duplicates across groups
    series_info['series_entities']['people'] = processor.remove_duplicate_elements(grouped_people)

    # Group similar entity names using NLP
    grouped_entities = processor.group_similar_names(series_info['series_entities']['entities'])
    # Remove duplicates across groups
    series_info['series_entities']['entities'] = processor.remove_duplicate_elements(grouped_entities)

# Create result dictionaries mapping IDs to names
for series_id, series_info in all_entities_dict['series'].items():
    people = series_info['series_entities']['people']
    entities = series_info['series_entities']['entities']
    # Optionally, print the number of people and entities processed
    print(f"Series ID: {series_id}, People: {len(people)}, Entities: {len(entities)}")
    # Generate the final result dictionary for the series
    series_entities = processor.create_result_dict(people, entities, series_id)
    series_info['series_entities'] = series_entities

# Write the updated data back to 'entities.json'
with open('entities.json', 'w') as json_file:
    json.dump(all_entities_dict, json_file, indent=4)

# Example usage:
# To run the script, execute the following command in the terminal:
# $ python prepare_entities.py

# Example result:
# The script processes entity files, groups similar names, and updates 'entities.json' with the consolidated data.

"""
Example Output:

Series ID: 2, People: 85, Entities: 42
[['harry', 'harry potter'], ['ron', 'ronald weasley', 'ron weasley'], ...]
[['hogwarts', 'hogwarts school', 'hogwarts school of witchcraft and wizardry'], ...]
"""