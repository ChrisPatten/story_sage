import yaml
import json
import glob
import re
import pickle
import os
from story_sage.utils.entity_extractor import StorySageEntityExtractor

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

api_key = config['OPENAI_API_KEY']

TARGET_SERIES_ID = 3  # The series_id of the target series
TARGET_BOOK_NUMBERS = [ 1 ]  # List of book numbers to process

# Load series.yml to create a mapping from series_metadata_name to series_id
with open('series.yml', 'r') as file:
    series_list = yaml.safe_load(file)

# Get the target series information
target_series_info = next(series for series in series_list if series['series_id'] == TARGET_SERIES_ID)
series_metadata_name = target_series_info['series_metadata_name']

print(f'Extracting entities for series: {series_metadata_name}')

path_to_chunks = f'./chunks/{series_metadata_name}/semantic_chunks'
chunks = {}

# Load chunks for all target books
for filepath in glob.glob(f'{path_to_chunks}/*.pkl') + glob.glob(f'{path_to_chunks}/*.json'):
    match = re.match(r'(\d+)_(\d+)\.(pkl|json)', os.path.basename(filepath))
    if match:
        book_number, chapter_number, file_ext = match.groups()
        book_number, chapter_number = map(int, [book_number, chapter_number])
        if book_number in TARGET_BOOK_NUMBERS:
            with open(filepath, 'rb' if file_ext == 'pkl' else 'r') as f:
                if book_number not in chunks:
                    chunks[book_number] = {}
                if file_ext == 'pkl':
                    chunks[book_number][chapter_number] = pickle.load(f)
                else:
                    chunks[book_number][chapter_number] = json.load(f)

target_file_path = f'./entities/{series_metadata_name}'
if not os.path.exists(target_file_path):
    os.makedirs(target_file_path)

# Initialize the entity extractor
extractor = StorySageEntityExtractor(api_key=api_key)

# Iterate over each target book number
for book_number in TARGET_BOOK_NUMBERS:
    target_book_info = next(book for book in target_series_info['books'] if book['number_in_series'] == book_number)
    book_metadata_name = target_book_info['book_metadata_name']
    print(f'Processing book {book_number}: {book_metadata_name}')

    if book_number not in chunks:
        print(f'No chunks found for book {book_number}')
        continue

    target_filename = f'{target_file_path}/{book_metadata_name}.json'

    # Extract entities from the book chunks
    entities_dict_list, summaries_list = extractor.extract(chunks[book_number])

    # Process the extracted entities
    processed_entities_list = extractor.process_entities(entities_dict_list)

    # Save the processed entities to a file
    with open(target_filename, 'w') as json_file:
        json.dump(processed_entities_list, json_file, indent=4)

    # Save the summary chunks to a file
    summary_chunks_path = f'./chunks/{series_metadata_name}/summary_chunks'
    if not os.path.exists(summary_chunks_path):
        os.makedirs(summary_chunks_path)
    with open(f'{summary_chunks_path}/{book_number}.json', 'w') as json_file:
        json.dump(summaries_list, json_file, indent=4)

    print(f'Saved entities to {target_filename}')