import yaml
import json
import glob
import re
import pickle
import os
from story_sage.utils import StorySageEntityExtractor

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

api_key = config['OPENAI_API_KEY']

TARGET_SERIES_ID = 2 # harry potter
TARGET_BOOK_NUMBER = 7

# Load series.yml to create a mapping from series_metadata_name to series_id
with open('series.yml', 'r') as file:
    series_list = yaml.safe_load(file)

target_series_info = next(series for series in series_list if series['series_id'] == TARGET_SERIES_ID)
target_book_info = next(book for book in target_series_info['books'] if book['number_in_series'] == TARGET_BOOK_NUMBER)

series_metadata_name = target_series_info['series_metadata_name']
book_metadata_name = target_book_info['book_metadata_name']

print(f'Extracting entities for {series_metadata_name} - {book_metadata_name}')

path_to_chunks = f'./chunks/{series_metadata_name}/semantic_chunks'
chunks = {}
for filepath in glob.glob(f'{path_to_chunks}/*.pkl'):
    match = re.match(r'(\d+)_(\d+)\.pkl', os.path.basename(filepath))
    if match:
        book_number, chapter_number = map(int, match.groups())
        with open(filepath, 'rb') as f:
            if book_number not in chunks:
                chunks[book_number] = {}
            chunks[book_number][chapter_number] = pickle.load(f)

target_file_path = f'./entities/{series_metadata_name}'
if not os.path.exists(target_file_path):
    os.makedirs(target_file_path)
target_filename = f'{target_file_path}/{book_metadata_name}.json'

# Extract entities from the chunks of the target book
extractor = StorySageEntityExtractor(api_key=api_key)

extracted_entities_dict = extractor.extract(chunks[TARGET_BOOK_NUMBER])
with open(target_filename, 'w') as json_file:
    json.dump(extracted_entities_dict, json_file, default=lambda o: o.__dict__, indent=4)