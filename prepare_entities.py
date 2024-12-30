import yaml
from story_sage.utils.embedding import load_chunk_from_disk
import glob
from story_sage.utils.local_entity_extractor import StorySageEntityExtractor
from story_sage.story_sage_entity import StorySageEntityCollection
import os

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

api_key = config['OPENAI_API_KEY']
chroma_path = config['CHROMA_PATH']
chroma_collection = config['CHROMA_COLLECTION']
series_path = config['SERIES_PATH']

TARGET_SERIES_ID = 3 # wheel of time

# Load series.yml to create a mapping from series_metadata_name to series_id
with open(series_path, 'r') as file:
    series_list = yaml.safe_load(file)

target_series_info = next(series for series in series_list if series['series_id'] == TARGET_SERIES_ID)

series_metadata_name = target_series_info['series_metadata_name']

chunks = []

chunks_path = f'./chunks/{series_metadata_name}/semantic_chunks/*.pkl'

for chunk_path in glob.glob(chunks_path):
    chunks.extend(load_chunk_from_disk(chunk_path))

if os.path.exists(f'./entities/{series_metadata_name}/entities.json'):
    with open(f'./entities/{series_metadata_name}/entities.json', 'r') as file:
        entity_collection = StorySageEntityCollection.from_json(file.read())
else:
    entity_collection = StorySageEntityCollection()

extractor = StorySageEntityExtractor(series = target_series_info, device='mps', existing_collection=entity_collection, similarity_threshold=0.5)

entity_collection: StorySageEntityCollection = extractor.get_grouped_entities(chunks, labels=['PERSON', 'ANIMAL', 'OBJECT', 'LOCATION'])

entity_json = entity_collection.to_json()
with open(f'./entities/{series_metadata_name}/entities.json', 'w') as file:
    file.write(entity_json)