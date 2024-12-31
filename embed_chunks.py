"""Story Sage Document Embedding Module.

This module handles the embedding of document chunks into a vector store for the Story Sage
application. It processes story series documents, creates embeddings, and stores them in a
ChromaDB database along with associated metadata.

Example usage:
    ```python
    # Set up configuration
    series_metadata_name = 'wheel_of_time'
    
    # Load entities and series data
    with open('./entities/entities.json', 'r') as f:
        entity_json = json.load(f)
    
    # Initialize vector store and process documents
    chroma_client = chromadb.PersistentClient(path='./chroma_data')
    embedder = Embedder()
    vector_store = chroma_client.get_or_create_collection('story_sage')
    
    # Process and embed documents
    doc_collection = load_chunks_from_glob('./chunks/wheel_of_time/semantic_chunks/*.json')
    ```

Expected results:
    - Embedded documents stored in ChromaDB
    - Console output confirming series processing
"""

from story_sage.utils.embedding import Embedder, load_chunk_from_disk, embed_documents
from story_sage.story_sage_entity import *
from story_sage.data_classes.story_sage_series import StorySageSeries
import yaml
import glob
import chromadb
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process text files into semantic chunks.')
parser.add_argument('--series_name', type=str, help='Name of the series to process')
args = parser.parse_args()

series_metadata_name = args.series_name

# Load configuration settings
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

chroma_path = config['CHROMA_PATH']
chroma_collection = config['CHROMA_COLLECTION']
series_path = config['SERIES_PATH']
entities_path = config['ENTITIES_PATH']

# Load entity metadata from JSON file
with open(entities_path, 'r') as f:
    entity_json = json.load(f)
    entity_collection = {key: StorySageEntityCollection.from_dict(value) for key, value in entity_json.items()}

# Load series configuration from YAML
with open(series_path, 'r') as f:
    series_dict = yaml.safe_load(f.read())

# Initialize ChromaDB client and embedding function
chroma_client = chromadb.PersistentClient(path=chroma_path)
embedder = Embedder()
vector_store = chroma_client.get_or_create_collection('story_sage', embedding_function=embedder)

def load_chunks_from_glob(glob_path):
    """Load document chunks from files matching the given glob pattern.
    
    Args:
        glob_path (str): Glob pattern to match chunk files (e.g., './chunks/*/semantic_chunks/*.json')
    
    Returns:
        list: Collection of document chunks loaded from matched files
    
    Example:
        ```python
        chunks = load_chunks_from_glob('./chunks/wheel_of_time/semantic_chunks/*.json')
        ```
    """
    doc_collection = []
    for chunk_path in glob.glob(glob_path):
        doc_collection.extend(load_chunk_from_disk(chunk_path))
    return doc_collection

# Load document chunks for the specified series
doc_collection = load_chunks_from_glob(f'./chunks/{series_metadata_name}/semantic_chunks/*.json')
print('Loaded chunks')

# Parse series information from configuration
series_info = [StorySageSeries.from_dict(series) for series in series_dict]
print('Got series info')

# Find the series ID for the current series being processed
series_id = [series.series_id for series in series_info if series.series_metadata_name == series_metadata_name][0]
print(f'Processing series {series_metadata_name} with id {series_id}')

# Embed documents and store in vector database
embed_documents(doc_collection=doc_collection, vector_store=vector_store, series_id=series_id, entity_collection=entity_collection[series_metadata_name])