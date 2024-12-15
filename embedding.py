"""
Embedding Utility

This module provides functionality to generate embeddings for text documents using SentenceTransformer.
It is used to create vector representations of text chunks for similarity comparisons.

Requirements:
    - sentence-transformers
    - torch
"""

import pickle
from langchain_core.documents import Document
from typing import List
import re
from uuid import uuid4
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings, Collection
from sentence_transformers import SentenceTransformer
import torch
import tqdm
import glob
import os
import yaml

# Load series.yml to create a mapping from series_metadata_name to series_id
with open('series.yml', 'r') as f:
    series_list = yaml.safe_load(f)
metadata_to_id = {series['series_metadata_name']: series['series_id'] for series in series_list}

# Load all character dictionaries and merge them using the metadata_to_id mapping
character_dicts = {}
for filepath in glob.glob('./characters/*_characters.pkl'):
    with open(filepath, 'rb') as f:
        series_characters = pickle.load(f)
        # Extract series_metadata_name from filename
        filename = os.path.basename(filepath)
        match = re.match(r'(.+)_characters\.pkl', filename)
        if match:
            series_metadata_name = match.group(1)
            series_id = metadata_to_id.get(series_metadata_name)
            if series_id is not None:
                character_dicts[series_id] = series_characters
            else:
                print(f'Warning: No series_id found for series_metadata_name "{series_metadata_name}"')
        else:
            print(f'Warning: Filename "{filename}" does not match the expected pattern.')

def load_chunk_from_disk(file_path: str) -> List[Document]:
    """Load text from pkl and create Document."""
    doc_collection = []
    with open(file_path, 'rb') as f:
        chunks = pickle.load(f)
        # Extract series_metadata_name and get series_id
        filename = os.path.basename(file_path)
        match = re.match(r'(\d+)_(\d+)', filename)
        if match:
            series_metadata_name = match.group(1)
            book_number, chapter_number = map(int, match.groups())
            series_id = metadata_to_id.get(series_metadata_name)
            if series_id is None:
                print(f'Warning: No series_id found for series_metadata_name "{series_metadata_name}"')
                return doc_collection
        else:
            print(f'Warning: Filename "{filename}" does not match the expected pattern.')
            return doc_collection

        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    'book_number': book_number,
                    'chapter_number': chapter_number,
                    'series_id': series_id
                }
            )
            doc_collection.append(doc)
        del chunks
    return doc_collection

class Embedder(EmbeddingFunction):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the Embedder with a specified SentenceTransformer model.

        Args:
            model_name (str, optional): Name of the SentenceTransformer model to use. Defaults to 'all-MiniLM-L6-v2'.
        """
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        self.model = SentenceTransformer(model_name)
        self.model = self.model.to(device)

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()

    def embed_documents(self, documents: Documents) -> Embeddings:
        """
        Generate embeddings for a list of documents.

        Args:
            documents (Documents): List of document strings to generate embeddings for.

        Returns:
            Embeddings: List of embeddings corresponding to the input documents.
        """
        embedded_documents = []
        for document in tqdm(documents, desc='Embedding documents'):
            embedded_document = self.model.encode(document)
            embedded_documents.append(embedded_document)
        return embedded_documents

def embed_documents(doc_collection: List[Document], character_dict: dict, 
                    vector_store: Collection, series_name: str) -> None:
    """
    Embed documents and add them to the vector store.

    Args:
        doc_collection (List[Document]): List of Document objects to be embedded.
        character_dict (dict): Dictionary mapping character names to their IDs.
        vector_store (Collection): ChromaDB collection to store the embeddings.
        series_name (str): Name of the series to add to the document metadata.
    """
    uuids = [str(uuid4()) for _ in range(len(doc_collection))]
    documents_to_encode = []
    document_metadata = []
    for doc in doc_collection:
        characters_in_doc = set()
        for key in character_dict.keys():
            if key in str.lower(doc.page_content):
                characters_in_doc.add(character_dict[key])
        for char_id in characters_in_doc:
            doc.metadata[f'character_{char_id}'] = True
        doc.metadata['series_name'] = series_name
        documents_to_encode.append(doc.page_content)
        document_metadata.append(doc.metadata)

    vector_store.add(
        documents=documents_to_encode,
        metadatas=document_metadata,
        ids=uuids
    )

if __name__ == '__main__':
    chroma_client = chromadb.PersistentClient(path='./chroma_data')
    embedder = Embedder()
    vector_store = chroma_client.get_or_create_collection(
        name='story_sage',
        embedding_function=embedder
    )
    print('Created vector store')

    # Iterate over subdirectories in ./chunks
    for series_dir in glob.glob('./chunks/*'):
        if os.path.isdir(series_dir):
            series_name = os.path.basename(series_dir)
            print(f'Processing series: {series_name}')

            # Map series_name to series_id (assuming the directory name matches series_name in series.yml)
            series_id = next((s['series_id'] for s in series_list if s['series_name'] == series_name), None)
            if series_id is None:
                print(f'Could not find series_id for series: {series_name}')
                continue

            # Retrieve character dictionary for the series
            character_dict = character_dicts.get(series_id, {})
            print(f'Loaded character dictionary for series_id {series_id}')

            # Process chunks in the series directory
            for file in glob.glob(f'{series_dir}/*.pkl'):
                print(f'Embedding documents from {file}')
                doc_collection = load_chunk_from_disk(file)
                embed_documents(doc_collection, character_dict, vector_store, series_name)