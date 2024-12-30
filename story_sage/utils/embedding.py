"""
Embedding Utility

This module provides functionality to generate embeddings for text documents using SentenceTransformer.
It creates vector representations of text chunks for similarity comparisons and stores them in a ChromaDB vector store.

Features:
    - Load and process text chunks from disk.
    - Generate embeddings using a specified SentenceTransformer model.
    - Embed documents and add them to a vector store with associated metadata.
    - Utilize embeddings for similarity comparisons in applications like information retrieval.

Requirements:
    - sentence-transformers
    - torch
    - chromadb
    - tqdm
    - pyyaml

Example Usage:

    To run this script, execute the following command in the terminal:

        $ python embedding.py

    Ensure that the required files (`series.yml`, `entities.json`, `config.yml`) are present in the working directory,
    and that the necessary dependencies are installed.

Example Results:

    - The script will output progress bars indicating the processing of chunks and embedding of documents.
    - After running, the ChromaDB vector store will contain embeddings for all processed documents, accessible for similarity search.

Note:

    - The ChromaDB vector store will be created at the specified path (`./chroma_data`).
    - Ensure that the 'chunks' directory contains the appropriate pickle files with text chunks.
"""

import pickle
from langchain_core.documents import Document
from typing import List
import re
from uuid import uuid4
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings, Collection, GetResult
from sentence_transformers import SentenceTransformer
import torch
from tqdm.notebook import tqdm
import glob
import os
import yaml
import json
import argparse
from story_sage.story_sage_entity import StorySageEntityCollection

def load_chunk_from_disk(file_path: str) -> List[Document]:
    """
    Load text chunks from a pickle or JSON file and create a list of Document objects.

    Args:
        file_path (str): Path to the pickle or JSON file containing text chunks.

    Returns:
        List[Document]: A list of Document objects with page content and metadata.

    Example:
        >>> docs = load_chunk_from_disk('1_1.pkl')
        >>> print(len(docs))
        10  # Assuming the pickle file contains 10 chunks
    """
    doc_collection = []
    # Extract book and chapter numbers from the filename
    filename = os.path.basename(file_path)
    match = re.match(r'(\d+)_(\d+)', filename)
    if match:
        book_number, chapter_number = map(int, match.groups())
    else:
        print(f'Warning: Filename "{filename}" does not match the expected pattern.')
        return doc_collection

    # Determine file extension
    _, file_ext = os.path.splitext(file_path)
    file_ext = file_ext.lower()

    # Load chunks from the file
    if file_ext == '.pkl':
        with open(file_path, 'rb') as f:
            chunks = pickle.load(f)
    elif file_ext == '.json':
        with open(file_path, 'r') as f:
            chunks = json.load(f)
    else:
        print(f'Warning: Unsupported file extension "{file_ext}" for file "{filename}".')
        return doc_collection

    for chunk in chunks:
        # Create a Document object for each chunk
        doc = Document(
            page_content=chunk,
            metadata={
                'book_number': book_number,
                'chapter_number': chapter_number
            }
        )
        doc_collection.append(doc)
    del chunks  # Free memory
    return doc_collection

class Embedder(EmbeddingFunction):
    """
    Embedder class using SentenceTransformer to generate embeddings.

    This class wraps a SentenceTransformer model to generate embeddings compatible with ChromaDB.

    Attributes:
        model (SentenceTransformer): The loaded transformer model used for encoding texts.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the Embedder with a specified SentenceTransformer model.

        Args:
            model_name (str, optional): Name of the SentenceTransformer model to use.
                Defaults to 'all-MiniLM-L6-v2'.

        Example:
            >>> embedder = Embedder(model_name='distilbert-base-nli-stsb-mean-tokens')
        """
        # Select device: use MPS if available, else CPU
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        self.model = SentenceTransformer(model_name)
        self.model = self.model.to(device)

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the input documents.

        Args:
            input (Documents): List of document strings to generate embeddings for.

        Returns:
            Embeddings: List of embeddings corresponding to the input documents.

        Example:
            >>> embeddings = embedder(['Hello world', 'How are you?'])
            >>> print(len(embeddings))
            2
        """
        return self.model.encode(input).tolist()

    def embed_documents(self, documents: Documents) -> Embeddings:
        """
        Generate embeddings for a list of documents with progress indication.

        Args:
            documents (Documents): List of document strings to generate embeddings for.

        Returns:
            Embeddings: List of embeddings corresponding to the input documents.

        Example:
            >>> documents = ['Document one.', 'Document two.', 'Document three.']
            >>> embeddings = embedder.embed_documents(documents)
            >>> print(len(embeddings))
            3
        """
        embedded_documents = []
        # Embed documents with a progress bar
        for document in tqdm(documents, desc='Embedding documents'):
            embedded_document = self.model.encode(document)
            embedded_documents.append(embedded_document)
        return embedded_documents

def embed_documents(doc_collection: List[Document],
                    vector_store: Collection, series_id: int, 
                    entity_collection: StorySageEntityCollection) -> None:
    """
    Embed documents and add them to the vector store with metadata.

    Args:
        doc_collection (List[Document]): List of Document objects to be embedded.
        series_info (dict): Dictionary containing series information.
        vector_store (Collection): ChromaDB collection to store the embeddings.
        series_id (int): Identifier for the series.
    """
    ids = []
    documents_to_encode = []
    document_metadata = []
    doc_seq = 0  # Sequence counter for documents

    entities_by_name = entity_collection.get_group_ids_by_name()
    allowed_characters_pattern = r'[^a-z\s-]'

    for doc in tqdm(doc_collection, desc='Identifying characters in documents'):
        book_number = doc.metadata['book_number']
        chapter_number = doc.metadata['chapter_number']
        doc.metadata['series_id'] = series_id

        # Generate a unique ID for the document
        ids.append(f'{series_id}_{book_number}_{chapter_number}_{doc_seq}')
        doc_seq += 1

        entities_in_doc = set()

        # Extract entities for this document
        for entity_name, entity_group_id in entities_by_name.items():
            if entity_name in re.sub(allowed_characters_pattern, '', doc.page_content.lower(), flags=re.IGNORECASE):
                entities_in_doc.add(entity_name)
                doc.metadata[entity_group_id] = True

        documents_to_encode.append(doc.page_content)
        document_metadata.append(doc.metadata)

    batch_size = 1000
    for i in tqdm(range(0, len(documents_to_encode), batch_size), desc='Upserting documents in batches'):
        vector_store.upsert(
            documents=documents_to_encode[i:i+batch_size],
            metadatas=document_metadata[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

def update_tagged_entities(vector_store: Collection, entity_collection: StorySageEntityCollection, series_id: int, book_number: int = None) -> None:
    """
    Update tagged entities in the vector store with metadata.

    Args:
        vector_store (Collection): ChromaDB collection containing the embeddings.
        entity_collection (StorySageEntityCollection): Collection of tagged entities.
        series_id (int): Identifier for the series.
    """
    entities_by_name = entity_collection.get_group_ids_by_name()
    allowed_characters_pattern = r'[^a-z\s-]'

    where_document = {'series_id': series_id}
    if book_number is not None:
        where_document = { '$and': [ {'series_id': series_id}, {'book_number': book_number } ]}

    # Get all documents in the vector store
    results: GetResult = vector_store.get(where=where_document)
    num_documents = len(results['documents'])
    ids_to_update = []
    metadatas_to_update = []

    for idx, document in tqdm(enumerate(results['documents']), desc='Updating tagged entities', total=num_documents):
        entities_in_doc = set()
        document_metadata = { 'series_id': series_id, 'book_number': results['metadatas'][idx]['book_number'], 'chapter_number': results['metadatas'][idx]['chapter_number'] }

        # Extract entities for this document
        for entity_name, entity_group_id in entities_by_name.items():
            if entity_name in re.sub(allowed_characters_pattern, '', document.lower(), flags=re.IGNORECASE):
                entities_in_doc.add(entity_name)
                document_metadata[entity_group_id] = True

        ids_to_update.append(results['ids'][idx])
        metadatas_to_update.append(document_metadata)

    # Update documents and their metadata in the vector store
    vector_store.update(
        metadatas=metadatas_to_update,
        ids=ids_to_update
    )

if __name__ == '__main__':
    """
    Main Execution

    This main block demonstrates how to initialize the Embedder, load documents,
    generate embeddings, and store them in a ChromaDB vector store.

    Steps:
        - Initialize the ChromaDB client and Embedder.
        - Optionally delete existing collection.
        - Get or create the vector store collection.
        - Process each series specified.
        - For each series, process and embed documents from chunks.

    Example Usage:

        $ python embedding.py

    Example Results:

        - The script will output progress bars indicating the processing of chunks and embedding of documents.
        - After running, the vector store will contain embeddings for all processed documents, accessible for similarity search.

    Note:

        - Ensure that the 'series.yml', 'entities.json', and 'config.yml' files exist in the working directory.
        - The 'chunks' directory should contain the appropriate pickle files with text chunks.
        - The ChromaDB vector store will be created at the specified path ('./chroma_data').
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Embedding Utility Script')
    parser.add_argument('--series_list_path', type=str, default='series.yml',
                        help='Path to the series.yml file (default: series.yml)')
    parser.add_argument('--entities_path', type=str, default='entities.json',
                        help='Path to the entities.json file (default: entities.json)')
    parser.add_argument('--config_path', type=str, default='config.yml',
                        help='Path to the config.yml file (default: config.yml)')
    parser.add_argument('--series_id', type=list[int], default=None,
                        help='Series ID to process (default: None)')
    args = parser.parse_args()

    # Load the series list from the provided path
    with open(args.series_list_path, 'r') as f:
        series_list = yaml.safe_load(f)

    # Load the entities from the provided path
    with open(args.entities_path, 'r') as f:
        entities = json.load(f)

    # Load the configuration from the provided path
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize the ChromaDB client with a persistent storage path
    chroma_client = chromadb.PersistentClient(path=config['CHROMA_PATH'])
    embedder = Embedder()

    if args.series_id is not None:
        series_to_process = args.series_id

    #chroma_client.delete_collection(config['CHROMA_COLLECTION'])  # Delete the collection if it already exists

    # Get or create a collection in the vector store
    vector_store = chroma_client.get_or_create_collection(
        name=config['CHROMA_COLLECTION'],
        embedding_function=embedder
    )
    print('Got vector store')

    # Iterate over subdirectories in ./chunks
    for series_id in series_to_process:
        series_info = next((series for series in series_list if series['series_id'] == series_id), None)
        series_name = series_info['series_name']
        series_metadata_name = series_info['series_metadata_name']
        print(f'Processing series: {series_name} | {series_metadata_name} | {series_id}')
        series_dir = f'./chunks/{series_metadata_name}/semantic_chunks/'
        if os.path.isdir(series_dir):
            # Collect all .pkl and .json files in the series directory
            chunk_files = glob.glob(f'{series_dir}/*.pkl') + glob.glob(f'{series_dir}/*.json')
            if not chunk_files:
                raise ValueError(f'No chunk files found in directory {series_dir}')
            # Process chunks in the series directory
            for file in tqdm(chunk_files, desc=f'Processing chunks for {series_name}'):
                # Load documents from disk
                doc_collection = load_chunk_from_disk(file)
                # Embed documents and add them to the vector store
                embed_documents(doc_collection, series_info, vector_store, series_id=series_id)

    """
    Example Results:

    - The script will output progress bars indicating the processing of chunks and embedding of documents.
    - After running, the vector store will contain embeddings for all processed documents, accessible for similarity search.

    Note:

    - Ensure that the 'series.yml' and 'entities.json' files exist in the working directory.
    - The 'chunks' directory should contain the appropriate pickle files with text chunks.
    - The ChromaDB vector store will be created at the specified path ('./chroma_data').
    """