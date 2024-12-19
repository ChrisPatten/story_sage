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
from chromadb import Documents, EmbeddingFunction, Embeddings, Collection
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import glob
import os
import yaml
import json

# Load series.yml to create a mapping from series_metadata_name to series_id
with open('series.yml', 'r') as f:
    series_list = yaml.safe_load(f)

with open('entities.json', 'r') as f:
    entities = json.load(f)

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

def load_chunk_from_disk(file_path: str) -> List[Document]:
    """
    Load text chunks from a pickle file and create a list of Document objects.

    Args:
        file_path (str): Path to the pickle file containing text chunks.

    Returns:
        List[Document]: A list of Document objects with page content and metadata.

    Example:
        >>> docs = load_chunk_from_disk('1_1.pkl')
        >>> print(len(docs))
        10  # Assuming the pickle file contains 10 chunks
    """
    doc_collection = []
    with open(file_path, 'rb') as f:
        chunks = pickle.load(f)
        # Extract book and chapter numbers from the filename
        filename = os.path.basename(file_path)
        match = re.match(r'(\d+)_(\d+)', filename)
        if match:
            book_number, chapter_number = map(int, match.groups())
        else:
            print(f'Warning: Filename "{filename}" does not match the expected pattern.')
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

def embed_documents(doc_collection: List[Document], entities: dict,
                    vector_store: Collection, series_id: int) -> None:
    """
    Embed documents and add them to the vector store with metadata.

    Args:
        doc_collection (List[Document]): List of Document objects to be embedded.
        entities (dict): Dictionary containing entity information for the series.
        vector_store (Collection): ChromaDB collection to store the embeddings.
        series_id (int): Identifier for the series.

    Example:
        >>> embed_documents(doc_collection, entities, vector_store, series_id=3)
    """
    ids = []
    documents_to_encode = []
    document_metadata = []
    series_id_key = str(series_id)
    series = entities['series'][series_id_key]
    series_entities = series['series_entities']
    doc_seq = 0  # Sequence counter for documents

    for doc in doc_collection:
        book_number = doc.metadata['book_number']
        chapter_number = doc.metadata['chapter_number']
        doc.metadata['series_id'] = series_id

        # Generate a unique ID for the document
        ids.append(f'{series_id}_{book_number}_{chapter_number}_{doc_seq}')
        doc_seq += 1

        # Clean the document content for entity matching
        cleaned_page_content = ''.join(c for c in str.lower(doc.page_content) if c.isalpha() or c.isspace())

        # Add entity presence flags to metadata
        # ChromaDB doesn't support "where in" filters, so create a metadata field for each entity with value True
        for name, id in series_entities['people_by_name'].items():
            if name in cleaned_page_content:
                doc.metadata[id] = True

        for name, id in series_entities['entity_by_name'].items():
            if name in cleaned_page_content:
                doc.metadata[id] = True

        documents_to_encode.append(doc.page_content)
        document_metadata.append(doc.metadata)

    # Add documents and their embeddings to the vector store
    vector_store.upsert(
        documents=documents_to_encode,
        metadatas=document_metadata,
        ids=ids
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
    # Initialize the ChromaDB client with a persistent storage path
    chroma_client = chromadb.PersistentClient(path='./chroma_data')
    embedder = Embedder()

    chroma_client.delete_collection(config['CHROMA_COLLECTION'])  # Delete the collection if it already exists

    # Get or create a collection in the vector store
    vector_store = chroma_client.get_or_create_collection(
        name=config['CHROMA_COLLECTION'],
        embedding_function=embedder
    )
    print('Created vector store')

    series_to_process = [ 2, 3 ]  # Series IDs to process

    # Iterate over subdirectories in ./chunks
    for series_id in series_to_process:
        series_info = next((series for series in series_list if series['series_id'] == series_id), None)
        series_name = series_info['series_name']
        series_metadata_name = series_info['series_metadata_name']
        print(f'Processing series: {series_name} | {series_metadata_name} | {series_id}')
        series_dir = f'./chunks/{series_metadata_name}/semantic_chunks/'
        if os.path.isdir(series_dir):
            # Process chunks in the series directory
            for file in tqdm(glob.glob(f'{series_dir}/*.pkl'), desc=f'Processing chunks for {series_name}'):
                # Load documents from disk
                doc_collection = load_chunk_from_disk(file)
                # Embed documents and add them to the vector store
                embed_documents(doc_collection, entities, vector_store, series_id=series_id)

    """
    Example Results:

    - The script will output progress bars indicating the processing of chunks and embedding of documents.
    - After running, the vector store will contain embeddings for all processed documents, accessible for similarity search.

    Note:

    - Ensure that the 'series.yml' and 'entities.json' files exist in the working directory.
    - The 'chunks' directory should contain the appropriate pickle files with text chunks.
    - The ChromaDB vector store will be created at the specified path ('./chroma_data').
    """