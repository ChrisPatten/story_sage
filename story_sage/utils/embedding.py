"""
Embedding Utility

Provides functionality to generate embeddings for text documents using SentenceTransformer,
then store those embeddings in a ChromaDB vector store for similarity searches.

Features:
    • Load and process text chunks from disk.
    • Generate embeddings using SentenceTransformer.
    • Embed documents and add them to a vector store with metadata.
    • Utilize embeddings for similarity comparisons in retrieval applications.

Note:
    • ChromaDB vector store is created at the specified path (default: ./chroma_data).
    • 'chunks' directory should contain relevant `.pkl` or `.json` text chunk files.

Example Usage:
    $ python embedding.py

Example Results:
    • Displays progress bars for chunk processing and embedding.
    • After completion, all chunk embeddings are stored in the specified vector store.
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
    """Loads text chunks from disk and creates Document objects.

    This function detects whether the file is a pickle (.pkl) or JSON (.json),
    loads the contents, and converts each text chunk into a Document object
    including its book and chapter metadata.

    Args:
        file_path (str): Path to the pickle or JSON file containing text chunks.

    Returns:
        List[Document]: A list of Document objects with assigned metadata.

    Example:
        >>> docs = load_chunk_from_disk('1_1.pkl')
        >>> print(len(docs))
        10  # Number of chunks loaded
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
    """Wraps a SentenceTransformer model to generate embeddings for input documents.

    Attributes:
        model (SentenceTransformer): The transformer model used for creating embeddings.

    Example usage:
        >>> embedder = Embedder(model_name='all-MiniLM-L6-v2')
        >>> embeddings = embedder(['Hello world', 'Another doc'])
        >>> print(len(embeddings))
        2
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initializes an Embedder with a specific SentenceTransformer model.

        Args:
            model_name (str, optional): SentenceTransformer model name. Defaults to 'all-MiniLM-L6-v2'.
        """
        # Select device: use MPS if available, else CPU
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        self.model = SentenceTransformer(model_name)
        self.model = self.model.to(device)

    def __call__(self, input: Documents) -> Embeddings:
        """Generates embeddings for a list of text documents.

        Args:
            input (Documents): A list of text documents (strings).

        Returns:
            Embeddings: A list of embedding vectors for each document.

        Example:
            >>> embeddings = embedder(["Hello world", "How are you?"])
            >>> print(len(embeddings), len(embeddings[0]))
            2 384
        """
        return self.model.encode(input).tolist()

    def embed_documents(self, documents: Documents) -> Embeddings:
        """Produces embeddings for a list of documents with progress indication.

        Args:
            documents (Documents): A list of text documents.

        Returns:
            Embeddings: A list of embedding vectors, each representing a document.

        Example:
            >>> documents = ["Document one.", "Document two."]
            >>> embeddings = embedder.embed_documents(documents)
            >>> print(len(embeddings))
            2
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
    """Embeds a collection of Document objects and inserts them into the vector store.

    Assigns metadata (book number, chapter number, series ID) to each document, then
    processes entity tags if found within the text, marking the document accordingly
    before inserting into the vector store.

    Args:
        doc_collection (List[Document]): The documents to embed and store.
        vector_store (Collection): The vector store (ChromaDB collection) for insertion.
        series_id (int): Numeric ID to track the series.
        entity_collection (StorySageEntityCollection): Provides named entities for tagging.

    Example:
        >>> docs = [Document(page_content="Sample text.", metadata={"book_number":1, "chapter_number":1})]
        >>> embed_documents(docs, vector_store, 1, some_entity_collection)
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
    """Updates metadata for documents in the vector store by tagging relevant entities.

    Checks documents in the store that match the provided series ID (and optionally book number),
    identifies which entities are present in each document, and updates the corresponding metadata.

    Args:
        vector_store (Collection): The ChromaDB collection holding existing documents.
        entity_collection (StorySageEntityCollection): Contains entity data for tagging.
        series_id (int): Numeric ID for the series to match.
        book_number (int, optional): Restrict updates to a specific book. Defaults to None.

    Example:
        >>> update_tagged_entities(vector_store, entity_collection, series_id=1, book_number=1)
        # Updates tags for all documents from book 1 of series 1.
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
    """Main execution flow for embedding utility.

    1. Loads settings from YAML/JSON files.
    2. Initializes the ChromaDB client and Embedder.
    3. Retrieves or creates the vector store collection.
    4. Processes each series directory's text chunks.
    5. Embeds them and updates entity tags in the vector store.

    Example:
        $ python embedding.py

    Example result:
        • Creates/Populates a ChromaDB store with embeddings from loaded Documents.
        • Prints progress bars during chunk loading and embedding.
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