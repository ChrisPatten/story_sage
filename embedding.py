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

def load_chunk_from_disk(file_path: str) -> List[Document]:
    """Load text from pkl and create Document."""
    doc_collection = []
    with open(file_path, 'rb') as f:
        chunks = pickle.load(f)
        # Extract series_metadata_name and get series_id
        filename = os.path.basename(file_path)
        match = re.match(r'(\d+)_(\d+)', filename)
        if match:
            book_number, chapter_number = map(int, match.groups())
        else:
            print(f'Warning: Filename "{filename}" does not match the expected pattern.')
            return doc_collection

        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    'book_number': book_number,
                    'chapter_number': chapter_number
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

def embed_documents(doc_collection: List[Document], entities: dict, 
                    vector_store: Collection, series_id: int) -> None:
    """
    Embed documents and add them to the vector store.

    Args:
        doc_collection (List[Document]): List of Document objects to be embedded.
        character_dict (dict): Dictionary mapping character names to their IDs.
        vector_store (Collection): ChromaDB collection to store the embeddings.
        series_name (str): Name of the series to add to the document metadata.
    """
    ids = []
    documents_to_encode = []
    document_metadata = []
    series_metadata_name = next((item['series_metadata_name'] for item in series_list if item['series_id'] == series_id), None)
    if not series_metadata_name:
        print(f'Warning: No series_metadata_name found for series_id {series_id}')
        return
    series_entities = entities['series'][series_metadata_name]['series_entities']
    doc_seq = 0
    for doc in doc_collection:
        book_number = doc.metadata['book_number']
        chapter_number = doc.metadata['chapter_number']
        doc.metadata['series_id'] = series_id

        ids.append(f'{series_id}_{book_number}_{chapter_number}_{doc_seq}')
        doc_seq += 1

        series = entities['series'][series_metadata_name]
        book = series['books'][book_number - 1]
        cleaned_page_content = str.lower(doc.page_content.replace('’', "'").replace('‘', "'"))
        try:
            entities_in_chapter = next((ch for ch in book['chapters'] if ch['chapter'] == chapter_number), None)
            if not entities_in_chapter:
                print(f'Warning: No chapter found for book {book_number} chapter {chapter_number}')
                return
        except IndexError:
            print(f'Warning: No entities found for book {book_number} chapter {chapter_number}')
            return
        for name in entities_in_chapter['people']:
            if name in cleaned_page_content:
                doc.metadata[series_entities['people_by_name'][name]] = True
        for name in entities_in_chapter['places']:
            if name in cleaned_page_content:
                doc.metadata[series_entities['places_by_name'][name]] = True
        for name in entities_in_chapter['groups']:
            if name in cleaned_page_content:
                doc.metadata[series_entities['groups_by_name'][name]] = True
        for name in entities_in_chapter['animals']:
            if name in cleaned_page_content:
                doc.metadata[series_entities['animals_by_name'][name]] = True
        
        documents_to_encode.append(doc.page_content)
        document_metadata.append(doc.metadata)

    vector_store.add(
        documents=documents_to_encode,
        metadatas=document_metadata,
        ids=ids
    )

if __name__ == '__main__':
    chroma_client = chromadb.PersistentClient(path='./chroma_data')
    embedder = Embedder()
    #chroma_client.delete_collection('wot_retriever_test')
    vector_store = chroma_client.get_or_create_collection(
        name='wot_retriever_test',
        embedding_function=embedder
    )
    print('Created vector store')

    # Iterate over subdirectories in ./chunks
    for series_dir in glob.glob('./chunks/wheel_of_time/semantic_chunks'):
        if os.path.isdir(series_dir):
            # Process chunks in the series directory
            for file in tqdm(glob.glob(f'{series_dir}/1_*.pkl'), desc='Processing chunks'):
                doc_collection = load_chunk_from_disk(file)
                embed_documents(doc_collection, entities, vector_store, series_id=3)