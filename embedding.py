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

def load_chunk_from_disk(file_path: str) -> List[Document]:
    """Load text from pkl and create Document."""
    doc_collection = []
    with open(file_path, 'rb') as f:
        chunks = pickle.load(f)
        book_number, chapter_number = map(int, re.match(r'(\d+)_(\d+)', os.path.basename(file)).groups())
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
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        self.model = SentenceTransformer(model_name)
        self.model = self.model.to(device)

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()

    def embed_documents(self, documents: Documents) -> Embeddings:
        embedded_documents = []
        for document in tqdm(documents, desc='Embedding documents'):
            embedded_document = self.model.encode(document)
            embedded_documents.append(embedded_document)
        return embedded_documents

def embed_documents(doc_collection: List[Document], character_dict: dict, 
                    vector_store: Collection ) -> None:
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
        name="wheel_of_time",
        embedding_function=embedder
    )
    print('Created vector store')

    with open('merged_characters.pkl', 'rb') as f:
        character_dict = pickle.load(f)
    print('Loaded character dictionary')

    print('Load chunks from disk')
    for file in glob.glob('./chunks/*.pkl'):
        print(f'Embedding documents from {file}')
        doc_collection = load_chunk_from_disk(file)
        embed_documents(doc_collection, character_dict, vector_store)