#pip install --no-cache-dir jupyter langchain_openai langchain_community langchain langgraph faiss-cpu sentence-transformers ipywidgets transformers nltk scikit-learn matplotlib

CREATE_CHUNKS = True
USE_CHROMA = True

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS, InMemoryVectorStore
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import yaml
import ipywidgets as widgets
from IPython.display import display
import httpx
import torch
import pickle
from tqdm import tqdm
import glob
from collections import OrderedDict
import re
from uuid import uuid4
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('MPS backend available')
else:
    device = torch.device('cpu')
    print('MPS backend not available. Using CPU')

#tokenizer = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
tokenizer = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = tokenizer.to(device)

"""
Text Chunking Utility

This module provides functionality to intelligently chunk text documents into semantically coherent sections
using sentence embeddings and cosine similarity. It's particularly useful for processing large documents
while maintaining contextual relationships between sentences.

Requirements:
    - nltk
    - sentence-transformers
    - scikit-learn
    - numpy
    - matplotlib
"""

import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TextChunker:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v1'):
        """Initialize the TextChunker with a specified sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.model = self.model.to(device)

    def process_file(self, sentences, context_window=1, percentile_threshold=95, min_chunk_size=3):
        """
        Process a text file and split it into semantically meaningful chunks.
        
        Args:
            file_path: Path to the text file
            context_window: Number of sentences to consider on either side for context
            percentile_threshold: Percentile threshold for identifying breakpoints
            min_chunk_size: Minimum number of sentences in a chunk
            
        Returns:
            list: Semantically coherent text chunks
        """
        # Process the text file
        sentences = sent_tokenize(sentences)
        contextualized = self._add_context(sentences, context_window)
        embeddings = self.model.encode(contextualized)
        
        # Create and refine chunks
        distances = self._calculate_distances(embeddings)
        breakpoints = self._identify_breakpoints(distances, percentile_threshold)
        initial_chunks = self._create_chunks(sentences, breakpoints)
        
        # Merge small chunks for better coherence
        chunk_embeddings = self.model.encode(initial_chunks)
        final_chunks = self._merge_small_chunks(initial_chunks, chunk_embeddings, min_chunk_size)
        
        return final_chunks

    def _load_text(self, file_path):
        """Load and tokenize text from a file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return sent_tokenize(text)

    def _add_context(self, sentences, window_size):
        """Combine sentences with their neighbors for better context."""
        contextualized = []
        for i in range(len(sentences)):
            start = max(0, i - window_size)
            end = min(len(sentences), i + window_size + 1)
            context = ' '.join(sentences[start:end])
            contextualized.append(context)
        return contextualized

    def _calculate_distances(self, embeddings):
        """Calculate cosine distances between consecutive embeddings."""
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            distance = 1 - similarity
            distances.append(distance)
        return distances

    def _identify_breakpoints(self, distances, threshold_percentile):
        """Find natural breaking points in the text based on semantic distances."""
        threshold = np.percentile(distances, threshold_percentile)
        return [i for i, dist in enumerate(distances) if dist > threshold]

    def _create_chunks(self, sentences, breakpoints):
        """Create initial text chunks based on identified breakpoints."""
        chunks = []
        start_idx = 0
        
        for breakpoint in breakpoints:
            chunk = ' '.join(sentences[start_idx:breakpoint + 1])
            chunks.append(chunk)
            start_idx = breakpoint + 1
            
        # Add the final chunk
        final_chunk = ' '.join(sentences[start_idx:])
        chunks.append(final_chunk)
        
        return chunks

    def _merge_small_chunks(self, chunks, embeddings, min_size):
        """Merge small chunks with their most similar neighbor."""
        final_chunks = [chunks[0]]
        merged_embeddings = [embeddings[0]]
        
        for i in range(1, len(chunks) - 1):
            current_chunk_size = len(chunks[i].split('. '))
            
            if current_chunk_size < min_size:
                # Calculate similarities
                prev_similarity = cosine_similarity([embeddings[i]], [merged_embeddings[-1]])[0][0]
                next_similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                
                if prev_similarity > next_similarity:
                    # Merge with previous chunk
                    final_chunks[-1] = f"{final_chunks[-1]} {chunks[i]}"
                    merged_embeddings[-1] = (merged_embeddings[-1] + embeddings[i]) / 2
                else:
                    # Merge with next chunk
                    chunks[i + 1] = f"{chunks[i]} {chunks[i + 1]}"
                    embeddings[i + 1] = (embeddings[i] + embeddings[i + 1]) / 2
            else:
                final_chunks.append(chunks[i])
                merged_embeddings.append(embeddings[i])
        
        final_chunks.append(chunks[-1])
        return final_chunks

def read_text_file(file_path):
    text_dict = OrderedDict()
    for file in glob.glob(file_path):
        fname = os.path.basename(file)
        book_number = int(re.match(r'^(\d+)_', fname).group(1))
        print(f'Name: {fname} Book Number: {book_number}')
        with open(file, 'r') as f:
            book_info = {'book_number': book_number, 'chapters': {0: []}}
            # Remove any line breaks between the word "chapter" and following digits
            content = f.read()
            content = re.sub(r'(CHAPTER)\s+(\d+)', r'\1 \2', content, flags=re.IGNORECASE)
            chapter_number = 0
            for line in content.split('\n'):
                line = line.strip()
                if len(line) == 0:
                    continue
                if re.match(r'CHAPTER \d+', line, re.IGNORECASE):
                    chapter_number += 1
                    if chapter_number not in book_info['chapters']:
                        book_info['chapters'][chapter_number] = []
                if re.match(r'GLOSSARY', line, re.IGNORECASE):
                    break
                book_info['chapters'][chapter_number].append(line)
            text_dict[fname] = book_info
    return text_dict

file_path = './books/*.txt'
text_dict = read_text_file(file_path)
doc_collection = []
chunker = TextChunker(model_name='all-MiniLM-L6-v2')

if CREATE_CHUNKS:
  for book_name, book_info in text_dict.items():
      book_number = book_info['book_number']
      for chapter_number, chapter_text in tqdm(book_info['chapters'].items(), desc=f'Processing chapters in {book_name}'):
          # Concatenate the elements in chapter_text
          full_text = ' '.join(chapter_text)
          chunks = chunker.process_file(
              full_text,
              context_window=2,
              percentile_threshold=85,
              min_chunk_size=3
          )

          with open(f'chunks/{book_number}_{chapter_number}.pkl', 'wb') as f:
              pickle.dump(chunks, f)

          #print('Wrote chunks to disk for book', book_number, 'chapter', chapter_number)
          
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
else:
  print('Load chunks from disk')
  for file in glob.glob('./chunks/*.pkl'):
    with open(file, 'rb') as f:
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
  print('Loaded', len(doc_collection), 'chunks')

del chunker

def encode_documents(doc_collection):
  # Encode the documents and store the embeddings along with metadata
  embeddings = []
  metadata = []

  for doc in tqdm(doc_collection, desc='Encoding documents'):
    embedding = tokenizer.encode([doc.page_content])[0]
    embeddings.append(embedding)
    metadata.append(doc.metadata)

  embeddings = np.array(embeddings)

  # Create and populate the FAISS index
  index = faiss.IndexFlatIP(embeddings.shape[1])
  print('Adding embeddings to index...')
  index.add(embeddings)

  return index, metadata

if USE_CHROMA:
  
  class Embedder(EmbeddingFunction):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
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
  
  embedder = Embedder()

  chroma_client = chromadb.PersistentClient(path='./chroma_data')

  vector_store = chroma_client.get_or_create_collection(
     name="wheel_of_time",
     embedding_function=embedder
  )

  with open('merged_characters.pkl', 'rb') as f:
    character_dict = pickle.load(f)
  print('Loaded character dictionary')

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


  # Add documents to Chroma database
  vector_store.add(
      documents=documents_to_encode,
      metadatas=document_metadata,
      ids=uuids
  )