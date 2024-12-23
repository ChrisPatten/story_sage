import os
from langchain_core.documents import Document
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import sent_tokenize
import glob
from collections import OrderedDict
import re

class StorySageChunker:
    """
    A class that handles the chunking of text documents into semantically coherent sections.

    This class provides methods to process text and split it into meaningful chunks using
    sentence embeddings and semantic similarity.

    Attributes:
        model (SentenceTransformer): The sentence transformer model used for embeddings.
    """

    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v1'):
        # Initialize the sentence transformer model
        self.model = SentenceTransformer(model_name)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = self.model.to(device)

    def process_file(self, text: str, context_window=1, percentile_threshold=95, min_chunk_size=3):
        """
        Process text and split it into semantically meaningful chunks.

        Args:
            text (str): The text to process.
            context_window (int): Number of sentences to consider on either side for context.
            percentile_threshold (int): Percentile threshold for identifying breakpoints.
            min_chunk_size (int): Minimum number of sentences in a chunk.

        Returns:
            list: List of text chunks.
        """
        sentences = sent_tokenize(text)
        contextualized = self._add_context(sentences, context_window)
        embeddings = self.model.encode(contextualized)
        distances = self._calculate_distances(embeddings)
        breakpoints = self._identify_breakpoints(distances, percentile_threshold)
        initial_chunks = self._create_chunks(sentences, breakpoints)
        final_chunks = self._merge_small_chunks(initial_chunks, embeddings, min_chunk_size)
        return final_chunks

    def read_text_files(self, file_path):
        """
        Read text files from the specified file path and organize them into a structured dictionary.
    
        Args:
            file_path (str): File path pattern to read text files from.
    
        Returns:
            OrderedDict: Ordered dictionary containing book information and text content.
        """
        chapter_pattern = re.compile(
            r'^\s*(CHAPTER)\s+(\d+|\w+)',
            re.IGNORECASE | re.MULTILINE
        )
    
        text_dict = OrderedDict()
        for file in glob.glob(file_path):
            fname = os.path.basename(file)
            book_number_match = re.match(r'^(\d+)_', fname)
            if not book_number_match:
                print(f'Warning: Filename "{fname}" does not start with a number followed by an underscore.')
                continue
            book_number = int(book_number_match.group(1))
            print(f'Name: {fname} Book Number: {book_number}')
            with open(file, 'r') as f:
                book_info = {'book_number': book_number, 'chapters': {0: []}}
                content = f.read()
                content = re.sub(chapter_pattern, r'\1 \2', content)
                chapter_number = 0
                for line in content.split('\n'):
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    if re.match(chapter_pattern, line):
                        chapter_number += 1
                        if chapter_number not in book_info['chapters']:
                            book_info['chapters'][chapter_number] = []
                    if re.match(r'GLOSSARY', line, re.IGNORECASE):
                        break
                    book_info['chapters'][chapter_number].append(line)
                text_dict[fname] = book_info
                print(f'Book {book_number} has {len(book_info["chapters"])} chapters (0 indexed to include prologue).')
        return text_dict

    def _add_context(self, sentences, window_size):
        """
        Combine sentences with their neighbors for better context.

        Args:
            sentences (list): List of sentences.
            window_size (int): Number of neighboring sentences to include.

        Returns:
            list: List of sentences combined with their context.
        """
        contextualized = []
        for i in range(len(sentences)):
            start = max(0, i - window_size)
            end = min(len(sentences), i + window_size + 1)
            context = ' '.join(sentences[start:end])
            contextualized.append(context)
        return contextualized

    def _calculate_distances(self, embeddings):
        """
        Calculate cosine distances between consecutive sentence embeddings.

        Args:
            embeddings (list): List of sentence embeddings.

        Returns:
            list: List of cosine distances between embeddings.
        """
        if len(embeddings) < 2:
            raise ValueError(f'At least two embeddings are required to calculate distances. Got {len(embeddings)}')
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            distance = 1 - similarity
            distances.append(distance)
        return distances

    def _identify_breakpoints(self, distances, threshold_percentile):
        """
        Identify natural breaking points in the text based on semantic distances.

        Args:
            distances (list): List of cosine distances between embeddings.
            threshold_percentile (float): Percentile threshold to identify breakpoints.

        Returns:
            list: Indices of sentences where breakpoints occur.
        """
        threshold = np.percentile(distances, threshold_percentile)
        return [i for i, dist in enumerate(distances) if dist > threshold]

    def _create_chunks(self, sentences, breakpoints):
        """
        Create initial text chunks based on identified breakpoints.

        Args:
            sentences (list): List of sentences.
            breakpoints (list): Indices where breakpoints occur.

        Returns:
            list: Initial list of text chunks.
        """
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
        """
        Merge small chunks with their most similar neighbor to enhance coherence.

        Args:
            chunks (list): List of initial text chunks.
            embeddings (list): List of embeddings corresponding to the chunks.
            min_size (int): Minimum acceptable chunk size.

        Returns:
            list: Optimized list of text chunks.
        """
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