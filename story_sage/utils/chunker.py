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
    """A class for chunking text documents into semantically coherent sections.

    This class uses a sentence transformer model to compute embeddings for text
    sentences, then identifies chunk boundaries based on semantic similarity.
    Sentences that have high dissimilarity with their immediate neighbors are
    used as natural breakpoints, effectively splitting the text into segments.

    Attributes:
        model (SentenceTransformer): The sentence transformer model used for obtaining sentence embeddings.

    Example usage:
        >>> chunker = StorySageChunker()
        >>> text = "Once upon a time. A hero began a quest. Another sentence."
        >>> chunks = chunker.process_file(text)
        >>> print(chunks)
        ['Once upon a time.', 'A hero began a quest.', 'Another sentence.']

    Example results:
        ['Once upon a time.', 'A hero began a quest.', 'Another sentence.']
    """

    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v1'):
        """Initializes a StorySageChunker with a specified transformer model.

        Args:
            model_name (str): Name of the sentence transformer model to load. Defaults to 'sentence-transformers/all-mpnet-base-v1'.
        """
        # Initialize the sentence transformer model
        self.model = SentenceTransformer(model_name)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = self.model.to(device)

    def process_file(self, text: str, context_window=1, percentile_threshold=95, min_chunk_size=3):
        """Processes text into semantically meaningful chunks.

        Args:
            text (str): The text to be chunked.
            context_window (int): Number of sentences around a target for context.
            percentile_threshold (int): Percentile used to identify semantic breakpoints.
            min_chunk_size (int): Minimum acceptable number of sentences before merging.

        Returns:
            list: A list of text chunks inferred from semantic boundaries.

        Example:
            >>> chunker = StorySageChunker()
            >>> sample_text = "Sentence one. Sentence two. Sentence three."
            >>> result = chunker.process_file(sample_text, context_window=1, percentile_threshold=90, min_chunk_size=2)
            >>> print(result)
            ['Sentence one. Sentence two.', 'Sentence three.']
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
        """Reads multiple text files and organizes them by book and chapter.

        Parses filenames for book numbers, splits text by chapters, and saves
        results in an OrderedDict structure. Stop at 'GLOSSARY' if present.

        Args:
            file_path (str): A file path pattern (e.g., './texts/*.txt').

        Returns:
            OrderedDict: Keys are filenames, values contain book and chapter data.

        Example:
            >>> text_dict = chunker.read_text_files('./books/*.txt')
            >>> print(len(text_dict))
            3
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
        """Adds neighboring sentences to each sentence for improved context.

        Args:
            sentences (list): A list of sentence strings.
            window_size (int): The number of neighboring sentences to include.

        Returns:
            list: A list of contextualized sentences.
        """
        contextualized = []
        for i in range(len(sentences)):
            start = max(0, i - window_size)
            end = min(len(sentences), i + window_size + 1)
            context = ' '.join(sentences[start:end])
            contextualized.append(context)
        return contextualized

    def _calculate_distances(self, embeddings):
        """Calculates pairwise distances between consecutive sentence embeddings.

        Utilizes cosine similarity to determine semantic closeness before
        converting similarity to distance.

        Args:
            embeddings (list): Embeddings for each sentence.

        Returns:
            list: Distances between consecutive sentence embeddings.

        Raises:
            ValueError: If fewer than two embeddings are provided.
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
        """Identifies indices that exceed a given percentile threshold of distance.

        Args:
            distances (list): List of cosine distances between adjacent embeddings.
            threshold_percentile (float): Numeric percentile used to locate large gaps.

        Returns:
            list: Indices indicating semantically significant boundaries.
        """
        threshold = np.percentile(distances, threshold_percentile)
        return [i for i, dist in enumerate(distances) if dist > threshold]

    def _create_chunks(self, sentences, breakpoints):
        """Creates text chunks using identified breakpoints.

        Splits the list of sentences at each breakpoint and includes the final
        segment as its own chunk.

        Args:
            sentences (list): The original list of sentences.
            breakpoints (list): Indices that define where to split.

        Returns:
            list: A list of raw text chunks.
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
        """Merges chunks that fall below a minimum sentence count.

        This method ensures chunks are semantically meaningful by comparing
        the current chunk to its preceding and following chunks, choosing
        the most similar for merging if size is too small.

        Args:
            chunks (list): Initial list of chunked text.
            embeddings (list): Embeddings corresponding to each chunk.
            min_size (int): Minimum acceptable chunk size to avoid merging.

        Returns:
            list: A refined list of chunks ensuring each meets the size threshold.
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