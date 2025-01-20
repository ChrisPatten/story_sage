"""Text chunking utility for breaking documents into semantically meaningful segments.

This module provides functionality to split large text documents into smaller,
semantically coherent chunks using transformer-based embeddings and similarity
metrics. It's particularly useful for processing long-form content like books
or articles where maintaining context and meaning across chunk boundaries is
important.

Typical usage example:
    chunker = StorySageChunker()
    
    # Process a single text
    text = '''
    The wizard raised his staff. Lightning crackled through the air.
    Meanwhile, in the village below, people watched in terror.
    '''
    chunks = chunker.process_file(text)
    
    # Process multiple book files
    book_data = chunker.read_text_files('./books/*.txt')
"""

import os
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import sent_tokenize
import glob
from collections import OrderedDict
import re
from llama_index.core.node_parser import SentenceSplitter
from typing import Union, List
import logging

class ChapterData():
    def __init__(self, chapter_number: int):
        self.chapter_number = chapter_number
        self.full_text: str = ''
        self.lines: list[str] = []

class BookData():
    def __init__(self, book_number: int):
        self.book_number = book_number
        self.chapters: dict[int, ChapterData] = {0: ChapterData(0)}
class StorySageChunker:
    """Chunks text documents into semantically coherent sections using AI embeddings.

    This class implements an intelligent text chunking system that preserves semantic
    meaning across chunk boundaries. It uses transformer models to understand text
    context and identifies natural break points where chunks should be split.

    The chunking process involves:
    1. Converting sentences to embeddings using a transformer model
    2. Computing semantic similarity between adjacent sentences
    3. Identifying natural break points based on similarity thresholds
    4. Merging small chunks to maintain context

    Attributes:
        model: A SentenceTransformer instance used for text embeddings.

    Example:
        >>> chunker = StorySageChunker()
        >>> text = '''
        ...     The dragon soared through clouds. Its scales glittered in sunlight.
        ...     Meanwhile, in the castle below, the knights prepared for battle.
        ...     They sharpened their swords and donned their armor.
        ... '''
        >>> chunks = chunker.process_file(text)
        >>> for chunk in chunks:
        ...     print(f"Chunk: {chunk}\n")
        
        Chunk: The dragon soared through clouds. Its scales glittered in sunlight.
        
        Chunk: Meanwhile, in the castle below, the knights prepared for battle.
        They sharpened their swords and donned their armor.
    """

    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v1'):
        """Initializes the chunker with a specified transformer model.

        Args:
            model_name: String name of the sentence transformer model to use.
                Defaults to 'sentence-transformers/all-mpnet-base-v1'.
        """
        # Initialize the sentence transformer model
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except EnvironmentError:
            self.model = SentenceTransformer(model_name, local_files_only=False)
            
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.logger = logging.getLogger(__name__)

    

    def process_file(self, text: str, context_window=1, percentile_threshold=95, min_chunk_size=3) -> list:
        """Processes text into semantically cohesive chunks.

        Splits input text into chunks while maintaining semantic meaning and context.
        Uses a sliding window approach to consider surrounding context when making
        split decisions.

        Args:
            text: The input text to be chunked.
            context_window: Number of sentences to consider on either side when
                computing semantic similarity. Larger values preserve more context.
            percentile_threshold: Threshold percentile for identifying break points.
                Higher values result in fewer, larger chunks.
            min_chunk_size: Minimum number of sentences per chunk before attempting
                to merge with neighbors.

        Returns:
            List of string chunks, where each chunk is a semantically related
            group of sentences.

        Example:
            >>> text = '''
            ...     The sun set behind mountains. Stars began to appear.
            ...     In the forest, creatures stirred. A wolf howled distantly.
            ... '''
            >>> chunker = StorySageChunker()
            >>> chunks = chunker.process_file(
            ...     text,
            ...     context_window=1,
            ...     percentile_threshold=90,
            ...     min_chunk_size=2
            ... )
            >>> print(chunks)
            [
                'The sun set behind mountains. Stars began to appear.',
                'In the forest, creatures stirred. A wolf howled distantly.'
            ]
        """
        sentences = sent_tokenize(text)
        contextualized = self._add_context(sentences, context_window)
        embeddings = self.model.encode(contextualized)
        distances = self._calculate_distances(embeddings)
        breakpoints = self._identify_breakpoints(distances, percentile_threshold)
        initial_chunks = self._create_chunks(sentences, breakpoints)
        final_chunks = self._merge_small_chunks(initial_chunks, embeddings, min_chunk_size)
        return final_chunks

    def read_text_files(self, file_path: Union[str, List[str]]) -> OrderedDict[str, BookData]:
        """Reads and organizes multiple text files by book and chapter.

        Processes text files matching the given path pattern, extracting book
        numbers from filenames and splitting content into chapters. Stops
        processing at 'GLOSSARY' if present.

        Args:
            file_path: Glob pattern matching text files to process
                (e.g., './books/*.txt').

        Returns:
            OrderedDict with structure:
            {
                'filename.txt': {
                    'book_number': int,
                    'chapters': {
                        0: ['prologue content...'],
                        1: ['chapter 1 content...'],
                        ...
                    }
                },
                ...
            }

        Example:
            >>> chunker = StorySageChunker()
            >>> books = chunker.read_text_files('./fantasy_series/*.txt')
            >>> print(f"Found {len(books)} books")
            Found 3 books
            >>> print(f"Chapters in book 1: {len(books['1_book.txt']['chapters'])}")
            Chapters in book 1: 12
        """
        self.logger.debug(f'Reading text files from glob path: {file_path}')
        chapter_pattern = re.compile(
            r'^\s*(CHAPTER)\s+(\d+|\w+)',
            re.IGNORECASE | re.MULTILINE
        )
    
        text_dict = OrderedDict()
        if isinstance(file_path, str):
            file_path = [file_path]
        for path in file_path:
            for file in glob.glob(path):
                fname = os.path.basename(file)
                self.logger.debug(f'Reading file: {fname}')
                book_number_match = re.match(r'^(\d+)_', fname)
                if not book_number_match:
                    print(f'Warning: Filename "{fname}" does not start with a number followed by an underscore.')
                    continue
                book_number = int(book_number_match.group(1))
                print(f'Name: {fname} Book Number: {book_number}')
                with open(file, 'r') as f:
                    book_info = BookData(book_number)
                    content = f.read()
                    content = re.sub(chapter_pattern, r'\1 \2', content)
                    chapter_number = 0
                    for line in content.split('\n'):
                        line = line.strip()
                        if len(line) == 0:
                            continue
                        if re.match(chapter_pattern, line):
                            chapter_number += 1
                            if chapter_number not in book_info.chapters.keys():
                                book_info.chapters[chapter_number] = ChapterData(chapter_number)
                        if re.match(r'GLOSSARY', line, re.IGNORECASE):
                            break
                        line = re.sub(chapter_pattern, '', line)
                        # Strip all non-alphanumeric characters from the beginning of the line
                        line = re.sub(r'^\W+', '', line)
                        book_info.chapters[chapter_number].lines.append(line)
                        book_info.chapters[chapter_number].full_text += line + ' '
                    text_dict[fname] = book_info
                    print(f'Book {book_number} has {len(book_info.chapters)} chapters (0 indexed to include prologue).')
        return text_dict

    def _add_context(self, sentences: list, window_size: int) -> list:
        """Enhances sentences with surrounding context.

        For each sentence, creates a context window including neighboring sentences
        to improve semantic understanding when generating embeddings.

        Args:
            sentences: List of individual sentences.
            window_size: Number of sentences to include on either side.

        Returns:
            List of strings where each string contains the target sentence with
            its context window.
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

    def sentence_splitter(self, text: str, chunk_size: int = 1024, chunk_overlap: int = 128) -> list:
        """Split text into chunks using llama-index's SentenceSplitter.
        
        This method provides an alternative chunking strategy using llama-index's
        implementation which is optimized for maintaining sentence boundaries.
        
        Args:
            text: The input text to be split into chunks
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks split at sentence boundaries
            
        Example:
            >>> chunker = StorySageChunker()
            >>> text = "First sentence. Second sentence. Third sentence."
            >>> chunks = chunker.sentence_splitter(text, chunk_size=20)
            >>> print(chunks)
            ['First sentence.', 'Second sentence.', 'Third sentence.']
        """
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(text)
        return chunks