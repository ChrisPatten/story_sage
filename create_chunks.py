"""
chunking.py

Provides a text chunking utility for semantically splitting long documents using
sentence embeddings and cosine similarity. The resulting chunks can be stored in
JSON files for further processing and retrieval.

Example usage:
    $ python chunking.py

After running:
    1. The script reads all text files in a series directory (e.g., './books/harry_potter/*.txt').
    2. Each chapter from each book is processed into coherent chunks.
    3. Generated chunks are saved as JSON in './chunks/<series_name>/semantic_chunks/'.

Example results:
    The script creates JSON files for each chapter with a list of text chunks. 
    For instance:
    [
        "First chunk of text...",
        "Second chunk of text...",
        ...
    ]

Note:
    - Requires nltk, sentence-transformers, scikit-learn, and numpy packages.
    - Chunks are written to disk for potential embedding or indexing elsewhere.
"""

import os
import json
from story_sage.utils.chunker import StorySageChunker
from tqdm import tqdm
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process text files into semantic chunks.')
parser.add_argument('--series_name', type=str, help='Name of the series to process')
args = parser.parse_args()

SERIES_NAME = args.series_name



file_path = f'./books/{SERIES_NAME}/*.txt'

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

# Removed the standalone read_text_file function

if __name__ == '__main__':
    """Main entry point for the chunking utility.

    Iterates through all books and chapters for the specified series, uses the
    StorySageChunker to create coherent text chunks, and saves them as JSON.
    """
    chunker = StorySageChunker(model_name='all-MiniLM-L6-v2')
    text_dict = chunker.read_text_files(file_path)
    
    for _, book_info in text_dict.items():
        book_number = book_info['book_number']
        for chapter_number, chapter_text in tqdm(book_info['chapters'].items(), desc=f'Processing book {book_number}'):
            chapter_text_length = len(''.join(chapter_text).replace(' ', ''))
            if chapter_text_length < 100:
                continue
            # Concatenate the elements in chapter_text
            full_text = ' '.join(chapter_text)
            chunks = chunker.process_file(
                text=full_text,
                context_window=2,
                percentile_threshold=80,
                min_chunk_size=4
            )

            if not os.path.exists(f'chunks/{SERIES_NAME}'):
                os.makedirs(f'chunks/{SERIES_NAME}')
            if not os.path.exists(f'chunks/{SERIES_NAME}/semantic_chunks'):
                os.makedirs(f'chunks/{SERIES_NAME}/semantic_chunks')
            json.dump(chunks, open(f'chunks/{SERIES_NAME}/semantic_chunks/{book_number}_{chapter_number}.json', 'w'))