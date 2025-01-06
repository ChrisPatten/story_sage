"""Text Processing and Semantic Chunking Utility.

This module provides functionality for processing large text documents (like books) into 
semantically coherent chunks using advanced NLP techniques. It leverages sentence embeddings 
and cosine similarity to ensure chunks maintain contextual relationships.

Key Features:
    - Processes multiple books and chapters from a series
    - Creates semantically meaningful text chunks
    - Saves processed chunks as JSON files for further use
    - Configurable chunk sizes and processing parameters

Example Usage:
    Process a specific series and book:
        $ python create_chunks.py --series_name "harry_potter" --book_numbers 1 2 3

    Process an entire series:
        $ python create_chunks.py --series_name "harry_potter"

Input Directory Structure:
    ./books/
        ├── series_name/
        │   ├── 01_book_title.txt
        │   ├── 02_book_title.txt
        │   └── ...

Output Directory Structure:
    ./chunks/
        ├── series_name/
        │   ├── bigger_chunks/
        │   │   ├──
"""

import os
import json
from story_sage.utils.chunker import StorySageChunker
from tqdm import tqdm
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process text files into semantic chunks.')
parser.add_argument('--series_name', type=str, help='Name of the series to process')
parser.add_argument('--book_numbers', type=str, nargs='+', help='List of book numbers to process')
args = parser.parse_args()

SERIES_NAME = args.series_name
BOOK_NUMBERS = args.book_numbers

# Create file patterns for each book number
file_patterns = []
if BOOK_NUMBERS:
    file_patterns = [f'./books/{SERIES_NAME}/{str(num).zfill(2)}_*.txt' for num in BOOK_NUMBERS]
else:
    file_patterns = [f'./books/{SERIES_NAME}/*_*.txt']

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
    
    # Process each file pattern
    for pattern in file_patterns:
        text_dict = chunker.read_text_files(pattern)
        
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
                    context_window=4,
                    percentile_threshold=70,
                    min_chunk_size=100
                )

                if not os.path.exists(f'chunks/{SERIES_NAME}'):
                    os.makedirs(f'chunks/{SERIES_NAME}')
                if not os.path.exists(f'chunks/{SERIES_NAME}/bigger_chunks'):
                    os.makedirs(f'chunks/{SERIES_NAME}/bigger_chunks')
                json.dump(chunks, open(f'chunks/{SERIES_NAME}/bigger_chunks/{book_number}_{chapter_number}.json', 'w'), indent=4)