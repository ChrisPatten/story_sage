import os
import json
from story_sage.utils.chunker import StorySageChunker
from tqdm import tqdm

SERIES_NAME = 'the_expanse'

file_path = f'./books/{SERIES_NAME}/02_*.txt'

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
    chunker = StorySageChunker(model_name='all-MiniLM-L6-v2')
    text_dict = chunker.read_text_files(file_path)
    
    for book_name, book_info in text_dict.items():
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
                percentile_threshold=85,
                min_chunk_size=3
            )

            if not os.path.exists(f'chunks/{SERIES_NAME}'):
                os.makedirs(f'chunks/{SERIES_NAME}')
            if not os.path.exists(f'chunks/{SERIES_NAME}/semantic_chunks'):
                os.makedirs(f'chunks/{SERIES_NAME}/semantic_chunks')
            json.dump(chunks, open(f'chunks/{SERIES_NAME}/semantic_chunks/{book_number}_{chapter_number}.json', 'w'))