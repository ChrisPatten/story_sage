"""Process and analyze books for semantic search and story analysis.

This script processes books into semantic chunks for advanced story analysis and search.
It performs the following main functions:
1. Reads book text files from a specified series
2. Chunks text into semantic segments using StorySageChunker
3. Generates AI summaries using GPT-4
4. Creates embeddings via SentenceTransformer and stores them in ChromaDB
5. Extracts metadata like characters, locations, and key objects

Key Features:
    - Processes multiple books in a series
    - Handles chapter-by-chapter analysis
    - Generates both summary and full-text embeddings
    - Extracts structured story elements (characters, locations, etc.)
    - Supports incremental processing with skip options

Dependencies:
    - StorySageChunker: Handles semantic text chunking with configurable windows
    - StorySageConfig: Manages application settings and series configurations
    - Embedder: Generates text embeddings using SentenceTransformer

Example Usage:
    # Process specific books in a series:
    python process_books.py --series_name "harry_potter" --book_numbers 1 2

    # Process entire series, skip chunking step:
    python process_books.py --series_name "lord_of_rings" --skip_chunking

Directory Structure:
    chunks/
        series_name/
            bigger_chunks/      # Raw text chunks
                01_1.json      # Book 1, Chapter 1
                01_2.json      # Book 1, Chapter 2
            summaries/         # Processed summaries
                01_1.json     # Book 1, Chapter 1
                01_2.json     # Book 1, Chapter 2

Returns:
    None. Results are saved to disk and ChromaDB.

Note:
    - The script requires a valid configuration file (config.yml)
    - OpenAI API key must be configured in the config file
    - Input books should follow the naming convention: XX_bookname.txt
    - Requires sufficient disk space for storing chunks and embeddings
"""

import os
import json
from story_sage.utils.chunker import StorySageChunker
from tqdm import tqdm
import argparse
from openai import OpenAI
from pydantic import BaseModel
from story_sage.models.story_sage_config import StorySageConfig
import yaml
import httpx
from glob import glob
from story_sage.utils.embedding import Embedder
import chromadb
import pickle
from typing import List, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process text files into semantic chunks.')
parser.add_argument('--series_name', type=str, help='Name of the series to process')
parser.add_argument('--book_numbers', type=str, nargs='+', help='List of book numbers to process')
parser.add_argument('--skip_chunking', action='store_true', help='Skip the chunking process')
args = parser.parse_args()

SERIES_NAME = args.series_name
BOOK_NUMBERS = args.book_numbers
SKIP_CHUNKING = args.skip_chunking

def load_config(config_path):
    """Load and parse StorySage configuration from YAML.

    Args:
        config_path (str): Path to YAML configuration file.

    Returns:
        dict: Parsed configuration containing:
            - chroma_path: Path to ChromaDB storage
            - chroma_collection: Name of collection
            - openai_api_key: API key for GPT access
            - series_configurations: Book series settings

    Raises:
        yaml.YAMLError: If YAML parsing fails
        FileNotFoundError: If config file not found
    """
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

config = StorySageConfig.from_config(load_config('config.yml'))
CHROMA_PATH = config.chroma_path
CHROMA_COLLECTION = config.chroma_collection

series_info = config.get_series_by_meta_name(SERIES_NAME)

print('Config loaded')

# Create file patterns for each book number
file_patterns = []
if BOOK_NUMBERS:
    file_patterns = [f'./books/{SERIES_NAME}/{num.zfill(2)}_*.txt' for num in BOOK_NUMBERS]
else:
    file_patterns = [f'./books/{SERIES_NAME}/*_*.txt']

if not SKIP_CHUNKING:
    chunker = StorySageChunker(model_name='all-MiniLM-L6-v2')
    client = OpenAI(
        api_key=config.openai_api_key, http_client=httpx.Client(verify=False)
    )
    print('OpenAI client created')

class CharacterActions(BaseModel):
    """Models character actions within a text chunk.
    
    Represents a character and their actions as extracted from the text,
    used for structured story element extraction.

    Attributes:
        character (str): Name of the character
        actions (str): Description of character's actions/behavior in the chunk
    """
    character: str
    actions: str

class ChunkSummary(BaseModel):
    """
    Data model for storing chunk summaries and extracted information.

    Attributes:
        summary (str): Condensed summary of the text chunk
        characters (list[CharacterActions]): List of character actions
        locations (list[str]): List of locations mentioned
        creatures (list[str]): List of creatures mentioned
        objects (list[str]): List of significant objects mentioned
    """
    summary: str
    characters: list[CharacterActions]
    locations: list[str]
    creatures: list[str]
    objects: list[str]

def get_summary(text: str):
    """Generate structured summary and extract story elements using GPT-4.

    Uses AI to analyze text chunks and extract key story elements in a structured format.
    Automatically adjusts summary length based on input text length.

    Args:
        text (str): Text chunk to analyze.

    Returns:
        tuple:
            str: Condensed summary of the chunk
            ChunkSummary: Structured data containing extracted elements
            dict: API usage statistics for monitoring

    Dependencies:
        Requires a configured OpenAI client with access to GPT-4
        Uses StorySageConfig for API settings and prompt templates
    """
    summary_len = round(len(text.split(' ')) / 5)
    if summary_len < 50:
        summary_len = 50
    chat_completion = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "system",
                "content": f"""
                    You are an advanced litarary assistant who specializes in 
                        summarizing chunks from novels to optimize the quality of 
                        embeddings for a RAG application.
                    Use simple language and focus on capturing as much meaning 
                        in as few words as possible to help with the similarity search.

                    DO NOT INCLUDE ANYTHING THAT DOESN'T APPEAR IN THE PASSAGE.
                    
                    Please extract SUMMARY that is concise and should be no longer than {summary_len} words. 
                    Please extract CHARACTERS along with a brief description of their ACTIONS.
                    Please extract LOCATIONS.
                    Please extract CREATURES.
                    Please extract OBJECTS.
                """
            },
            {
                "role": "user",
                "content": text
            }
        ],
        model="gpt-4o-mini",
        response_format=ChunkSummary
    )
    summary = chat_completion.choices[0].message.parsed.summary
    if len(summary.split(' ')) > len(text.split(' ')):
        summary = text
    return summary, chat_completion.choices[0].message.parsed, chat_completion.usage

# Main processing logic
if not SKIP_CHUNKING:
    # Initialize storage for summaries and API usage tracking
    summaries: List[Tuple[str, str, int, dict]] = []
    usage = []

    # Process each book file pattern
    for pattern in file_patterns:
        # Load and process text files matching the pattern
        text_dict = chunker.read_text_files(pattern)
        print(f'Processing {pattern}')
        
        # Store summaries for the current book
        book_summaries: List[Tuple[str, str, int, dict]] = []

        # Process each book's chapters
        for _, book_info in text_dict.items():
            book_number = book_info['book_number']
            processed_chapters = 0
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

                for idx, chunk in enumerate(chunks):
                    chunk_summary, response, usage_ = get_summary(chunk)
                    book_summaries.append((book_number, chapter_number, idx, {
                        'full_chunk': chunk,
                        'summary': chunk_summary,
                        'characters': [{character.character: character.actions} for character in response.characters],
                        'locations': response.locations,
                        'creatures': response.creatures,
                        'objects': response.objects
                    }))
                    usage.append(usage_)
                
                processed_chapters += 1

        if not os.path.exists(f'chunks/{SERIES_NAME}/summaries'):
            os.makedirs(f'chunks/{SERIES_NAME}/summaries')

        with open(f'chunks/{SERIES_NAME}/summaries/summaries_{book_number}.pkl', 'wb') as f:
            pickle.dump(book_summaries, f)

        by_book_and_chapter: dict[str, dict[str, list[tuple[int, dict]]]] = {}
        for chunk_summary in book_summaries:
            book_number, chapter_number, chunk_index, summary = chunk_summary
            book_number = str(book_number)
            chapter_number = str(chapter_number)
            if book_number not in by_book_and_chapter:
                by_book_and_chapter[str(book_number)] = {}
            if chapter_number not in by_book_and_chapter[str(book_number)]:
                by_book_and_chapter[str(book_number)][str(chapter_number)] = []
            by_book_and_chapter[str(book_number)][str(chapter_number)].append((chunk_index, summary))

        for book_number, chapters in by_book_and_chapter.items():
            for chapter_number, book_summaries in chapters.items():
                book_summaries.sort(key=lambda x: x[0])
                summary = [summary[1] for summary in book_summaries]
                json.dump(summary, open(f'chunks/{SERIES_NAME}/summaries/{book_number}_{chapter_number}.json', 'w'), indent=4)


if BOOK_NUMBERS:
    glob_expression = f'chunks/{SERIES_NAME}/summaries/[{",".join(BOOK_NUMBERS)}]_*.json'
else:
    glob_expression = f'chunks/{SERIES_NAME}/summaries/*_*.json'

# Collect all summaries into a flat list with their metadata
summaries = []
for file in glob(glob_expression):
    # Extract book and chapter numbers from filename
    filename = os.path.splitext(os.path.basename(file))[0]
    book_number, chapter_number = filename.split('_')
    # Parse each summary chunk from the JSON file
    for chunk_idx, summary in enumerate(json.load(open(file, 'r'))):
        try:
            summaries.append((book_number, chapter_number, chunk_idx, summary))
        except KeyError:
            print(summary)
            raise

print('Getting Chroma client')
# Initialize ChromaDB client and embedding function for vector storage
chroma_client = chromadb.PersistentClient(CHROMA_PATH)
embedder = Embedder()
# Create or get collections for both summaries and full text
collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION, embedding_function=embedder)
full_text_collection = chroma_client.get_or_create_collection(config.chroma_full_text_collection, embedding_function=embedder)

# Initialize lists to store document data for batch processing
ids = []          # Unique identifiers for each chunk
summary_docs = [] # Processed summaries
full_docs = []    # Original full text chunks
meta = []         # Associated metadata

print('Processing summaries to add to collection')
for summary in summaries:
    # Unpack and convert metadata to appropriate types
    book_number, chapter_number, chunk_index, summary_obj = summary
    book_number = int(book_number)
    chapter_number = int(chapter_number)
    chunk_index = int(chunk_index)
    
    # Prepare metadata dictionary with all extracted information
    metadatas = {
        'characters': json.dumps(summary_obj['characters']),  # Serialize nested structures
        'locations': json.dumps(summary_obj['locations']),
        'creatures': json.dumps(summary_obj['creatures']),
        'objects': json.dumps(summary_obj['objects']),
        'full_chunk': summary_obj['full_chunk'],
        'summary': summary_obj['summary'],
        'book_number': book_number,
        'chapter_number': chapter_number,
        'chunk_index': chunk_index,
        'series_metadata_name': SERIES_NAME,
        'series_id': series_info.series_id
    }
    
    # Create unique identifier for this chunk
    ids.append(f'{SERIES_NAME}_{book_number}_{chapter_number}_{chunk_index}')
    summary_docs.append(summary_obj['summary'])
    full_docs.append(summary_obj['full_chunk'])
    meta.append(metadatas)

# Process documents in batches to avoid memory issues
batch_size = 50
for i in tqdm(range(0, len(ids), batch_size), desc='Adding to collection'):
    # Extract current batch
    batch_ids = ids[i:i + batch_size]
    batch_summary_docs = summary_docs[i:i + batch_size]
    batch_full_docs = full_docs[i:i + batch_size]
    batch_summary_meta = meta[i:i + batch_size]
    batch_full_meta = meta[i:i + batch_size]
    
    # Remove redundant fields from metadata
    for summary_meta in batch_summary_meta:
        summary_meta.pop('summary')     # Summary collection doesn't need summary in metadata
    # We leave 'full_chunk' in metadata for the full text collection since we're lowercasing
    #   the document for keyword search purposes.
    
    # Add documents to both collections
    collection.add(ids=batch_ids, documents=batch_summary_docs, metadatas=batch_summary_meta)
    full_text_collection.add(ids=batch_ids, documents=batch_full_docs, metadatas=batch_full_meta)

print(f'Added {len(ids)} documents to collection')