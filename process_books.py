import os
import json
from story_sage.utils.chunker import StorySageChunker
from tqdm import tqdm
import argparse
from openai import OpenAI
from pydantic import BaseModel
from story_sage.data_classes.story_sage_config import StorySageConfig
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
    character: str
    actions: str

class ChunkSummary(BaseModel):
    summary: str
    characters: list[CharacterActions]
    locations: list[str]
    creatures: list[str]
    objects: list[str]

def get_summary(text: str):
    """Gets a summary of the text using GPT-4o"""
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

if not SKIP_CHUNKING:
    summaries: List[Tuple[str, str, int, dict]] = []
    usage = []
    for pattern in file_patterns:
        text_dict = chunker.read_text_files(pattern)
        print(f'Processing {pattern}')
        book_summaries: List[Tuple[str, str, int, dict]] = []
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

        summaries.extend(book_summaries)
else:
    print('Skipping chunking')
    glob_expression = f'chunks/{SERIES_NAME}/summaries/[{",".join(BOOK_NUMBERS)}]_*.json'
    
    summaries = []
    for file in glob(glob_expression):
        filename = os.path.splitext(os.path.basename(file))[0]
        book_number, chapter_number = filename.split('_')
        for chunk_idx, summary in enumerate(json.load(open(file, 'r'))):
            try:
                summaries.append((book_number, chapter_number, chunk_idx, summary))
            except KeyError:
                print(summary)
                raise

print('Getting Chroma client')
chroma_client = chromadb.PersistentClient(CHROMA_PATH)
embedder = Embedder()
collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION, embedding_function=embedder)

ids = []
docs = []
meta = []

print('Processing summaries to add to collection')
for summary in summaries:
    book_number, chapter_number, chunk_index, summary_obj = summary
    book_number = int(book_number)
    chapter_number = int(chapter_number)
    chunk_index = int(chunk_index)
    metadatas = {
        'characters': json.dumps(summary_obj['characters']),
        'locations': json.dumps(summary_obj['locations']),
        'creatures': json.dumps(summary_obj['creatures']),
        'objects': json.dumps(summary_obj['objects']),
        'full_chunk': summary_obj['full_chunk'],
        'book_number': book_number,
        'chapter_number': chapter_number,
        'chunk_index': chunk_index,
        'series_metadata_name': SERIES_NAME,
        'series_id': series_info.series_id
    }
    ids.append(f'{SERIES_NAME}_{book_number}_{chapter_number}_{chunk_index}')
    docs.append(summary_obj['summary'])
    meta.append(metadatas)


batch_size = 50
for i in tqdm(range(0, len(ids), batch_size), desc='Adding to collection'):
    batch_ids = ids[i:i + batch_size]
    batch_docs = docs[i:i + batch_size]
    batch_meta = meta[i:i + batch_size]
    #collection.delete(ids=batch_ids)
    collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)

print(f'Added {len(ids)} documents to collection')