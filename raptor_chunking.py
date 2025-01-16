# %%
from story_sage import StorySageConfig, StorySageRetriever
from story_sage.utils import Embedder
from story_sage.utils.raptor import RaptorProcessor, Chunk, _RaptorResults
from openai import OpenAI
import yaml
import httpx
import os
import chromadb
from pprint import pprint
from typing import OrderedDict
import logging
import glob


# Enable debug logging to stdout
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


RERUN = True
SERIES_METADATA_NAME = 'throne_of_glass'
book_nums = [1, 2, 3]
file_patterns = [f'./books/{SERIES_METADATA_NAME}/{str(num).zfill(2)}_*.txt' for num in book_nums]

os.environ['TOKENIZERS_PARALLELISM'] = "false"

config_path = './config.yml'
ssconfig = StorySageConfig.from_file(config_path)
selected_series = next(series for series in ssconfig.series if series.series_metadata_name == SERIES_METADATA_NAME)

raptor = RaptorProcessor(config_path=config_path,
                skip_summarization=False,
                chunk_size=1000,
                max_tokens=200,
                target_dim=5,
                max_levels=3,
                max_processes=2,
                max_summary_threads=3)

retriever = StorySageRetriever(chroma_path=ssconfig.chroma_path, chroma_collection_name=ssconfig.raptor_collection, n_chunks=15)

# %%

if RERUN and __name__ == '__main__':
    for idx, pattern in enumerate(file_patterns):
        processed_file_name = f'./chunks/{SERIES_METADATA_NAME}/raptor_chunks/{SERIES_METADATA_NAME}.json'
        results: _RaptorResults = raptor.process_texts(pattern)
        raptor.save_chunk_tree(processed_file_name)

for load_file_name in glob.glob(f'./chunks/{SERIES_METADATA_NAME}/raptor_chunks/{SERIES_METADATA_NAME}*.json.gz'):
    print(f"Loading {load_file_name}")
    retriever.load_processed_files(load_file_name, selected_series.series_id)