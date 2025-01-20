# %%
from story_sage import StorySageConfig, StorySageRetriever
from story_sage.utils import Embedder
from story_sage.services.raptor import RaptorProcessor, Chunk, _RaptorResults
import argparse
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


def main(series_metadata_names: list[str], create_chunks: bool = False, all_books: bool = False):
    for series_metadata_name in series_metadata_names:
        print(f"\nProcessing series: {series_metadata_name}")
        
        if all_books:
            file_patterns = [f'./books/{series_metadata_name}/*.txt']
        else:
            book_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            file_patterns = [f'./books/{series_metadata_name}/{str(num).zfill(2)}_*.txt' for num in book_nums]

        os.environ['TOKENIZERS_PARALLELISM'] = "false"

        config_path = './config.yml'
        ssconfig = StorySageConfig.from_file(config_path)
        selected_series = next(series for series in ssconfig.series if series.series_metadata_name == series_metadata_name)

        raptor = RaptorProcessor(config_path=config_path,
                        skip_summarization=False,
                        chunk_size=1000,
                        max_tokens=200,
                        target_dim=5,
                        max_levels=3,
                        max_processes=2,
                        max_summary_threads=3)

        retriever = StorySageRetriever(chroma_path=ssconfig.chroma_path, chroma_collection_name=ssconfig.raptor_collection, n_chunks=15)

        if create_chunks:
            for idx, pattern in enumerate(file_patterns):
                processed_file_name = f'./chunks/{series_metadata_name}/raptor_chunks/{series_metadata_name}.json'
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(processed_file_name), exist_ok=True)
                results: _RaptorResults = raptor.process_texts(pattern)
                raptor.save_chunk_tree(processed_file_name)

        for load_file_name in glob.glob(f'./chunks/{series_metadata_name}/raptor_chunks/{series_metadata_name}*.json.gz'):
            print(f"Loading {load_file_name}")
            retriever.load_processed_files(load_file_name, selected_series.series_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process book series into chunks')
    parser.add_argument('series_metadata_names', type=str, nargs='+',
                      help='Names of the series metadata to process')
    parser.add_argument('--create-chunks', action='store_true', default=False,
                      help='Create new chunks (default: False)')
    parser.add_argument('--all-books', action='store_true', default=False,
                      help='Process all txt files in series directory (default: False)')
    
    args = parser.parse_args()
    main(args.series_metadata_names, args.create_chunks, args.all_books)