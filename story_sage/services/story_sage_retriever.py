# Import necessary libraries and modules
import logging  # For logging debug information
from typing import List, Union, Tuple  # For type annotations
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb  # ChromaDB client for vector storage and retrieval
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api import Collection
from chromadb.api.types import GetResult, QueryResult
from story_sage.utils import Chunk, ChunkMetadata
from story_sage.utils.raptor import _RaptorResults, RaptorProcessor
from copy import copy


class StorySageEmbedder(EmbeddingFunction):
    """Embedding function using SentenceTransformer for generating text embeddings.
    
    This class provides an interface to encode text documents into vector representations
    using a SentenceTransformer model. Each document's embedding can then be used for
    similarity searches against other text embeddings.

    Example usage:
        embedder = StorySageEmbedder(model_name='all-MiniLM-L6-v2')
        embeddings = embedder(["This is a sample text.", "Another text snippet."])
        # embeddings: [[...], [...]] (lists of float values)

    Example result:
        [
            [
                -0.0072062592, 0.01234567, ... (embedding vector)
            ],
            [
                0.008765459, -0.0212345, ... (embedding vector)
            ]
        ]
    """

    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v1', *args, **kwargs):
        """Initializes the StorySageEmbedder with a specified SentenceTransformer model.

        Args:
            model_name (str): Name of the SentenceTransformer model.
        """
        # Call the parent class initializer
        super().__init__(*args, **kwargs)
        # Set up logging for debugging purposes
        self.logger = logging.getLogger(__name__)
        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer(model_name, local_files_only=True)
        # Determine the device to run the model on (GPU if available, else CPU)
        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        # Move the model to the selected device
        self.model = self.model.to(self.device)

        if 'log_level' in kwargs:
            self.logger.setLevel(kwargs['log_level'])

    def __call__(self, input: Documents) -> Embeddings:
        """Generates embeddings for the provided documents.

        Args:
            input (Documents): A list of texts to generate embeddings for.

        Returns:
            Embeddings: A list of lists of floats, where each inner list is the
                embedding for a single document.
        """
        # Log the number of texts to embed
        self.logger.debug(f"Embedding {len(input)} texts.")
        # Generate embeddings using the model
        embeddings = self.model.encode(input).tolist()
        # Log that embedding is completed
        self.logger.debug("Embedding completed.")
        # Return the embeddings
        return embeddings


class StorySageRetriever:
    """Retrieves relevant text chunks from a vector store based on the user's query.

    This class is responsible for interacting with a ChromaDB vector store to
    find relevant text chunks for a given query. It filters based on various
    metadata (such as book number, chapter number, series ID, etc.) to narrow
    down the most contextually relevant matches.

    Example usage:
        retriever = StorySageRetriever(
            chroma_path='/path/to/chroma',
            chroma_collection_name='my_collection',
            n_chunks=5
        )
        results = retriever.retrieve_chunks(
            query_str='What is the significance of the lost sword?',
            context_filters={
                'book_number': 2,
                'chapter_number': 15,
                'series_id': 1,
                'entities': ['some_character_id']
            }
        )

    Example result:
        [
            {
                'documents': ['The lost sword belonged to ...'],
                'metadatas': [{'book_number': 1, 'chapter_number': 10, ...}]
            },
            ...
        ]
    """

    def __init__(self, chroma_path: str = None, chroma_collection_name: str = None,
                 n_chunks: int = 5, logger: logging.Logger = None):
        """Initializes the StorySageRetriever with necessary configuration.

        Args:
            chroma_path (str): Path to the Chroma database.
            chroma_collection_name (str): Name of the Chroma collection to use.
            n_chunks (int, optional): Number of chunks to retrieve per query.
                Defaults to 5.
            logger (logging.Logger, optional): Optional logger for debugging.
                Defaults to None.
        """
        # Initialize the embedding function using StorySageEmbedder
        self.embedder = StorySageEmbedder()
        # Set up the ChromaDB client with persistent storage at the specified path
        if chroma_path is not None:
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        else:
            self.chroma_client = chromadb.EphemeralClient()
        # Get the vector store collection from ChromaDB using the embedder
        chroma_collection_name = chroma_collection_name or 'story_sage_collection'
        self.vector_store = self.chroma_client.get_or_create_collection(
            name=chroma_collection_name,
            embedding_function=self.embedder
        )
        # Set the number of chunks to retrieve per query
        self.n_chunks = n_chunks
        # Initialize the logger for this module
        self.logger = logger or logging.getLogger(__name__)

    def retrieve_chunks(self, query_str, context_filters: dict, n_results: int = None, 
                       sort_order: str = None) -> QueryResult:
        """Retrieves text chunks relevant to the query and context.

        Args:
            query_str (str): The user's query string
            context_filters (dict): Filters to narrow down the search scope
            n_results (int, optional): Number of results to return
            sort_order (str, optional): How to sort results:
                - 'chronological': Sort by book/chapter ascending
                - 'reverse_chronological': Sort by book/chapter descending
                - None: Sort by similarity score (default)
        """
        # Log the incoming query and filters for debugging
        self.logger.debug(f"Retrieving chunks with query: {query_str}, context_filters: {context_filters}")
        n_results = n_results or self.n_chunks

        if not context_filters:
            raise ValueError("Context filters are required to retrieve relevant chunks.")

        combined_filter = self.get_where_filter(context_filters)
        # Log the combined filter being used for the query
        self.logger.debug(f"Combined filter: {combined_filter}")

        # Query the vector store with the combined filter and retrieve the results
        query_result = self.vector_store.query(
            query_texts=[query_str],  # The user's query
            n_results=n_results,  # Number of results to return
            include=['metadatas', 'documents'],  # Include metadata and documents in the results
            where=combined_filter  # Apply the combined filter
        )

        # Apply temporal sorting if requested
        if sort_order and len(query_result['ids'][0]) > 0:
            results_with_meta = list(zip(
                query_result['ids'][0],
                query_result['documents'][0],
                query_result['metadatas'][0],
                query_result['distances'][0]
            ))
            
            if sort_order == 'chronological':
                results_with_meta.sort(key=lambda x: (x[2]['book_number'], x[2]['chapter_number']))
            elif sort_order == 'reverse_chronological':
                results_with_meta.sort(key=lambda x: (x[2]['book_number'], x[2]['chapter_number']), reverse=True)
                
            # Reconstruct query result
            query_result['ids'] = [[x[0] for x in results_with_meta]]
            query_result['documents'] = [[x[1] for x in results_with_meta]]
            query_result['metadatas'] = [[x[2] for x in results_with_meta]]
            query_result['distances'] = [[x[3] for x in results_with_meta]]

        if len(query_result['ids'][0]) < 1:
            self.logger.warning(f'No results found with query {query_str} and filters {combined_filter}')
        # Log the retrieved documents for debugging purposes
        self.logger.debug(f"Retrieved documents: {query_result['ids']}")
        # Return the query results
        return query_result
    
    def get_by_keyword(self, keywords: List[str], context_filters: dict) -> QueryResult:
        """Retrieves text chunks containing specific keywords within given context.

        Performs an exact keyword match search within the vector store, considering
        the provided context filters. Keywords are case-insensitive.

        Args:
            keywords (List[str]): List of keywords to search for (e.g., ["sword", "castle"])
            context_filters (dict): Filters to narrow down the search scope, containing:
                - book_number (int): The current book number
                - chapter_number (int): The current chapter number
                - series_id (int): The ID of the book series
                - entities (list, optional): List of entity IDs to filter by

        Returns:
            QueryResult: A ChromaDB query result containing:
                - documents: List of text chunks containing the keywords
                - metadatas: List of metadata for each chunk
                - ids: Unique IDs for each chunk

        Example:
            >>> context = {'series_id': 1, 'book_number': 2, 'chapter_number': 3}
            >>> result = retriever.get_by_keyword(['sword', 'castle'], context)
            >>> print(result['documents'])  # Texts containing 'sword' or 'castle'
        """
        where_filter = self.get_where_filter(context_filters=context_filters)
        keywords_dict_list = [{'$contains': str.lower(keyword)} for keyword in keywords]
        if len(keywords_dict_list) > 1:
            where_doc = {'$or': keywords_dict_list}
        else:
            where_doc = keywords_dict_list[0]

        query_result = self.vector_store.get(
            limit=self.n_chunks,
            include=['metadatas', 'documents'],
            where=where_filter,
            where_document=where_doc
        )

        self.logger.debug(f"Retrieved documents: {query_result['ids']}")
        return query_result
        
        
    
    def get_where_filter(self, context_filters: dict) -> dict:
        """Constructs a filter dictionary for ChromaDB queries.

        Creates a complex filter that ensures retrieved chunks are from either:
        1. Earlier books in the series
        2. Earlier chapters in the same book
        3. Match the specified entity filters (if include_entities is True)

        Args:
            context_filters (dict): Dictionary containing filtering criteria
            include_entities (bool): Whether to include entity filters in the result

        Returns:
            dict: A nested dictionary structure compatible with ChromaDB's where clause

        Example:
            >>> filters = {
            ...     'series_id': 1,
            ...     'book_number': 2,
            ...     'chapter_number': 3,
            ...     'entities': ['character_123']
            ... }
            >>> filter_dict = retriever.get_where_filter(filters)
            >>> print(filter_dict)
            # Outputs a complex nested dictionary for ChromaDB filtering
        """

        def _safe_add_and(filter: dict, new_item: dict) -> dict:
            if '$and' in filter.keys():
                filter['$and'].append(new_item)
                return filter
            else:
                return {'$and': [filter, new_item]}

        # Extract book and chapter numbers, if present
        book_number = context_filters.get('book_number')
        chapter_number = context_filters.get('chapter_number')

        # Build a filter to retrieve documents from earlier books or chapters
        where_filter = {
            '$or': [
                {'book_number': {'$lt': book_number}},  # Books before the current one
                {'$and': [  # Chapters before the current one in the same book
                    {'book_number': book_number},
                    {'chapter_number': {'$lt': chapter_number}}
                ]}
            ]
        }

        # Add additional filters as necessary
        if 'series_id' in context_filters:
            where_filter = _safe_add_and(where_filter, {'series_id': int(context_filters.get('series_id'))})

        # The following filters act as flags (default to False unless set otherwise)
        if context_filters.get('summaries_only', False):
            where_filter = _safe_add_and(where_filter, {'is_summary': True})

        if context_filters.get('top_level_only', False):
            where_filter = _safe_add_and(where_filter, {'parents': ''})

        if context_filters.get('exclude_summaries', False):
            where_filter = _safe_add_and(where_filter, {'is_summary': False})
        
        # Build filters based on entities like people, places, groups, and animals
        entity_filters = []
        if len(context_filters.get('entities', [])) > 0:
            for entity_id in context_filters['entities']:
                where_filter = _safe_add_and(where_filter, {'entities': entity_id})

        # Handle temporal constraints
        max_book = context_filters.get('max_book')
        max_chapter = context_filters.get('max_chapter')
        
        if max_book is not None:
            where_filter = _safe_add_and(where_filter, {'book_number': {'$lte': max_book}})
            
        if max_chapter is not None and book_number == max_book:
            where_filter = _safe_add_and(where_filter, {'chapter_number': {'$lte': max_chapter}})

        return where_filter
    
            
    
    def _recursive_retrieve_from_hierarchy(self, parent_ids: List[str]) -> List[str]:
        """Recursively retrieves the ultimate children of the given parent IDs.

        Args:
            parent_ids (List[str]): List of parent IDs to start the search from.

        Returns:
            List[str]: List of ultimate children IDs.
        """
        self.logger.debug(f"Recursively retrieving children for parent IDs: {parent_ids}")
        
        children_ids = set()
        for parent_id in parent_ids:
            result = self.vector_store.get(ids=[parent_id], include=['metadatas'])
            if result['metadatas'][0]['children']:
                child_ids = result['metadatas'][0]['children'].split(',')
                children_ids.update(self._recursive_retrieve_from_hierarchy(child_ids))
            else:
                children_ids.add(parent_id)
        
        return list(children_ids)

    def retrieve_from_hierarchy(self, parent_ids: List[str]) -> QueryResult:
        """Finds the ultimate children of the provided parent IDs.

        Args:
            parent_ids (List[str]): List of parent IDs to start the search from.

        Returns:
            QueryResult: A ChromaDB query result containing:
                - documents: List of text chunks containing the ultimate children
                - metadatas: List of metadata for each chunk
                - ids: Unique IDs for each chunk
        """
        self.logger.debug(f"Retrieving ultimate children from parent IDs: {parent_ids}")
        
        ultimate_children_ids = self._recursive_retrieve_from_hierarchy(parent_ids)
        
        results = self.vector_store.get(
            ids=ultimate_children_ids,
            include=['metadatas', 'documents']
        )
        
        self.logger.debug(f"Retrieved ultimate children documents: {results['ids']}")
        return results

    def load_processed_files(self, chunk_tree: Union[_RaptorResults, str], 
                             series_id: int) -> None:
        """Loads processed files from RaptorProcessor into the vector store.

        Takes the output from RaptorProcessor's process_texts() method or a JSON file path
        and inserts all chunks into the ChromaDB collection, preserving embeddings 
        and metadata.

        Args:
            chunk_tree: Either:
                - _RaptorResults: Direct output from RaptorProcessor
                - str: Path to JSON or JSON.gz file

        Example:
            >>> processor = RaptorProcessor('config.yaml')
            >>> results = processor.process_texts('./books/*.txt')
            >>> retriever = StorySageRetriever()
            >>> retriever.load_processed_files(results)  # From RaptorResults
            >>> retriever.load_processed_files('chunks.json.gz')  # From compressed JSON
            >>> retriever.load_processed_files('chunks.json')  # From JSON file
        """
        self.logger.debug("Loading processed files into vector store")
        
        raptor_chunks: _RaptorResults = None

        # Handle different input types
        if isinstance(chunk_tree, str):
            raptor_chunks = RaptorProcessor.load_chunk_tree(chunk_tree)
        else:
            raptor_chunks = chunk_tree
        
        # Process the chunk tree
        for book_filename, book_data in raptor_chunks.items():
            for chapter_key, chapter_data in book_data.items():
                for level_key, chunks in chapter_data.items():
                    # Prepare batch data
                    ids = []
                    documents = []
                    metadatas = []
                    embeddings = []
                    
                    for chunk in chunks:
                        # Skip if this chunk is already in the collection
                        try:
                            if chunk.chunk_key in self.vector_store.get(ids=[chunk.chunk_key])['ids']:
                                continue
                        except Exception:  # Handle case where chunk doesn't exist
                            raise

                        if not chunk.chunk_key.startswith('series_'):
                            chunk.chunk_key = f'series_{series_id}|{chunk.chunk_key}'
                        child_chunks: List[str] = copy(chunk.children)
                        for idx, child_key in enumerate(child_chunks):
                            if not child_key.startswith('series_'):
                                child_key = f'series_{series_id}|{child_key}'
                                chunk.children[idx] = child_key
                        parent_chunks: List[str] = copy(chunk.parents)
                        for idx, parent_key in enumerate(parent_chunks):
                            if not parent_key.startswith('series_'):
                                parent_key = f'series_{series_id}|{parent_key}'
                                chunk.parents[idx] = parent_key
                            
                        ids.append(chunk.chunk_key)
                        documents.append(str.lower(chunk.text))
                        
                        # Prepare metadata
                        metadata = chunk.metadata.__dict__.copy()
                        metadata['book_filename'] = book_filename
                        metadata['is_summary'] = chunk.is_summary
                        metadata['parents'] = ','.join(chunk.parents)
                        metadata['children'] = ','.join(chunk.children)
                        metadata['full_chunk'] = chunk.text
                        metadata['series_id'] = series_id
                        metadatas.append(metadata)
                        
                        if chunk.embedding is not None:
                            embeddings.append(chunk.embedding.tolist())
                    
                    if not ids:  # Skip if no new chunks to add
                        continue
                        
                    # Add to collection with or without embeddings
                    try:
                        if embeddings:
                            self.vector_store.upsert(
                                ids=ids,
                                documents=documents,
                                metadatas=metadatas,
                                embeddings=embeddings
                            )
                        else:
                            self.vector_store.upsert(
                                ids=ids,
                                documents=documents,
                                metadatas=metadatas
                            )
                        
                        self.logger.debug(f"Added {len(ids)} chunks from {book_filename} {chapter_key} {level_key}")
                    except Exception as e:
                        self.logger.error(f"Error adding chunks: {e}")
                        raise

