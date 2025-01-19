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
from story_sage.models import ChunkMetadata, Chunk
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
        if (chroma_path is not None):
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
                       sort_order: str = None) -> List[Chunk]:
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
        
        #self.logger.debug(f"Query result from retrieve_chunks: {query_result}")
        
        try:
            results: List[Chunk] = Chunk.from_chroma_results(query_result)
        except Exception as e:
            self.logger.error(f"Error converting query results to Chunk objects in retrieve_chunks: {e}")
            raise e
        
        if sort_order and len(results) > 0:
            if sort_order == 'chronological':
                results.sort(key=lambda x: (x.metadata.book_number, x.metadata.chapter_number))
            elif sort_order == 'reverse_chronological':
                results.sort(key=lambda x: (x.metadata.book_number, x.metadata.chapter_number), reverse=True)

        if len(results) < 1:
            self.logger.warning(f'No results found with query {query_str} and filters {combined_filter}')
        # Log the retrieved documents for debugging purposes
        self.logger.debug(f"Retrieved {len(results)} chunks.")
        # Return the query results
        return results
    
    def get_by_keyword(self, keywords: List[str], context_filters: dict) -> List[Chunk]:
        """Retrieves text chunks containing specific keywords within given context."""
        where_filter = self.get_where_filter(context_filters=context_filters)
        
        # Handle empty or invalid keywords
        if not keywords:
            self.logger.warning("No keywords provided for search")
            return []
        
        try:
            # Debug log the input
            self.logger.debug(f"Input keywords type: {type(keywords)}, value: {keywords}")
            
            # Ensure keywords is a list of strings
            if isinstance(keywords, str):
                keywords = [keywords]
            elif not isinstance(keywords, list):
                self.logger.warning(f"Converting keywords of type {type(keywords)} to list")
                keywords = list(keywords)
            
            # Convert each keyword to lowercase string if possible
            processed_keywords = []
            for k in keywords:
                try:
                    if k is not None:
                        if isinstance(k, list):
                            self.logger.error(f"Keyword is a list instead of a string: {k}")
                            continue
                        processed_keyword = str(k).lower()
                        self.logger.debug(f"Processed keyword: {k} -> {processed_keyword}")
                        processed_keywords.append(processed_keyword)
                except Exception as e:
                    self.logger.error(f"Error processing keyword {k}: {e}")
            
            keywords = [k for k in processed_keywords if k]  # Remove empty strings
            
            if not keywords:
                self.logger.warning("No valid keywords after processing")
                return []
                
            # Debug log the final keywords
            self.logger.debug(f"Final processed keywords: {keywords}")
            
            keywords_dict_list = [{'$contains': keyword} for keyword in keywords]
            where_doc = {'$or': keywords_dict_list} if len(keywords_dict_list) > 1 else keywords_dict_list[0]
            
            self.logger.debug(f"Querying with where_filter: {where_filter}")
            self.logger.debug(f"Querying with where_document: {where_doc}")
            
            query_result = self.vector_store.get(
                limit=self.n_chunks,
                include=['metadatas', 'documents'],
                where=where_filter,
                where_document=where_doc
            )

            self.logger.debug(f"Query result from get_by_keyword: {query_result}")
            
            try:
                result: List[Chunk] = Chunk.from_chroma_results(query_result)
            except Exception as e:
                self.logger.error(f"Error converting query results to Chunk objects in get_by_keyword: {e}")
                raise e
            
            #self.logger.debug(f"Formatted result: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in keyword search: {e}", exc_info=True)
            return []

    def get_where_filter(self, context_filters: dict) -> dict:
        """Constructs a filter dictionary for ChromaDB queries."""
        def _safe_add_and(filter: dict, new_item: dict) -> dict:
            if '$and' in filter.keys():
                filter['$and'].append(new_item)
                return filter
            else:
                return {'$and': [filter, new_item]}

        # Extract book number and position/chapter
        book_number = context_filters.get('book_number')
        book_position = context_filters.get('book_position')
        chapter_number = context_filters.get('chapter_number')

        # Check if this is a specific point query
        is_specific_point = context_filters.get('query_type') == 'specific_point'

        # Build filter based on either book_position or chapter_number
        if is_specific_point:
            # For specific points, we want exact matches up to the specified point
            if book_position is not None:
                where_filter = {
                    '$and': [
                        {'book_number': book_number},
                        {'book_position': {'$lt': book_position}}
                    ]
                }
            else:
                where_filter = {
                    '$and': [
                        {'book_number': book_number},
                        {'chapter_number': {'$lt': chapter_number}}
                    ]
                }
        else:
            # Original logic for non-specific point queries
            if book_position is not None:
                where_filter = {
                    '$or': [
                        {'book_number': {'$lt': book_number}},
                        {'$and': [
                            {'book_number': book_number},
                            {'book_position': {'$lt': book_position}}
                        ]}
                    ]
                }
            else:
                where_filter = {
                    '$or': [
                        {'book_number': {'$lt': book_number}},
                        {'$and': [
                            {'book_number': book_number},
                            {'chapter_number': {'$lt': chapter_number}}
                        ]}
                    ]
                }

        # Add additional filters
        if 'series_id' in context_filters:
            where_filter = _safe_add_and(where_filter, {'series_id': int(context_filters.get('series_id'))})

        # Handle flags
        if context_filters.get('summaries_only', False):
            where_filter = _safe_add_and(where_filter, {'is_summary': True})

        if context_filters.get('top_level_only', False):
            where_filter = _safe_add_and(where_filter, {'parents': ''})

        if context_filters.get('exclude_summaries', False):
            where_filter = _safe_add_and(where_filter, {'is_summary': False})
        
        # Handle entity filters
        if len(context_filters.get('entities', [])) > 0:
            for entity_id in context_filters['entities']:
                where_filter = _safe_add_and(where_filter, {'entities': entity_id})

        # Handle temporal constraints
        max_book = context_filters.get('max_book')
        max_position = context_filters.get('max_position')
        max_chapter = context_filters.get('max_chapter')
        
        if max_book is not None:
            where_filter = _safe_add_and(where_filter, {'book_number': {'$lte': max_book}})
            
        if max_position is not None and book_number == max_book:
            where_filter = _safe_add_and(where_filter, {'book_position': {'$lte': max_position}})
        elif max_chapter is not None and book_number == max_book:
            where_filter = _safe_add_and(where_filter, {'chapter_number': {'$lte': max_chapter}})

        return where_filter
    
            
    
    def _recursive_retrieve_from_hierarchy(self, parent_ids: List[str]) -> List[str]:
        """Recursively retrieves the ultimate children of the given parent IDs.

        Args:
            parent_ids (List[str]): List of parent IDs to start the search from.

        Returns:
            List[str]: List of ultimate children IDs.
        """
        #self.logger.debug(f"Recursively retrieving children for parent IDs: {parent_ids}")
        
        children_ids = set()
        for parent_id in parent_ids:
            try:
                result = self.vector_store.get(ids=[parent_id], include=['metadatas'])
                #self.logger.debug(f"Result for parent_id {parent_id}: {result}")
                if result['metadatas'] and result['metadatas'][0].get('children'):
                    child_ids = result['metadatas'][0]['children'].split(',')
                    children_ids.update(self._recursive_retrieve_from_hierarchy(child_ids))
                else:
                    children_ids.add(parent_id)
            except IndexError as e:
                self.logger.error(f"IndexError for parent_id {parent_id}: {e}")
            except Exception as e:
                self.logger.error(f"Error retrieving children for parent_id {parent_id}: {e}")
        
        return list(children_ids)

    def retrieve_from_hierarchy(self, parent_ids: List[str]) -> List[Chunk]:
        """Finds the ultimate children of the provided parent IDs.

        Args:
            parent_ids (List[str]): List of parent IDs to start the search from.

        Returns:
            List[Chunk]: A list of Chunks containing the ultimate children.
        """
        self.logger.debug(f"Retrieving ultimate children from parent IDs: {parent_ids}")
        
        if not parent_ids:
            self.logger.warning("No parent IDs provided")
            return []
            
        if not isinstance(parent_ids, list):
            self.logger.warning(f"Converting parent_ids of type {type(parent_ids)} to list")
            parent_ids = list(parent_ids)
            
        try:
            # Filter out non-string IDs
            valid_ids = [id for id in parent_ids if isinstance(id, str)]
            if not valid_ids:
                self.logger.warning("No valid string IDs found in parent_ids")
                return []
            
            ultimate_children_ids = self._recursive_retrieve_from_hierarchy(valid_ids)
            self.logger.debug(f"Retrieved ultimate children IDs: {ultimate_children_ids}")
            
            if not ultimate_children_ids:
                return []
            
            results = self.vector_store.get(
                ids=ultimate_children_ids,
                include=['metadatas', 'documents']
            )
            
            #self.logger.debug(f"Query result from retrieve_from_hierarchy: {results}")
            
            # Ensure results are in the correct format
            formatted_results = {
                'ids': [results.get('ids', [])],
                'documents': [results.get('documents', [])],
                'metadatas': [results.get('metadatas', [])],
                'distances': [[0.0] * len(results.get('ids', []))]
            }
            
            #self.logger.debug(f"Formatted results: {formatted_results}")
            return Chunk.from_chroma_results(formatted_results)
            
        except Exception as e:
            self.logger.error(f"Error retrieving hierarchy: {e}", exc_info=True)
            return []

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

