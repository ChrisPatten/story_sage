# Import necessary libraries and modules
import logging  # For logging debug information
from typing import List, Union,TypedDict, Optional  # For type annotations
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb  # ChromaDB client for vector storage and retrieval
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api import Collection
from chromadb.api.types import GetResult, QueryResult
from story_sage.models import Chunk, StorySageConfig, ContextFilters
from story_sage.services.raptor import _RaptorResults, RaptorProcessor
from copy import copy


class StorySageEmbedder(EmbeddingFunction):
    """Embedding function using SentenceTransformer for generating text embeddings.
    
    This class provides an interface to encode text documents into vector representations
    using a SentenceTransformer model. Each document's embedding can then be used for
    similarity searches against other text embeddings.
    
    Attributes:
        model: SentenceTransformer instance used for generating embeddings
        device: torch.device for model computation (GPU/CPU)
        logger: Logger instance for debugging

    Example:
        embedder = StorySageEmbedder()
        embeddings = embedder(["This is a sample text."])
        # Returns: List[List[float]] representing document embeddings
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

    def __call__(self, input: Documents) -> Embeddings:
        """Generates embeddings for the provided documents.

        Args:
            input (Documents): A list of texts to generate embeddings for.

        Returns:
            Embeddings: A list of lists of floats, where each inner list is the
                embedding for a single document.
        """
        # Log the number of texts to embed
        self.logger.info(f"Embedding {len(input)} texts.")
        # Generate embeddings using the model
        embeddings = self.model.encode(input).tolist()
        # Return the embeddings
        return embeddings

class StorySageRetriever:
    """Main retriever class for finding relevant text chunks using vector similarity.

    Manages interactions with ChromaDB to store and query text chunks using vector similarity search.
    Supports filtering by metadata, hierarchical chunk relationships, and various retrieval strategies.

    Attributes:
        embedder (StorySageEmbedder): Embedding function for text vectorization
        chroma_client (chromadb.Client): Client for ChromaDB operations
        vector_store (chromadb.Collection): Collection storing chunks and vectors
        n_chunks (int): Default number of chunks to retrieve per query
        logger (logging.Logger): Logger for debugging information

    Args:
        config (StorySageConfig, optional): Configuration object with ChronaDB settings
        chroma_path (str, optional): Path to ChromaDB storage
        chroma_collection_name (str, optional): Name of ChromaDB collection
        n_chunks (int, optional): Default chunks to retrieve per query (default: 5)
        logger (logging.Logger, optional): Custom logger instance
    """

    def __init__(self, config: StorySageConfig = None, 
                 chroma_path: str = None, chroma_collection_name: str = None,
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
        elif config is not None:
            self.chroma_client = chromadb.PersistentClient(path=config.chroma_path)
        else:
            self.chroma_client = chromadb.EphemeralClient()

        # Get the vector store collection from ChromaDB using the embedder
        if chroma_collection_name is not None:
            collection_name = chroma_collection_name
        elif config is not None:
            collection_name = config.raptor_collection
        else:
            collection_name = 'story_sage_collection'

        self.vector_store = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedder
        )
        # Set the number of chunks to retrieve per query
        if config is not None:
            self.n_chunks = config.n_chunks
        else:
            self.n_chunks = n_chunks

        # Initialize the logger for this module
        self.logger = logger or logging.getLogger(__name__)

    def retrieve_by_similarity(self, query_str: str, context_filters: ContextFilters, 
                             n_results: Optional[int] = None, 
                             sort_order: Optional[str] = None) -> List[Chunk]:
        """Retrieves text chunks relevant to a query using vector similarity search.

        Args:
            query_str: User query to find similar chunks for
            context_filters: Filtering criteria for chunk selection
            n_results: Number of chunks to retrieve (default: self.n_chunks)  
            sort_order: How to order results:
                - 'chronological': Sort by book/chapter ascending
                - 'reverse_chronological': Sort by book/chapter descending
                - None: Sort by similarity score

        Returns:
            List of Chunk objects containing matching text and metadata

        Raises:
            Exception: If chunk conversion fails
        """
        # Log the incoming query and filters for debugging
        self.logger.info(f"Retrieving chunks by similarity for query: {query_str}")
        n_results = n_results or self.n_chunks


        if not isinstance(context_filters, ContextFilters):
            context_filters = ContextFilters(**context_filters)

        combined_filter = self._get_where_filter(context_filters)
        # Log the combined filter being used for the query
        self.logger.debug(f"Combined filter: {combined_filter}")

        # Query the vector store with the combined filter and retrieve the results
        query_result = self.vector_store.query(
            query_texts=[query_str],  # The user's query
            n_results=n_results,  # Number of results to return
            include=['metadatas', 'documents'],  # Include metadata and documents in the results
            where=combined_filter  # Apply the combined filter
        )

        self.logger.info(f"Retrieved {len(query_result['ids'][0])} chunks by similarity")
        
        try:
            results = Chunk.from_chroma_results(query_result)
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
        self.logger.info(f"Retrieved {len(results)} chunks.")
        # Return the query results
        return results
    
    def retrieve_all_with_filters(self, context_filters: Union[ContextFilters, dict]) -> List[Chunk]:
        """Retrieves all chunks matching the given filters without similarity search.

        Args:
            context_filters: ContextFilters object or dict with filter criteria

        Returns:
            List of matching Chunk objects

        Raises:
            Exception: If retrieval or chunk conversion fails
        """
        # Convert dict to ContextFilters if needed
        if not isinstance(context_filters, ContextFilters):
            context_filters = ContextFilters(**context_filters)

        where_filter = self._get_where_filter(context_filters=context_filters)
        
        self.logger.info(f"Retrieving all chunks with filters: {where_filter}")
        try:
            results = self.vector_store.get(
                where=where_filter,
                include=['metadatas']
            )
            return Chunk.from_chroma_results(results)
            
        except Exception as e:
            self.logger.error(f"Error in keyword search: {e}", exc_info=True)
            self.logger.error(f"Results: {results}")
            return []
        
    def retrieve_from_parents(self, parent_ids: List[str]) -> List[Chunk]:
        """Recursively retrieves leaf node chunks starting from given parent IDs.

        Traverses the chunk hierarchy to find ultimate children (leaf nodes) of the
        provided parent chunks.

        Args:
            parent_ids: List of parent chunk IDs to start search from

        Returns:
            List of leaf node Chunk objects descended from parents

        Raises:
            Exception: If hierarchy traversal fails
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
            
            #self.logger.debug(f"Formatted results: {formatted_results}")
            return Chunk.from_chroma_results(results)
            
        except Exception as e:
            self.logger.error(f"Error retrieving hierarchy: {e}", exc_info=True)
            return []

    def _get_where_filter(self, context_filters: ContextFilters) -> dict:
        """Builds ChromaDB filter dictionary from context filters.

        Internal method to convert ContextFilters into ChromaDB where clause format.

        Args:
            context_filters: Filter criteria to convert

        Returns:
            Dict containing ChromaDB-compatible where conditions
        """
        def _safe_add_and(filter: dict, new_item: dict) -> dict:
            if ('$and' in filter.keys()):
                filter['$and'].append(new_item)
                return filter
            else:
                return {'$and': [filter, new_item]}

        # Check if this is a specific point query
        is_specific_point = context_filters.query_type == 'specific_point'

        # Build filter based on either book_position or chapter_number
        if is_specific_point:
            # For specific points, we want exact matches up to the specified point
            if context_filters.book_position is not None:
                where_filter = {
                    '$and': [
                        {'book_number': context_filters.book_number},
                        {'book_position': {'$lt': context_filters.book_position}}
                    ]
                }
            else:
                where_filter = {
                    '$and': [
                        {'book_number': context_filters.book_number},
                        {'chapter_number': {'$lt': context_filters.chapter_number}}
                    ]
                }
        else:
            # Original logic for non-specific point queries
            if context_filters.book_position is not None:
                where_filter = {
                    '$or': [
                        {'book_number': {'$lt': context_filters.book_number}},
                        {'$and': [
                            {'book_number': context_filters.book_number},
                            {'book_position': {'$lt': context_filters.book_position}}
                        ]}
                    ]
                }
            else:
                where_filter = {
                    '$or': [
                        {'book_number': {'$lt': context_filters.book_number}},
                        {'$and': [
                            {'book_number': context_filters.book_number},
                            {'chapter_number': {'$lt': context_filters.chapter_number}}
                        ]}
                    ]
                }

        # Add additional filters
        if context_filters.series_id is not None:
            where_filter = _safe_add_and(where_filter, {'series_id': int(context_filters.series_id)})

        if context_filters.level is not None:
            where_filter = _safe_add_and(where_filter, {'level': int(context_filters.level)})

        # Handle flags
        if context_filters.summaries_only:
            where_filter = _safe_add_and(where_filter, {'is_summary': True})

        if context_filters.top_level_only:
            where_filter = _safe_add_and(where_filter, {'parents': ''})

        if context_filters.exclude_summaries:
            where_filter = _safe_add_and(where_filter, {'is_summary': False})
        return where_filter
            
    def _recursive_retrieve_from_hierarchy(self, parent_ids: List[str]) -> List[str]:
        """Recursively retrieves the ultimate children of the given parent IDs.

        Args:
            parent_ids (List[str]): List of parent IDs to start the search from.

        Returns:
            List[str]: List of ultimate children IDs.
        """
        #self.logger.debug(f"Recursively retrieving children for parent IDs: {parent_ids}")
        #print(f"Recursively retrieving children for parent IDs: {parent_ids}")
        children_ids = set(parent_ids)

        results = self.vector_store.get(ids=parent_ids, include=['metadatas'])
        
        for meta in results['metadatas']:
            child_ids = meta['children'].split(',')
            if len(child_ids) > 0:
                children_ids.update(self._recursive_retrieve_from_hierarchy(child_ids))


        return list(children_ids)

    def load_processed_files(self, chunk_tree: Union[_RaptorResults, str], 
                           series_id: int) -> None:
        """Loads processed text chunks into the vector store.

        Takes output from RaptorProcessor or a serialized chunk tree file and stores
        chunks in ChromaDB with their embeddings and metadata.

        Args:
            chunk_tree: RaptorProcessor results or path to JSON/JSON.gz file
            series_id: Identifier for the book series

        Raises:
            ValueError: If chunk metadata is incomplete
            Exception: If chunk insertion fails
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
                        metadata = {}
                        metadata['book_filename'] = book_filename
                        metadata['is_summary'] = len(chunk.children) == 0
                        metadata['parents'] = ','.join(chunk.parents)
                        metadata['children'] = ','.join(chunk.children)
                        metadata['full_chunk'] = chunk.text
                        metadata['series_id'] = series_id
                        metadatas.append(metadata)
                        
                        if chunk.embedding is not None:
                            embeddings.append(chunk.embedding.tolist())
                    
                    if not ids:  # Skip if no new chunks to add
                        continue

                    for metadata in metadatas:
                        if any(value is None for value in metadata.values()):
                            _ = metadata.pop('full_chunk', None)
                            self.logger.error(f"Missing metadata in chunk: {metadata}")
                            raise ValueError("Missing metadata in chunk")
                        
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

