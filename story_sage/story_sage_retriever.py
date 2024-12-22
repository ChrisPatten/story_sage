# Import necessary libraries and modules
import logging  # For logging debug information
from .story_sage_embedder import StorySageEmbedder  # Custom embedder class for text embeddings
from .story_sage_config import StorySageConfig  # Configuration class for Story Sage
from typing import List  # For type annotations
import chromadb  # ChromaDB client for vector storage and retrieval
import logging

class StorySageRetriever:
    """
    A class to retrieve story elements and context for the Story Sage system.

    Attributes:
        config (dict): Configuration dictionary.
        series_collection (list): List of StorySageSeries objects.
        logger (logging.LoggerAdapter): Logger for logging messages.
    """

    def __init__(self, config: StorySageConfig, series_collection: list, logger: logging.LoggerAdapter):
        """
        Initialize the StorySageRetriever with the given configuration and series collection.

        Args:
            config (dict): Configuration dictionary.
            series_collection (list): List of StorySageSeries objects.
            logger (logging.LoggerAdapter): Logger for logging messages.
        """
        self.config = config
        self.series_collection = series_collection
        self.logger = logger

        # Initialize the embedding function using StorySageEmbedder
        self.embedder = StorySageEmbedder()
        # Set up the ChromaDB client with persistent storage at the specified path
        self.chroma_client = chromadb.PersistentClient(path=self.config.chroma_path)
        # Get the vector store collection from ChromaDB using the embedder
        self.vector_store = self.chroma_client.get_collection(
            name=self.config.chroma_collection,
            embedding_function=self.embedder
        )

    

    def retrieve_chunks(self, query_str, context_filters: dict) -> List[str]:
        """
        Retrieve chunks of text relevant to the query and filtering parameters.

        Args:
            query_str (str): The user's query.
            context_filters (dict): Dictionary containing context filters such as entities and book details.

        Returns:
            List[str]: Retrieved documents containing relevant context.
        """
        # Log the incoming query and filters for debugging
        self.logger.debug(f"Retrieving chunks with query: {query_str}, context_filters: {context_filters}")

        combined_filter = {'series_id': int(context_filters.get('series_id'))}  # Initialize the combined filter dictionary

        # Extract book and chapter numbers, if present
        book_number = context_filters.get('book_number')
        chapter_number = context_filters.get('chapter_number')

        # Build a filter to retrieve documents from earlier books or chapters
        book_chapter_filter = {
            '$or': [
                {'book_number': {'$lt': book_number}},  # Books before the current one
                {'$and': [  # Chapters before the current one in the same book
                    {'book_number': book_number},
                    {'chapter_number': {'$lt': chapter_number}}
                ]}
            ]
        }

        # Combine filters for book and chapter
        combined_filter = {'$and': [combined_filter, book_chapter_filter]}

        # Create a fallback filter if we need to requery without tags
        fallback_filter = combined_filter.copy()

        # Build filters based on entities like people, places, groups, and animals
        entity_filters = []
        if len(context_filters.get('entities', [])) > 0:
            for entity_id in context_filters['entities']:
                entity_filter = {entity_id: True}
                entity_filters.append(entity_filter)

        # Combine entity filters if any exist
        if len(entity_filters) == 1:
            entity_meta_filter = entity_filters[0]
        elif len(entity_filters) == 0:
            pass
        else:
            # Use an '$and' clause to require all entity conditions
            entity_meta_filter = {'$and': entity_filters}
            # Add entity filters to the combined filter
            combined_filter = {'$and': [combined_filter, entity_meta_filter]} if combined_filter else entity_meta_filter

        # Log the combined filter being used for the query
        self.logger.debug(f"Combined filter: {combined_filter}")

        # Query the vector store with the combined filter and retrieve the results
        query_result = self.vector_store.query(
            query_texts=[query_str],  # The user's query
            n_results=self.config.n_chunks,  # Number of results to return
            include=['metadatas', 'documents'],  # Include metadata and documents in the results
            where=combined_filter  # Apply the combined filter
        )

        # If no results are found, retry the query without entity filters
        if len(query_result) == 0:
            # Log that we are retrying the query without entity filters
            self.logger.debug("Retrying query without entity filters")
            # Query the vector store again without entity filters
            query_result = self.vector_store.query(
                query_texts=[query_str],  # The user's query
                n_results=self.config.n_chunks,  # Number of results to return
                include=['metadatas', 'documents'],  # Include metadata and documents in the results
                where=fallback_filter  # Apply the fallback filter
            )

        # Log the retrieved documents for debugging purposes
        self.logger.debug(f"Retrieved documents: {query_result}")
        # Return the query results
        return query_result

    def retrieve_summary_chunks(self, query_str, context_filters: dict) -> List[str]:
        """
        Retrieve summary chunks relevant to the query and filtering parameters.

        Args:
            query_str (str): The user's query.
            context_filters (dict): Dictionary containing context filters such as entities and book details.

        Returns:
            List[str]: Retrieved documents containing relevant context.
        """
        # Log the incoming query and filters for debugging
        self.logger.debug(f"Retrieving summary chunks with query: {query_str}, context_filters: {context_filters}")

        combined_filter = {'series_id': int(context_filters.get('series_id'))}  # Initialize the combined filter dictionary

        # Extract book and chapter numbers, if present
        book_number = context_filters.get('book_number')
        chapter_number = context_filters.get('chapter_number')

        # Build a filter to retrieve documents from earlier books or chapters
        book_chapter_filter = {
            '$or': [
                {'book_number': {'$lt': book_number}},  # Books before the current one
                {'$and': [  # Chapters before the current one in the same book
                    {'book_number': book_number},
                    {'chapter_number': {'$lt': chapter_number}}
                ]}
            ]
        }

        # Combine filters for book and chapter
        combined_filter = {'$and': [combined_filter, book_chapter_filter]}

        # Log the combined filter being used for the query
        self.logger.debug(f"Combined filter: {combined_filter}")

        # Query the vector store with the combined filter and retrieve the results
        query_result = self.vector_store.query(
            query_texts=[query_str],  # The user's query
            n_results=self.config.n_chunks,  # Number of results to return
            include=['metadatas', 'documents'],  # Include metadata and documents in the results
            where=combined_filter  # Apply the combined filter
        )

        # Log the retrieved documents for debugging purposes
        self.logger.debug(f"Retrieved summary documents: {query_result}")
        # Return the query results
        return query_result

 