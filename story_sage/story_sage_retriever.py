# Import necessary libraries and modules
import logging  # For logging debug information
from .story_sage_embedder import StorySageEmbedder  # Custom embedder class for text embeddings
from typing import List  # For type annotations
import chromadb  # ChromaDB client for vector storage and retrieval

class StorySageRetriever:
    """Class responsible for retrieving relevant chunks of text based on the user's query."""

    def __init__(self, chroma_path: str, chroma_collection_name: str,
                 entities: dict, n_chunks: int = 5):
        """
        Initialize the StorySageRetriever instance.

        Args:
            chroma_path (str): Path to the Chroma database.
            chroma_collection_name (str): Name of the Chroma collection.
            entities (dict): Dictionary containing character and entity information.
            n_chunks (int, optional): Number of chunks to retrieve per query. Defaults to 5.
        """
        # Initialize the embedding function using StorySageEmbedder
        self.embedder = StorySageEmbedder()
        # Set up the ChromaDB client with persistent storage at the specified path
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        # Get the vector store collection from ChromaDB using the embedder
        self.vector_store = self.chroma_client.get_collection(
            name=chroma_collection_name,
            embedding_function=self.embedder
        )
        # Store entities for filtering during retrieval
        self.entities = entities
        # Set the number of chunks to retrieve per query
        self.n_chunks = n_chunks
        # Initialize the logger for this module
        self.logger = logging.getLogger(__name__)

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

        combined_filter = {}  # Initialize the combined filter dictionary

        # Extract the series ID from context filters
        series_id = context_filters['series_id']

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
        combined_filter = {'$and': [combined_filter, book_chapter_filter]} if combined_filter else book_chapter_filter

        # Build filters based on entities like people, places, groups, and animals
        entity_filters = []
        for entity_type in ['people', 'places', 'groups', 'animals']:
            if entity_type in context_filters and context_filters[entity_type]:
                for entity in context_filters[entity_type]:
                    # Add a filter for each entity, ensuring they are included in the metadata
                    entity_filters.append({entity: True})

        # Combine entity filters if any exist
        if entity_filters and False: # Disable entity filtering for now while I fix entity lookup
            if len(entity_filters) == 1:
                entity_meta_filter = entity_filters[0]
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
            n_results=self.n_chunks,  # Number of results to return
            include=['metadatas', 'documents'],  # Include metadata and documents in the results
            where=combined_filter  # Apply the combined filter
        )

        # Log the retrieved documents for debugging purposes
        self.logger.debug(f"Retrieved documents: {query_result}")
        # Return the query results
        return query_result