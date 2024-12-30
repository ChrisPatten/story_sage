# Import necessary libraries and modules
import logging  # For logging debug information
from typing import List  # For type annotations
import chromadb  # ChromaDB client for vector storage and retrieval
from chromadb import Documents, EmbeddingFunction, Embeddings
import logging
from sentence_transformers import SentenceTransformer
import torch


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

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', *args, **kwargs):
        """Initializes the StorySageEmbedder with a specified SentenceTransformer model.

        Args:
            model_name (str): Name of the SentenceTransformer model.
        """
        # Call the parent class initializer
        super().__init__(*args, **kwargs)
        # Set up logging for debugging purposes
        self.logger = logging.getLogger(__name__)
        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer(model_name)
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

    def __init__(self, chroma_path: str, chroma_collection_name: str,
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
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        # Get the vector store collection from ChromaDB using the embedder
        self.vector_store = self.chroma_client.get_collection(
            name=chroma_collection_name,
            embedding_function=self.embedder
        )
        # Set the number of chunks to retrieve per query
        self.n_chunks = n_chunks
        # Initialize the logger for this module
        self.logger = logger or logging.getLogger(__name__)

    def retrieve_chunks(self, query_str, context_filters: dict) -> List[str]:
        """Retrieves text chunks relevant to the query and context.

        If no results are found with entity filters, the query is retried
        without those filters to broaden the search.

        Args:
            query_str (str): The user's query string.
            context_filters (dict): Contains 'book_number', 'chapter_number',
                'series_id', and optionally 'entities' (list), which are used
                to filter results.

        Returns:
            List[str]: A list of retrieved documents containing relevant context.
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
            n_results=self.n_chunks,  # Number of results to return
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
                n_results=self.n_chunks,  # Number of results to return
                include=['metadatas', 'documents'],  # Include metadata and documents in the results
                where=fallback_filter  # Apply the fallback filter
            )

        # Log the retrieved documents for debugging purposes
        self.logger.debug(f"Retrieved documents: {query_result}")
        # Return the query results
        return query_result