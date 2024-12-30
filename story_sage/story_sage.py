import logging
import yaml
import uuid
from typing import Tuple, Optional, Dict, List
from .data_classes.story_sage_state import StorySageState
from .vector_store import StorySageRetriever
from .story_sage_chain import StorySageChain
from .story_sage_entity import StorySageEntityCollection
from .data_classes.story_sage_series import StorySageSeries

class ConditionalRequestIDFormatter(logging.Formatter):
    """Custom formatter to include request_id only if it's not None."""
    
    def format(self, record):
        if hasattr(record, 'request_id') and record.request_id:
            original_msg = super().format(record)
            return f"{original_msg} [Request ID: {record.request_id}]"
        return super().format(record)

class StorySage:
    """
    Main class for the Story Sage system that helps readers track story elements.
    Coordinates between the retriever, chain, and state management components.
    """

    def __init__(self, api_key: str, chroma_path: str, chroma_collection_name: str, 
                 entities_dict: dict[str, StorySageEntityCollection],
                 series_list: List[dict] = [], n_chunks: int = 5):
        """
        Initialize the StorySage instance with necessary components and configuration.

        Args:
            api_key: API key for accessing external services
            chroma_path: Path to the chroma database
            chroma_collection_name: Name of the chroma collection
            entities_dict: Dictionary of {<series_metadata_name>: <StorySageEntityCollection>}
            series_list: List of series information
            n_chunks: Number of chunks for the retriever to process
        """
        # Set up logging
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = True  # Allow logs to root logger
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = ConditionalRequestIDFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

        # Initialize request_id
        self.request_id = None

        self.entities = {key: StorySageEntityCollection.from_dict(value) for key, value in entities_dict.items()}

        # Initialize series info
        self.series_list = [StorySageSeries.from_dict(series) for series in series_list]

        # Create a LoggerAdapter to include class attributes
        self.logger = logging.LoggerAdapter(self._logger, {'request_id': self.request_id})

        # Initialize retriever and chain components
        self.retriever = StorySageRetriever(chroma_path, chroma_collection_name, n_chunks)
        self.chain = StorySageChain(api_key, self.entities, self.series_list, self.retriever, self.logger)
        

    def invoke(self, question: str, book_number: int = None, 
               chapter_number: int = None, series_id: int = None) -> Tuple[str, List[str]]:
        """
        Process a user's question about the story with optional context parameters.

        Args:
            question: The user's question about the story
            book_number: Optional book number for context filtering
            chapter_number: Optional chapter number for context filtering
            series_id: Optional series ID for context filtering

        Returns:
            Tuple containing (answer, context_list, request_id)
        """
        self.logger.info(f"Processing question: {question}")
        
        # Generate and set request_id
        self.request_id = str(uuid.uuid4())
        self.logger = logging.LoggerAdapter(self._logger, {'request_id': self.request_id})
        self.logger.debug(f"Set request_id to {self.request_id}")
        self.chain.logger = self.logger

        # Initialize state with default values
        state = StorySageState(
            question=question,
            book_number=book_number or 100,  # Default to end of series if not specified
            chapter_number=chapter_number or 0,
            series_id=series_id or 0,
            context=None,
            answer=None,
            entities=[]
        )

        try:
            # Process the question through the chain
            result = self.chain.graph.invoke(state)
            
            # Log the results
            self.logger.debug(f"Generated answer: {result['answer']}")
            self.logger.debug(f"Retrieved context: {result['context']}")
            
            return result['answer'], result['context'], self.request_id
            
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            raise e

# Example usage:
# Initialize StorySage with required parameters
# story_sage = StorySage(api_key="your_api_key", chroma_path="path/to/chroma", chroma_collection_name="collection_name")

# Invoke the system with a question
# answer, context, request_id = story_sage.invoke(question="Who is the main character?", book_number=1, chapter_number=1, series_id=1)

# Example result:
# answer: "The main character is Rand al'Thor."
# context: ["Rand al'Thor is introduced in the first chapter of the first book."]
# request_id: "123e4567-e89b-12d3-a456-426614174000"