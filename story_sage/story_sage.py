import logging
import yaml
import uuid
from typing import Tuple, Optional, Dict, List
from .story_sage_state import StorySageState
from .story_sage_retriever import StorySageRetriever
from .story_sage_chain import StorySageChain

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
                 entities: dict, series_yml_path: str, n_chunks: int = 5):
        """Initialize the StorySage instance with necessary components and configuration."""
        # Set up logging
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = True  # Allow logs to root logger

        # Create handler for logger
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)

        # Define the custom formatter
        formatter = ConditionalRequestIDFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add the handler to the logger
        self._logger.addHandler(handler)

        # Initialize request_id
        self.request_id = None

        # Create a LoggerAdapter to include class attributes
        self.logger = logging.LoggerAdapter(self._logger, {'request_id': self.request_id})

        self.retriever = StorySageRetriever(chroma_path, chroma_collection_name, entities, n_chunks)
        self.chain = StorySageChain(api_key, entities, self.retriever, self.logger)
        
        # Load series configuration
        try:
            with open(series_yml_path, 'r') as file:
                self.series_dict = yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Failed to load series configuration: {e}")
            raise

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
            Tuple containing (answer, context_list)
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
            raise