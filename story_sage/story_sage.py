import chromadb
import logging
import yaml
from typing import Tuple, Optional, Dict, List
from .story_sage_state import StorySageState
from .story_sage_retriever import StorySageRetriever
from .story_sage_chain import StorySageChain

class StorySage():
    """
    Main class for the Story Sage system that helps readers track story elements.
    Coordinates between the retriever, chain, and state management components.
    """

    def __init__(self, api_key: str, chroma_path: str, chroma_collection_name: str, 
                 entities: dict, series_yml_path: str, n_chunks: int = 5):
        """Initialize the StorySage instance with necessary components and configuration."""
        # Initialize components
        self.logger = logging.getLogger(__name__)
        self.retriever = StorySageRetriever(chroma_path, chroma_collection_name, entities, n_chunks)
        self.chain = StorySageChain(api_key, entities, self.retriever)
        
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
        
        # Initialize state with default values
        state = StorySageState(
            question=question,
            book_number=book_number or 100,  # Default to end of series if not specified
            chapter_number=chapter_number or 0,
            series_id=series_id or 0,
            context=None,
            answer=None,
            people=[],
            places=[],
            groups=[],
            animals=[]
        )

        try:
            # Process the question through the chain
            result = self.chain.graph.invoke(state)
            
            # Log the results
            self.logger.debug(f"Generated answer: {result['answer']}")
            self.logger.debug(f"Retrieved context: {result['context']}")
            
            return result['answer'], result['context']
            
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return "I apologize, but I encountered an error processing your question.", []
