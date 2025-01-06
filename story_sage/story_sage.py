import logging
import uuid
from typing import Tuple, Optional, Dict, List
from .data_classes.story_sage_state import StorySageState
from .vector_store import StorySageRetriever
from .story_sage_chain import StorySageChain
from .data_classes.story_sage_config import StorySageConfig
from .story_sage_conversation import StorySageConversation

class ConditionalRequestIDFormatter(logging.Formatter):
    """Custom formatter to include request_id only if it's not None.
    
    Extends the built-in logging.Formatter to conditionally append a
    request_id to the log message if present.
    """
    
    def format(self, record):
        if hasattr(record, 'request_id') and record.request_id:
            original_msg = super().format(record)
            return f"{original_msg} [Request ID: {record.request_id}]"
        return super().format(record)

class StorySage:
    """Main class for the Story Sage system.

    Coordinates between the retriever, chain, and state management components
    to help readers track story elements.

    Example usage:
        story_sage = StorySage(
            api_key="YOUR_API_KEY",
            chroma_path="path/to/chroma",
            chroma_collection_name="collection_name",
            entities_dict={"meta": StorySageEntityCollection(...)},
            series_list=[{"title": "Series Title"}],
            n_chunks=5
        )
        answer, context, request_id = story_sage.invoke(
            question="Who is the main character?",
            book_number=1,
            chapter_number=1,
            series_id=1
        )

        Example result:
            answer: "The main character is Rand al'Thor."
            context: ["Rand al'Thor is introduced in the first chapter..."]
            request_id: "123e4567-e89b-12d3-a456-426614174000"
    """

    def __init__(self, config: StorySageConfig, log_level: int = logging.INFO):
        """Initializes the StorySage instance.

        Args:
            api_key (str): API key for accessing external services.
            chroma_path (str): Path to the Chroma database.
            chroma_collection_name (str): Name of the Chroma collection.
            entities_dict (dict[str, StorySageEntityCollection]): Mapping of series metadata to entity collections.
            series_list (List[dict], optional): A list of dictionaries representing series info. Defaults to [].
            n_chunks (int, optional): Number of chunks for the retriever to process. Defaults to 5.
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Initialize request_id
        self.request_id = None

        self.entities = config.entities

        # Initialize series info
        self.series_list = config.series

        # Initialize retriever and chain components
        self.retriever = StorySageRetriever(config.chroma_path, config.chroma_collection, config.n_chunks)
        self.chain = StorySageChain(config.openai_api_key, self.entities, self.series_list, self.retriever, log_level)
        

    def invoke(self, question: str, book_number: int = None, 
               chapter_number: int = None, series_id: int = None,
               conversation: StorySageConversation = None) -> Tuple[str, List[str], str, List[str]]:
        """Invokes the question-processing logic through the chain.

        Args:
            question (str): The user's question about the story.
            book_number (int, optional): Optional book number for context filtering. Defaults to None.
            chapter_number (int, optional): Optional chapter number for context filtering. Defaults to None.
            series_id (int, optional): Optional series ID for context filtering. Defaults to None.

        Returns:
            Tuple[str, List[str], str, List[str]]: A tuple containing the answer, context list, generated request ID, and list of entity IDs.
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
            entities=[],
            order_by='most_recent',
            conversation=conversation,
            node_history=['START'],
            tokens_used=0
        )

        try:
            # Process the question through the chain
            result = self.chain.graph.invoke(state)
            
            # Log the results
            self.logger.debug(f"Generated answer: {result['answer']}")
            self.logger.debug(f"Retrieved context: {result['context']}")
            
            return result['answer'], result['context'], self.request_id, result['entities']
            
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