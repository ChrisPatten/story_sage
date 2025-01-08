import logging
import uuid
from typing import Tuple, Optional, Dict, List
from .types import StorySageConfig, StorySageContext, StorySageState, StorySageConversation, StorySageChain, StorySageRetriever

class ConditionalRequestIDFormatter(logging.Formatter):
    """Custom formatter that conditionally includes a request ID in log messages.
    
    This formatter extends the standard logging.Formatter to add request ID tracking
    to log messages. If a request_id is present in the log record, it will be appended
    to the end of the message in brackets.
    
    Example:
        formatter = ConditionalRequestIDFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # With request_id: "2023-01-01 10:00:00 - StorySage - INFO - Processing query [Request ID: abc-123]"
        # Without request_id: "2023-01-01 10:00:00 - StorySage - INFO - Processing query"
    """
    
    def format(self, record):
        if hasattr(record, 'request_id') and record.request_id:
            original_msg = super().format(record)
            return f"{original_msg} [Request ID: {record.request_id}]"
        return super().format(record)

class StorySage:
    """A sophisticated story comprehension and tracking system.

    StorySage helps readers understand and track elements of complex narratives by providing
    context-aware answers to questions about the story. It combines vector storage, LLM chains,
    and state management to maintain coherent conversations about story elements.

    Attributes:
        logger: A configured logging instance for the class.
        request_id: A unique identifier for tracking individual requests.
        entities: A dictionary of story entities and their metadata.
        series_list: Information about the book series being processed.
        retriever: Component for retrieving relevant text chunks.
        chain: Component for processing questions and generating responses.

    Example:
        config = StorySageConfig(
            openai_api_key="sk-...",
            chroma_path="/path/to/db",
            chroma_collection="my_books",
            entities={"characters": CharacterCollection(...)},
            series=[{"id": 1, "title": "The Example Series"}],
            n_chunks=5,
            prompts={"main": "...", "followup": "..."}
        )
        
        sage = StorySage(config)
        
        # Simple question about the story
        answer, context, request_id, entities = sage.invoke(
            question="What happens in chapter 1?",
            book_number=1,
            chapter_number=1,
            series_id=1
        )
        
        # Using conversation history
        conversation = StorySageConversation(...)
        answer, context, request_id, entities = sage.invoke(
            question="What happened next?",
            conversation=conversation
        )

    Returns:
        answer (str): The generated response to the question
        context (List[str]): Relevant text chunks used to generate the answer
        request_id (str): Unique identifier for the request
        entities (List[str]): IDs of story entities referenced in the response
    """

    def __init__(self, config: StorySageConfig, log_level: int = logging.INFO):
        """Initializes a new StorySage instance with the provided configuration.

        Args:
            config (StorySageConfig): Configuration object containing all necessary parameters
                including API keys, database paths, entity collections, and prompt templates.
            log_level (int, optional): The logging level to use. Defaults to logging.INFO.
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Initialize request_id
        self.request_id = None
        
        self.config = config

        self.entities = config.entities

        # Initialize series info
        self.series_list = config.series

        # Initialize retriever and chain components
        self.summary_retriever = StorySageRetriever(config.chroma_path, config.chroma_collection, config.n_chunks)
        self.full_retriever = StorySageRetriever(config.chroma_path, config.chroma_full_text_collection, round(config.n_chunks / 3))
        

    def invoke(self, question: str, book_number: int = None, 
               chapter_number: int = None, series_id: int = None,
               conversation: StorySageConversation = None) -> Tuple[str, List[StorySageContext], str, List[str]]:
        """Processes a question about the story and generates a contextual response.

        This method coordinates the retrieval of relevant context and the generation
        of appropriate responses, taking into account the current position in the story
        and any ongoing conversation history.

        Args:
            question (str): The user's question about the story.
            book_number (int, optional): The book number to use for context. 
                Defaults to None (will use series end).
            chapter_number (int, optional): The chapter number to use for context. 
                Defaults to None (will use 0).
            series_id (int, optional): The ID of the book series. 
                Defaults to None (will use 0).
            conversation (StorySageConversation, optional): Ongoing conversation context.
                Defaults to None.

        Returns:
            Tuple[str, List[str], str, List[str]]: A tuple containing:
                - answer: The generated response to the question
                - context: List of text chunks used to generate the answer
                - request_id: Unique identifier for this request
                - entities: List of entity IDs referenced in the response

        Raises:
            Exception: If there's an error during question processing
        
        Example:
            >>> sage = StorySage(config)
            >>> answer, context, req_id, entities = sage.invoke(
            ...     question="Who is John meeting in chapter 3?",
            ...     book_number=1,
            ...     chapter_number=3
            ... )
            >>> print(answer)
            "John meets Sarah at the coffee shop."
            >>> print(entities)
            ["character_john", "character_sarah", "location_coffee_shop"]
        """
        self.logger.info(f"Processing question: {question}")

        # Initialize state with default values
        state = StorySageState(
            question=question,
            book_number=book_number,  
            chapter_number=chapter_number,
            series_id=series_id,
            conversation=conversation
        )

        self.chain = StorySageChain(config=self.config, state=state, log_level=self.logger.level)

        try:
            # Process the question through the chain
            result = self.chain.invoke()
            
            # Log the results
            self.logger.debug(f"Generated answer: {result.answer}")
            self.logger.debug(f"Retrieved context: {result.context}")
            
            return result.answer, result.context, None, result.entities
            
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