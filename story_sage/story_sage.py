import logging
import uuid
import time
from typing import Tuple, Optional, Dict, List
from .models import StorySageConfig, StorySageContext, StorySageState, StorySageConversation
from .services import StorySageChain, StorySageRetriever

class StorySage:
    """A sophisticated story comprehension and tracking system.

    StorySage helps readers understand and track elements of complex narratives by providing
    context-aware answers to questions about the story. It combines vector storage, LLM chains,
    and state management to maintain coherent conversations about story elements.

    The system maintains conversation history and context awareness to provide more accurate
    and contextually relevant answers to follow-up questions.

    Attributes:
        logger: A configured logging instance for tracking system events and debugging.
        request_id: A unique identifier for tracking individual question/answer sessions.
        entities: A dictionary mapping entity types to their collections (e.g., characters, locations).
        series_list: List of dictionaries containing metadata about book series in the system.
        summary_retriever: Component for retrieving relevant summary text chunks.
        full_retriever: Component for retrieving relevant full text chunks.
        chain: Processing chain for generating answers from context and questions.
        config: Configuration object containing system parameters and settings.

    Example:
        >>> # Initialize with configuration from dictionary
        >>> config_dict = {
        ...     'OPENAI_API_KEY': 'sk-...',
        ...     'CHROMA_PATH': './chromadb',
        ...     'CHROMA_COLLECTION': 'book_embeddings',
        ...     'CHROMA_FULL_TEXT_COLLECTION': 'book_texts',
        ...     'N_CHUNKS': 5,
        ...     'COMPLETION_MODEL': 'gpt-3.5-turbo',
        ...     'COMPLETION_TEMPERATURE': 0.7,
        ...     'COMPLETION_MAX_TOKENS': 2000,
        ...     'SERIES_PATH': 'configs/series.yaml',
        ...     'ENTITIES_PATH': 'configs/entities.json',
        ...     'REDIS_URL': 'redis://localhost:6379/0',
        ...     'REDIS_EXPIRE': 3600,
        ...     'PROMPTS_PATH': 'configs/prompts.yaml'
        ... }
        >>> config = StorySageConfig.from_config(config_dict)
        >>> sage = StorySage(config)
        >>> 
        >>> # Ask a question about specific book/chapter
        >>> answer, context, req_id, entities = sage.invoke(
        ...     question="Who appears in the first scene?",
        ...     book_number=1,
        ...     chapter_number=1,
        ...     series_id=1
        ... )
            
        >>> # Follow-up question using conversation history
        >>> redis_client = redis.Redis.from_url('redis://localhost:6379/0')
        >>> conv = StorySageConversation(redis=redis_client)
        >>> conv.add_turn(
        ...     question="What happens when Alice enters the garden?",
        ...     detected_entities=["character_alice", "location_garden"],
        ...     context=["Alice discovers a magical fountain..."],
        ...     response="Alice discovers a magical fountain in the garden...",
        ...     request_id="turn_001"
        ... )
        >>> answer, context, req_id, entities = sage.invoke(
        ...     question="What does she do next?",
        ...     conversation=conv
        ... )
    """

    def __init__(self, config: StorySageConfig, log_level: int = logging.INFO):
        """Initializes a new StorySage instance with the provided configuration.

        Args:
            config (StorySageConfig): Configuration object containing all necessary parameters
                including API keys, database paths, entity collections, and prompt templates.
                See class docstring for detailed config example.
            log_level (int, optional): The logging level to use. Defaults to logging.INFO.
                Use logging.DEBUG for more detailed output during development.
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.logger.info("Initializing StorySage instance")

        # Initialize request_id
        self.request_id = None
        
        self.config = config

        self.entities = config.entities

        # Initialize series info
        self.series_list = config.series
        
        self.logger.debug("Loaded %d series and %d entity collections", 
                         len(self.series_list), len(self.entities))

    def invoke(self, question: str, book_number: Optional[int] = None, 
               chapter_number: Optional[int] = None, series_id: Optional[int] = None,
               conversation: Optional[StorySageConversation] = None
               ) -> Tuple[str, List[StorySageContext], Optional[str], List[str]]:
        """Processes a question about the story and generates a contextual response.

        This method coordinates the retrieval of relevant context and the generation
        of appropriate responses, taking into account the current position in the story
        and any ongoing conversation history.

        Args:
            question (str): The user's question about the story content.
            book_number (int, optional): The specific book number to reference for context.
                When None, uses the latest book in the series.
            chapter_number (int, optional): The specific chapter number to reference.
                When None, considers the entire book context.
            series_id (int, optional): The ID of the book series being discussed.
                When None, defaults to the first available series.
            conversation (StorySageConversation, optional): Previous conversation context
                for maintaining coherent multi-turn discussions. Contains prior
                questions, answers, and referenced entities.

        Returns:
            tuple: A tuple containing:
                - answer (str): The generated response to the question
                - context (List[StorySageContext]): Relevant text chunks used for the answer
                - request_id (str): Unique identifier for tracking this Q&A turn
                - entities (List[str]): IDs of story entities (characters, locations, etc.)
                    referenced in the response

        Raises:
            Exception: If there's an error during question processing, context retrieval,
                or response generation.

        Example:
            >>> # Simple question about a specific chapter
            >>> answer, context, req_id, entities = sage.invoke(
            ...     question="What happens when Alice enters the garden?",
            ...     book_number=1,
            ...     chapter_number=3
            ... )
            >>> print(f"Answer: {answer}")
            "Answer: Alice discovers a magical fountain in the garden..."
            >>> print(f"Referenced entities: {entities}")
            "Referenced entities: ['character_alice', 'location_garden', 'item_fountain']"
            
            >>> # Follow-up question using conversation history
            >>> redis_client = redis.Redis.from_url('redis://localhost:6379/0')
            >>> conv = StorySageConversation(redis=redis_client)
            >>> conv.add_turn(
            ...     question="What happens when Alice enters the garden?",
            ...     detected_entities=["character_alice", "location_garden"],
            ...     context=["Alice discovers a magical fountain..."],
            ...     response="Alice discovers a magical fountain in the garden...",
            ...     request_id="turn_001"
            ... )
            >>> answer, context, req_id, entities = sage.invoke(
            ...     question="What does she do next?",
            ...     conversation=conv
            ... )
        """
        start_time = time.time()
        self.logger.info("Processing question: '%s'", question[:100])
        self.logger.debug("Context - Book: %s, Chapter: %s, Series: %s", 
                         book_number, chapter_number, series_id)
        
        if conversation:
            self.logger.debug("Using existing conversation with %d turns", 
                            len(conversation.turns))

        # Initialize state with default values
        state = StorySageState(
            question=question,
            book_number=book_number,  
            chapter_number=chapter_number,
            series_id=series_id,
            conversation=conversation
        )

        try:
            # Initialize processing chain
            chain_start = time.time()
            self.chain = StorySageChain(
                config=self.config, 
                state=state, 
                log_level=self.logger.level
            )
            self.logger.debug("Chain initialization took %.2fs", 
                            time.time() - chain_start)

            # Process the question through the chain
            self.logger.info("Starting chain processing")
            result = self.chain.invoke()
            
            # Log completion and timing
            duration = time.time() - start_time
            self.logger.info("Question processed in %.2fs", duration)
            self.logger.debug("Answer length: %d chars", len(result.answer))
            self.logger.debug("Retrieved %d context chunks", len(result.context))
            self.logger.debug("Referenced entities: %s", result.entities)
            
            # Log cost metrics
            cost = result.get_cost()
            self.logger.info(f"Processing cost: {cost}")
            
            return result.answer, result.context, None, result.entities
            
        except Exception as e:
            self.logger.error("Failed to process question: %s", str(e), 
                            exc_info=True)
            self.logger.error("Failed state: %s", state)
            duration = time.time() - start_time
            self.logger.warning("Failed request duration: %.2fs", duration)
            raise