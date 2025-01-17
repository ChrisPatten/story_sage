import json
from typing import List, Tuple
import uuid
import redis
import logging
import time
from .story_sage_context import StorySageContext

logger = logging.getLogger(__name__)

class TurnType():
    """Represents a single interaction (turn) in a conversation between user and system.

    A turn consists of a user's question, any entities detected in that question,
    the context used to generate the response, and the system's response.

    Attributes:
        question (str): The user's input question or statement.
        detected_entities (List[str]): Entities (e.g., names, places) detected in the question.
        context (List[str]): Relevant context snippets used to generate the response.
        response (str): The system's generated response to the question.
        sequence (int): The position of this turn in the conversation (0-based).

    Example:
        turn = TurnType(
            question="Who is Harry Potter?",
            detected_entities=["Harry Potter"],
            context=["Harry Potter is a fictional wizard created by J.K. Rowling"],
            response="Harry Potter is a famous fictional wizard and the main protagonist...",
            sequence=0
        )
    """
    
    def __init__(self, question: str, detected_entities: List[str], context: List[StorySageContext], 
                 response: str, sequence: int):
        """Initialize a new conversation turn.

        Args:
            question (str): The user's question or input.
            detected_entities (List[str]): List of entities detected in the question.
            context (List[StorySageContext]): Relevant context used for generating the response.
            response (str): The system's response to the question.
            sequence (int): Position of this turn in the conversation (0-based).
        """
        self.question = question
        self.detected_entities = detected_entities
        self.context = context
        self.response = response
        self.sequence = sequence

    def to_json(self) -> dict:
        """Converts the turn to a JSON dictionary.

        Returns:
            dict: A dictionary representation of the turn.
        """
        try:
            return {
                'question': self.question,
                'detected_entities': self.detected_entities,
                'context': [c.to_dict() for c in self.context],
                'response': self.response,
                'sequence': self.sequence
            }
        except Exception as e:
            logger.error("Error converting turn to JSON", exc_info=True, stack_info=True, stacklevel=50)
            raise e
        
    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, value: List[StorySageContext]):
        # Intercept the setting of context here
        if not type(value) == list and type(value[0]) == StorySageContext:
            raise ValueError("Context must be a list of StorySageContext objects")
        self._context = value

class StorySageConversation():
    """Manages a conversation session in the StorySage system with persistence support.

    This class handles the complete conversation lifecycle, including:
    - Creating and managing conversation turns
    - Persisting conversation history to Redis
    - Retrieving conversation history
    - Formatting conversation logs

    Attributes:
        conversation_id (uuid.UUID): Unique identifier for the conversation session.
        turns (List[TurnType]): Ordered list of conversation turns.
        redis (redis.Redis): Redis client for conversation persistence.
        redis_ex (int): Redis cache expiration time in seconds.

    Example:
        # Initialize with Redis for persistence
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        conversation = StorySageConversation(redis=redis_client)

        # Add a new turn to the conversation
        conversation.add_turn(
            question="What happens in Chapter 1?",
            detected_entities=["Chapter 1"],
            context=["Chapter 1 introduces the main character..."],
            response="In Chapter 1, we meet the protagonist..."
        )

        # Get conversation history
        history = conversation.get_history()
        
        # Get formatted conversation log
        log = conversation.get_log()
        # Returns:
        # [
        #     {
        #         "question": "What happens in Chapter 1?",
        #         "detected_entities": ["Chapter 1"],
        #         "context": ["Chapter 1 introduces the main character..."],
        #         "response": "In Chapter 1, we meet the protagonist...",
        #         "sequence": 0
        #     }
        # ]
    """

    def __init__(self, conversation_id: str = None, redis: redis.Redis = None, redis_ex: int = 3600):
        """Initialize a new conversation session.

        Args:
            conversation_id (str, optional): Unique identifier for the conversation. Defaults to None.
            redis (redis.Redis, optional): Redis client for caching conversation data. Defaults to None.
            redis_ex (int, optional): Expiration time for Redis cache in seconds. Defaults to 3600.
        """
        self.redis = redis
        self.redis_ex = redis_ex
        
        if conversation_id is None:
            self.conversation_id = uuid.uuid4()
            self.turns: List[TurnType] = []
            logger.info("Created new conversation with ID: %s", self.conversation_id)
        else:
            try:
                self.conversation_id = uuid.UUID(conversation_id)
                logger.info("Initializing existing conversation: %s", self.conversation_id)
            except ValueError:
                logger.error("Invalid conversation ID provided: %s", conversation_id)
                raise ValueError("conversation_id must be a valid UUID")
            
            if redis is not None:
                self.turns = self._load_from_cache()
                logger.debug("Loaded %d turns from cache for conversation %s", 
                           len(self.turns), self.conversation_id)
            else:
                self.turns = []
                logger.debug("No Redis cache provided, starting with empty conversation")

    def add_turn(self, question: str, context: List[StorySageContext],
                 response: str, detected_entities: List[str] = None) -> None:
        """Add a new turn to the conversation.

        Args:
            question (str): The user's question
            context (List[StorySageContext]): Context used for response
            response (str): System's response
            detected_entities (List[str], optional): Detected entities. Defaults to empty list.
        """
        detected_entities = detected_entities or []
        next_sequence = len(self.turns)
        
        logger.info("Adding turn %d to conversation %s", next_sequence, self.conversation_id)
        logger.debug("Question: '%s', Entities: %s", question[:100], detected_entities)
        
        turn = TurnType(
            question=question, 
            detected_entities=detected_entities,
            context=context, 
            response=response,
            sequence=next_sequence
        )
        self.turns.append(turn)
        
        # Persist to cache if Redis is configured
        if self.redis:
            start_time = time.time()
            self._save_to_cache()
            duration = time.time() - start_time
            logger.debug("Saved turn to cache in %.2fs", duration)

    def get_history(self) -> List[Tuple[str, List[str], List[StorySageContext], str]]:
        """Retrieve complete conversation history.

        Returns:
            List[Tuple[str, List[str], List[StorySageContext], str]]: List of tuples containing
                (question, detected_entities, context, response) for each turn.
        """
        logger.debug("Retrieving history for conversation %s (%d turns)", 
                    self.conversation_id, len(self.turns))
        return [(turn.question, turn.detected_entities, turn.context, turn.response) 
                for turn in self.turns]
    
    def _load_from_cache(self) -> List[TurnType]:
        """Load conversation history from Redis cache.

        Returns:
            List[TurnType]: List of conversation turns.
            
        Raises:
            redis.RedisError: If there's an error accessing Redis
            json.JSONDecodeError: If cached data is invalid
        """
        if not self.redis:
            return []
        
        key = f"conversation:{self.conversation_id}"
        start_time = time.time()
        
        try:
            data = self.redis.get(key)
            if not data:
                logger.debug("No cached data found for conversation %s", self.conversation_id)
                return []
            
            turns = json.loads(data)
            # Convert context dictionaries back to StorySageContext objects
            for turn in turns:
                turn['context'] = [StorySageContext.from_dict(c) for c in turn['context']]
            
            duration = time.time() - start_time
            logger.info("Loaded %d turns from cache in %.2fs", len(turns), duration)
            return [TurnType(**turn) for turn in turns]
            
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error("Failed to load conversation from cache: %s", str(e), 
                        exc_info=True)
            raise
        
    def _save_to_cache(self) -> None:
        """Save conversation history to Redis cache.
        
        Raises:
            redis.RedisError: If there's an error accessing Redis
            json.JSONEncodeError: If turns cannot be serialized
        """
        if not self.redis:
            return
        
        key = f"conversation:{self.conversation_id}"
        start_time = time.time()
        
        try:
            data = json.dumps([turn.to_json() for turn in self.turns])
            self.redis.set(key, data, ex=self.redis_ex)
            
            duration = time.time() - start_time
            logger.debug("Saved conversation to cache in %.2fs", duration)
            
        except (redis.RedisError, TypeError) as e:
            logger.error("Failed to save conversation to cache: %s", str(e), 
                        exc_info=True)
            raise

    def get_log(self) -> str:
        """Get a formatted JSON log of the conversation.

        Returns:
            str: JSON string containing the complete conversation history.
        """
        try:
            conversation = [turn.to_json() for turn in self.turns]
            return json.dumps(conversation, indent=4)
        except Exception as e:
            logger.error("Failed to generate conversation log: %s", str(e), 
                        exc_info=True)
            raise

# Example usage:
# redis_client = redis.Redis(host='localhost', port=6379, db=0)
# conversation = StorySageConversation(redis=redis_client)
# conversation.add_turn("What is the weather today?", [], ["Sunny"], "It's sunny today.", "req123")
# history = conversation.get_history()
# print(history)