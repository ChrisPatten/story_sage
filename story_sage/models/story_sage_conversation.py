import json
from typing import List, Tuple
import uuid
import redis
import logging
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
        request_id (str): Unique identifier for this specific turn.
        sequence (int): The position of this turn in the conversation (0-based).

    Example:
        turn = TurnType(
            question="Who is Harry Potter?",
            detected_entities=["Harry Potter"],
            context=["Harry Potter is a fictional wizard created by J.K. Rowling"],
            response="Harry Potter is a famous fictional wizard and the main protagonist...",
            request_id="123xyz",
            sequence=0
        )
    """
    
    def __init__(self, question: str, detected_entities: List[str], context: List[StorySageContext], 
                 response: str, request_id: str, sequence: int):
        """Initialize a new conversation turn.

        Args:
            question (str): The user's question or input.
            detected_entities (List[str]): List of entities detected in the question.
            context (List[str]): Relevant context used for generating the response.
            response (str): The system's response to the question.
            request_id (str): Unique identifier for this turn.
            sequence (int): Position of this turn in the conversation (0-based).
        """
        self.question = question
        self.detected_entities = detected_entities
        self.context = context
        self.response = response
        self.request_id = request_id
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
                'context': [c.format_for_llm() for c in self.context],
                'response': self.response,
                'request_id': self.request_id,
                'sequence': self.sequence
            }
        except Exception as e:
            logger.error(f"Error converting turn to JSON: {e}")
            logger.error(self.detected_entities)
            raise e

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
            response="In Chapter 1, we meet the protagonist...",
            request_id="req_123"
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
        #         "request_id": "req_123",
        #         "sequence": 0
        #     }
        # ]
    """

    def __init__(self, conversation_id: str = None, redis: redis.Redis = None, redis_ex: int = 3600):
        """
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
        else:
            try:
                self.conversation_id = uuid.UUID(conversation_id)
            except ValueError:
                raise ValueError("conversation_id must be a valid UUID")
            
            if redis is not None:
                self.turns = self._load_from_cache()
            else:
                self.turns = []

    def add_turn(self, question: str, detected_entities: List[str], 
                 context: List[StorySageContext], response: str, request_id: str) -> None:
        """Adds a new turn to the conversation and persists it to cache if Redis is configured.

        Args:
            question (str): The user's question or input.
            detected_entities (List[str]): Entities detected in the question.
            context (List[str]): Context used for response generation.
            response (str): The system's response.
            request_id (str): Unique identifier for this turn.

        Example:
            conversation.add_turn(
                question="Who is the antagonist?",
                detected_entities=["antagonist"],
                context=["The main antagonist is..."],
                response="The story's main antagonist is...",
                request_id="req_124"
            )
        """
        next_sequence = len(self.turns)
        turn = TurnType(question=question, detected_entities=detected_entities, 
                        context=context, response=response, request_id=request_id,
                        sequence=next_sequence)
        self.turns.append(turn)
        self._save_to_cache()

    def get_history(self) -> List[Tuple[str, List[str], List[str], str, str, int]]:
        """Retrieves the complete conversation history as a list of tuples.

        Returns:
            List[Tuple[str, List[str], List[str], str, str, int]]: List of tuples containing
                (question, detected_entities, context, response, request_id) for each turn.

        Example:
            history = conversation.get_history()
            # Returns: [
            #     ("Who is the antagonist?", ["antagonist"], ["The main antagonist is..."],
            #      "The story's main antagonist is...", "req_124")
            # ]
        """
        return [(turn.question, turn.detected_entities, turn.context, turn.response, turn.request_id) for turn in self.turns]
    
    def _load_from_cache(self) -> List[TurnType]:
        """Loads conversation history from cache.

        Returns:
            List[TurnType]: List of turns in the conversation.
        """
        if not self.redis:
            return []
        
        key = f"conversation:{self.conversation_id}"
        data = self.redis.get(key)
        if data:
            turns = json.loads(data)
            return [TurnType(turn['question'], turn['detected_entities'], turn['context'], turn['response'], turn['request_id'], turn['sequence']) for turn in turns]
        else:
            return []
        
    def _save_to_cache(self) -> None:
        """Saves conversation history to cache."""
        if not self.redis:
            return
        
        key = f"conversation:{self.conversation_id}"
        data = json.dumps([turn.to_json() for turn in self.turns])
        self.redis.set(key, data, ex=self.redis_ex)

    def __str__(self) -> str:
        conversation = []
        for turn in self.turns:
            conversation.append(f"HUMAN: {turn.question}")
            conversation.append(f"COMPUTER: {turn.response}")
        return '\n'.join(conversation)
    
    def get_log(self) -> str:
        """Returns the conversation log as a string.

        Returns:
            str: A string representation of the conversation log.
        """
        conversation = []
        for turn in self.turns:
            conversation.append(turn.to_json())

        return json.dumps(conversation, indent=4)

# Example usage:
# redis_client = redis.Redis(host='localhost', port=6379, db=0)
# conversation = StorySageConversation(redis=redis_client)
# conversation.add_turn("What is the weather today?", [], ["Sunny"], "It's sunny today.", "req123")
# history = conversation.get_history()
# print(history)