import json
from typing import List, Tuple
import uuid
import redis
import logging

logger = logging.getLogger(__name__)

class TurnType():
    """Represents a turn in a conversation.

    Attributes:
        question (str): The user's question.
        detected_entities (List[str]): List of detected entities in the question.
        context (List[str]): The context returned by the system.
        response (str): The system's response.
        request_id (str): The request identifier.
    """
    
    def __init__(self, question: str, detected_entities: List[str], context: List[str], response: str, request_id: str):
        """
        Args:
            question (str): The user's question.
            detected_entities (List[str]): List of detected entities in the question.
            context (List[str]): The context returned by the system.
            response (str): The system's response.
            request_id (str): The request identifier.
        """
        self.question = question
        self.detected_entities = detected_entities
        self.context = context
        self.response = response
        self.request_id = request_id

    def to_json(self) -> dict:
        """Converts the turn to a JSON dictionary.

        Returns:
            dict: A dictionary representation of the turn.
        """
        try:
            return {
                'question': self.question,
                'detected_entities': self.detected_entities,
                'context': self.context,
                'response': self.response,
                'request_id': self.request_id
            }
        except Exception as e:
            logger.error(f"Error converting turn to JSON: {e}")
            logger.error(self.detected_entities)
            raise e

class StorySageConversation():
    """Represents a conversation in the StorySage system.

    This class contains the conversation history for a user's session. It tracks
    the questions and responses along with the retrieved context. Each interaction
    is represented by a "turn". The turn includes the question, the detected entities,
    the returned context, and the response.

    Attributes:
        conversation_id (str): Unique identifier for the conversation.
        turns (List[TurnType]): List of turns in the conversation.
        redis (redis.Redis): Redis client for caching conversation data.
        redis_ex (int): Expiration time for Redis cache in seconds.
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
                 context: List[str], response: str, request_id: str) -> None:
        """Adds a turn to the conversation history.

        Args:
            question (str): The user's question.
            detected_entities (List[str]): List of detected entities in the question.
            context (List[str]): The context returned by the system.
            response (str): The system's response.
            request_id (str): The request identifier.
        """
        turn = TurnType(question=question, detected_entities=detected_entities, 
                        context=context, response=response, request_id=request_id)
        self.turns.append(turn)
        self._save_to_cache()

    def get_history(self) -> List[Tuple[str, List[str], List[str], str, str]]:
        """Returns the conversation history.

        Returns:
            List[Tuple[str, List[str], List[str], str, str]]: A list of tuples representing each turn in the conversation.
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
            return [TurnType(turn['question'], turn['detected_entities'], turn['context'], turn['response'], turn['request_id']) for turn in turns]
        else:
            return []
        
    def _save_to_cache(self) -> None:
        """Saves conversation history to cache."""
        if not self.redis:
            return
        
        key = f"conversation:{self.conversation_id}"
        data = json.dumps([turn.to_json() for turn in self.turns])
        self.redis.set(key, data, ex=self.redis_ex)

# Example usage:
# redis_client = redis.Redis(host='localhost', port=6379, db=0)
# conversation = StorySageConversation(redis=redis_client)
# conversation.add_turn("What is the weather today?", [], ["Sunny"], "It's sunny today.", "req123")
# history = conversation.get_history()
# print(history)