import tempfile
import json
from typing import List, Tuple
import uuid
from .story_sage_entity import StorySageEntityCollection, StorySageEntityGroup, StorySageEntity
import glob

class TurnType():
    """Represents a turn in a conversation.

    Attributes:
        question (str): The user's question.
        detected_entities (List[str]): List of detected entities in the question.
        context (str): The context returned by the system.
        response (str): The system's response.
    """
    
    def __init__(self, question: str, detected_entities: List[StorySageEntityGroup], context: List[str], response: str, request_id: str):
        self.question = question
        self.detected_entities = detected_entities
        self.context = context
        self.response = response
        self.request_id = request_id

class StorySageConversation():
    """Represents a conversation in the StorySage system.

    This class contains the conversation history for a user's session. It tracks
    the questions and responses along with the retrieved context. Each interaction
    is represented by a "turn". The turn includes the question, the detected entities,
    the returned context, and the response.

    Attributes:
        conversation_id (str): Unique identifier for the conversation.
        turns (List[TurnType]): List of turns in the conversation.
    """

    def __init__(self, conversation_id: str = None):
        if conversation_id is None:
            self.conversation_id = uuid.uuid4()
            self.turns: List[TurnType]= []
        else:
            try:
                self.conversation_id = uuid.UUID(conversation_id)
            except ValueError:
                raise ValueError("conversation_id must be a valid UUID")
            
            # Read from cache
            self.turns: List[TurnType] = []


    def add_turn(self, question: str, detected_entities: List[StorySageEntityGroup], 
                 context: List[str], response: str, request_id: str) -> None:
        """Adds a turn to the conversation history.

        Args:
            question (str): The user's question.
            detected_entities (List[StorySageEntityGroup]): List of detected entities in the question.
            context (List[str]): The context returned by the system.
            response (str): The system's response.
            request_id (str): The request identifier.
        """
        turn = TurnType(question, detected_entities, context, response, request_id)
        self.turns.append(turn)
        # Add to cache

    def get_history(self) -> List[Tuple[str, List[StorySageEntityGroup], List[str], str, str]]:
        """Returns the conversation history.

        Returns:
            List[Tuple[str, List[StorySageEntityGroup], List[str], str, str]]: A list of tuples representing each turn in the conversation.
        """
        return [(turn.question, turn.detected_entities, turn.context, turn.response, turn.request_id) for turn in self.turns]
        
