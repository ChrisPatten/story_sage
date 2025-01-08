import pytest
from unittest.mock import MagicMock, patch
from story_sage.models.story_sage_conversation import (
    TurnType,
    StorySageConversation
)
import json
import uuid

@pytest.fixture
def turn():
    return TurnType(
        question="What is your name?",
        detected_entities=["name"],
        context=["Standard prompt"],
        response="I am Story Sage.",
        request_id="req_001",
        sequence=0
    )

@pytest.fixture
def mock_redis_client():
    with patch('story_sage.models.story_sage_conversation.redis.Redis') as MockRedis:
        mock = MockRedis.return_value
        mock.get.return_value = None  # Default to None, can be overridden in tests
        mock.set.return_value = True  # Mock set method as successful
        yield mock

@pytest.fixture
def conversation_without_redis():
    return StorySageConversation()

@pytest.fixture
def conversation_with_redis(mock_redis_client):
    fixed_conversation_id = str(uuid.uuid4())
    return StorySageConversation(conversation_id=fixed_conversation_id, redis=mock_redis_client)

def test_turn_initialization(turn):
    assert turn.question == "What is your name?"
    assert turn.detected_entities == ["name"]
    assert turn.context == ["Standard prompt"]
    assert turn.response == "I am Story Sage."
    assert turn.request_id == "req_001"
    assert turn.sequence == 0

def test_turn_to_json(turn):
    expected_json = {
        'question': "What is your name?",
        'detected_entities': ["name"],
        'context': ["Standard prompt"],
        'response': "I am Story Sage.",
        'request_id': "req_001",
        'sequence': 0
    }
    assert turn.to_json() == expected_json

def test_conversation_initialization_without_redis(conversation_without_redis):
    assert conversation_without_redis.conversation_id is not None
    assert conversation_without_redis.redis is None
    assert conversation_without_redis.redis_ex == 3600
    assert len(conversation_without_redis.turns) == 0

def test_conversation_initialization_with_redis(conversation_with_redis, mock_redis_client):
    # Ensure get returns None when initializing
    mock_redis_client.get.return_value = None

    assert conversation_with_redis.redis == mock_redis_client
    assert conversation_with_redis.redis_ex == 3600
    
    # Ensure that the 'get' method was called with the correct key
    mock_redis_client.get.assert_called_once_with(f"conversation:{conversation_with_redis.conversation_id}")
    
    # Since the mock returns None by default, turns should be empty
    assert len(conversation_with_redis.turns) == 0

def test_conversation_add_turn_without_redis(conversation_without_redis):
    conversation_without_redis.add_turn(
        question="What is your purpose?",
        detected_entities=["purpose"],
        context=["Purpose context"],
        response="To assist you.",
        request_id="req_002"
    )
    assert len(conversation_without_redis.turns) == 1
    turn = conversation_without_redis.turns[0]
    assert turn.question == "What is your purpose?"
    assert turn.sequence == 0

def test_conversation_add_turn_with_redis(conversation_with_redis, mock_redis_client):
    # Setup mock_data for existing conversation state (optional)
    mock_redis_client.get.return_value = json.dumps([])  # No prior turns

    conversation_with_redis.add_turn(
        question="How are you?",
        detected_entities=["state"],
        context=["State context"],
        response="I am fine.",
        request_id="req_003"
    )
    assert len(conversation_with_redis.turns) == 1
    turn = conversation_with_redis.turns[0]
    assert turn.question == "How are you?"
    assert turn.sequence == 0
    mock_redis_client.set.assert_called_once()
    key = f"conversation:{conversation_with_redis.conversation_id}"
    data = json.dumps([turn.to_json()])
    mock_redis_client.set.assert_called_with(key, data, ex=3600)

def test_conversation_get_history(conversation_without_redis):
    conversation_without_redis.add_turn(
        question="What is your name?",
        detected_entities=["name"],
        context=["Standard prompt"],
        response="I am Story Sage.",
        request_id="req_001"
    )
    conversation_without_redis.add_turn(
        question="What can you do?",
        detected_entities=["capabilities"],
        context=["Capabilities context"],
        response="I can assist with your stories.",
        request_id="req_002"
    )
    history = conversation_without_redis.get_history()
    expected_history = [
        ("What is your name?", ["name"], ["Standard prompt"], "I am Story Sage.", "req_001"),
        ("What can you do?", ["capabilities"], ["Capabilities context"], "I can assist with your stories.", "req_002")
    ]
    assert history == expected_history

def test_conversation_str(conversation_without_redis):
    conversation_without_redis.add_turn(
        question="What is your name?",
        detected_entities=["name"],
        context=["Standard prompt"],
        response="I am Story Sage.",
        request_id="req_001"
    )
    conversation_without_redis.add_turn(
        question="What can you do?",
        detected_entities=["capabilities"],
        context=["Capabilities context"],
        response="I can assist with your stories.",
        request_id="req_002"
    )
    expected_str = "HUMAN: What is your name?\nCOMPUTER: I am Story Sage.\nHUMAN: What can you do?\nCOMPUTER: I can assist with your stories."
    assert str(conversation_without_redis) == expected_str

def test_conversation_get_log(conversation_without_redis):
    conversation_without_redis.add_turn(
        question="What is your name?",
        detected_entities=["name"],
        context=["Standard prompt"],
        response="I am Story Sage.",
        request_id="req_001"
    )
    log = conversation_without_redis.get_log()
    expected_log = json.dumps([
        {
            'question': "What is your name?",
            'detected_entities': ["name"],
            'context': ["Standard prompt"],
            'response': "I am Story Sage.",
            'request_id': "req_001",
            'sequence': 0
        }
    ], indent=4)
    assert log == expected_log

def test_conversation_load_from_cache(conversation_with_redis, mock_redis_client):
    # Setup mock data to be returned by Redis 'get' method
    past_turns = [
        {
            'question': "What is your name?",
            'detected_entities': ["name"],
            'context': ["Standard prompt"],
            'response': "I am Story Sage.",
            'request_id': "req_001",
            'sequence': 0
        }
    ]
    mock_redis_client.get.return_value = json.dumps(past_turns)
    
    conversation = StorySageConversation(conversation_id=str(conversation_with_redis.conversation_id), redis=mock_redis_client)
    assert len(conversation.turns) == 1
    turn = conversation.turns[0]
    assert turn.question == "What is your name?"
    assert turn.sequence == 0

def test_conversation_persistence_failure(conversation_with_redis, mock_redis_client):
    mock_redis_client.set.side_effect = Exception("Redis set failed")
    with pytest.raises(Exception) as exc_info:
        conversation_with_redis.add_turn(
            question="Test persistence failure",
            detected_entities=["test"],
            context=["Failure context"],
            response="This should fail.",
            request_id="req_004"
        )
    assert "Redis set failed" in str(exc_info.value)

def test_conversation_initialization_without_conversation_id():
    """Test that initializing without a conversation_id sets a valid UUID and empty turns."""
    conversation = StorySageConversation()
    assert conversation.conversation_id is not None
    assert isinstance(conversation.conversation_id, uuid.UUID)
    assert len(conversation.turns) == 0