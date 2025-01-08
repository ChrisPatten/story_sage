
import pytest
from story_sage.models.story_sage_state import StorySageState
from story_sage.models.story_sage_conversation import StorySageConversation
from typing import List, Tuple

@pytest.fixture
def conversation():
    """Fixture for creating a StorySageConversation instance."""
    return StorySageConversation()

def test_story_sage_state_initialization_required_fields():
    """Test initializing StorySageState with only required fields."""
    state = StorySageState(
        question="Who is Harry Potter?",
        book_number=1,
        chapter_number=1,
        series_id=42
    )
    assert state.question == "Who is Harry Potter?"
    assert state.book_number == 1
    assert state.chapter_number == 1
    assert state.series_id == 42
    assert state.context_filters == {}
    assert state.initial_context == []
    assert state.target_ids == []
    assert state.context == []
    assert state.answer is None
    assert state.entities == []
    assert state.conversation is None
    assert state.node_history == []
    assert state.tokens_used == 0

def test_story_sage_state_initialization_all_fields(conversation):
    """Test initializing StorySageState with all fields provided."""
    initial_context = [
        {
            'chunk_id': 'b1_ch1_001',
            'book_number': 1,
            'chapter_number': 1,
            'full_chunk': 'It was the best of times, it was the worst of times...'
        }
    ]
    context = [
        ("b1_ch1_001", "1", "1", "It was the best of times"),
        ("b1_ch1_002", "1", "1", "It was the worst of times")
    ]
    entities = ["Harry Potter", "Hermione Granger"]
    node_history = ["node1", "node2"]
    
    state = StorySageState(
        question="Summarize the first chapter.",
        book_number=1,
        chapter_number=1,
        series_id=42,
        context_filters={"environment": "forest"},
        initial_context=initial_context,
        target_ids=["b1_ch1_001", "b1_ch1_002"],
        context=context,
        answer="Harry Potter is the protagonist.",
        entities=entities,
        conversation=conversation,
        node_history=node_history,
        tokens_used=150
    )
    
    assert state.question == "Summarize the first chapter."
    assert state.book_number == 1
    assert state.chapter_number == 1
    assert state.series_id == 42
    assert state.context_filters == {"environment": "forest"}
    assert state.initial_context == initial_context
    assert state.target_ids == ["b1_ch1_001", "b1_ch1_002"]
    assert state.context == context
    assert state.answer == "Harry Potter is the protagonist."
    assert state.entities == ["Harry Potter", "Hermione Granger"]
    assert state.conversation == conversation
    assert state.node_history == ["node1", "node2"]
    assert state.tokens_used == 150

def test_story_sage_state_default_values():
    """Test that default values are correctly set when optional fields are not provided."""
    state = StorySageState(
        question="What is the capital of France?",
        book_number=2,
        chapter_number=3,
        series_id=7
    )
    assert state.context_filters == {}
    assert state.initial_context == []
    assert state.target_ids == []
    assert state.context == []
    assert state.answer is None
    assert state.entities == []
    assert state.conversation is None
    assert state.node_history == []
    assert state.tokens_used == 0

def test_story_sage_state_with_conversation(conversation):
    """Test initializing StorySageState with a conversation."""
    state = StorySageState(
        question="Explain the plot.",
        book_number=3,
        chapter_number=5,
        series_id=10,
        conversation=conversation
    )
    assert state.conversation == conversation

def test_story_sage_state_context_filters():
    """Test that context_filters are correctly assigned."""
    state = StorySageState(
        question="Describe the environment.",
        book_number=4,
        chapter_number=2,
        series_id=15,
        context_filters={"environment": "forest"}
    )
    assert state.context_filters == {"environment": "forest"}

def test_story_sage_state_initial_context():
    """Test that initial_context is correctly assigned."""
    initial_context = [
        {
            'chunk_id': 'b1_ch1_001',
            'book_number': 1,
            'chapter_number': 1,
            'full_chunk': 'It was the best of times, it was the worst of times...'
        }
    ]
    state = StorySageState(
        question="Summarize the first chapter.",
        book_number=1,
        chapter_number=1,
        series_id=42,
        initial_context=initial_context
    )
    assert state.initial_context == initial_context

def test_story_sage_state_target_ids():
    """Test that target_ids are correctly assigned."""
    target_ids = ["b1_ch1_001", "b1_ch2_005"]
    state = StorySageState(
        question="Find information about these sections.",
        book_number=1,
        chapter_number=2,
        series_id=42,
        target_ids=target_ids
    )
    assert state.target_ids == target_ids

def test_story_sage_state_context():
    """Test that context is correctly assigned."""
    context = [
        ("b1_ch1_001", "1", "1", "It was the best of times"),
        ("b1_ch1_002", "1", "1", "It was the worst of times")
    ]
    state = StorySageState(
        question="Describe the opening.",
        book_number=1,
        chapter_number=1,
        series_id=42,
        context=context
    )
    assert state.context == context

def test_story_sage_state_answer():
    """Test that answer is correctly assigned."""
    state = StorySageState(
        question="Who is the protagonist?",
        book_number=1,
        chapter_number=1,
        series_id=42,
        answer="Harry Potter is the protagonist."
    )
    assert state.answer == "Harry Potter is the protagonist."

def test_story_sage_state_entities():
    """Test that entities are correctly assigned."""
    entities = ["Harry Potter", "Hermione Granger", "Ron Weasley"]
    state = StorySageState(
        question="List the main characters.",
        book_number=1,
        chapter_number=1,
        series_id=42,
        entities=entities
    )
    assert state.entities == entities

def test_story_sage_state_node_history():
    """Test that node_history is correctly assigned."""
    node_history = ["node1", "node2", "node3"]
    state = StorySageState(
        question="What is the timeline?",
        book_number=1,
        chapter_number=1,
        series_id=42,
        node_history=node_history
    )
    assert state.node_history == node_history

def test_story_sage_state_tokens_used():
    """Test that tokens_used is correctly assigned."""
    state = StorySageState(
        question="Estimate tokens.",
        book_number=1,
        chapter_number=1,
        series_id=42,
        tokens_used=150
    )
    assert state.tokens_used == 150

def test_story_sage_state_partial_initialization():
    """Test initializing with some optional fields."""
    state = StorySageState(
        question="What is the significance of the wizarding world?",
        book_number=4,
        chapter_number=7,
        series_id=21,
        entities=["Wizarding World"]
    )
    assert state.question == "What is the significance of the wizarding world?"
    assert state.entities == ["Wizarding World"]
    # Ensure other optional fields have default values
    assert state.context_filters == {}
    assert state.initial_context == []
    assert state.target_ids == []
    assert state.context == []
    assert state.answer is None
    assert state.conversation is None
    assert state.node_history == []
    assert state.tokens_used == 0