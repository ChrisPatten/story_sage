
import pytest
from story_sage.models.story_sage_context import StorySageContext

@pytest.fixture
def context():
    return StorySageContext(
        chunk_id="b1_ch1_001",
        book_number=1,
        chapter_number=1,
        chunk="It was the best of times, it was the worst of times..."
    )

@pytest.fixture
def context_data():
    return {
        'chunk_id': 'b1_ch1_001',
        'book_number': 1,
        'chapter_number': 1,
        'chunk': 'It was the best of times, it was the worst of times...'
    }

def test_story_sage_context_initialization(context):
    assert context.chunk_id == "b1_ch1_001"
    assert context.book_number == 1
    assert context.chapter_number == 1
    assert context.chunk == "It was the best of times, it was the worst of times..."

def test_story_sage_context_from_dict(context_data):
    context = StorySageContext.from_dict(context_data)
    assert context.chunk_id == "b1_ch1_001"
    assert context.book_number == 1
    assert context.chapter_number == 1
    assert context.chunk == "It was the best of times, it was the worst of times..."

def test_story_sage_context_format_for_llm(context):
    formatted = context.format_for_llm()
    expected = "Book 1, Chapter 1: It was the best of times, it was the worst of times..."
    assert formatted == expected

def test_story_sage_context_invalid_initialization():
    with pytest.raises(TypeError):
        # Missing required fields
        StorySageContext(chunk_id="b1_ch1_002", book_number=1)