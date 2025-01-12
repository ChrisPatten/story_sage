import pytest
from story_sage.utils.raptor import Raptor, ChunkMetadata, Chunk
from unittest.mock import Mock, patch
import os
from collections import OrderedDict


class MockChapter:
    def __init__(self, text):
        self.full_text = text

class MockBook:
    def __init__(self, book_number, chapters):
        self.book_number = book_number
        self.chapters = {i: MockChapter(text) for i, text in enumerate(chapters)}

@pytest.fixture
def mock_raptor():
    """Create a mock Raptor instance with minimal configuration."""
    with patch('story_sage.utils.raptor.StorySageConfig') as mock_config:
        mock_config.from_file.return_value = Mock(openai_api_key='dummy-key')
        return Raptor(config_path="dummy_path")

@pytest.fixture
def mock_book_data():
    """Create mock book data structure."""
    
    return {
        'book1.txt': MockBook(1, ['Chapter 1 text', 'Chapter 2 text']),
        'book2.txt': MockBook(2, ['Chapter 1 text'])
    }

def test_chunk_metadata_basic_instantiation():
    """Test basic instantiation of ChunkMetadata."""
    metadata = ChunkMetadata(chunk_index=1, book_number=2, chapter_number=3, level=1)
    
    assert metadata.chunk_index == 1
    assert metadata.book_number == 2
    assert metadata.chapter_number == 3
    assert metadata.level == 1

def test_chunk_metadata_with_none_values():
    """Test ChunkMetadata instantiation with None values."""
    metadata = ChunkMetadata(chunk_index=1)
    
    assert metadata.chunk_index == 1
    assert metadata.book_number is None
    assert metadata.chapter_number is None
    assert metadata.level is None

def test_chunk_metadata_attribute_updates():
    """Test that ChunkMetadata attributes can be updated."""
    metadata = ChunkMetadata(chunk_index=1)
    
    metadata.book_number = 2
    metadata.chapter_number = 3
    metadata.level = 1
    
    assert metadata.book_number == 2
    assert metadata.chapter_number == 3
    assert metadata.level == 1

def test_chunk_metadata_invalid_chunk_index():
    """Test that ChunkMetadata requires chunk_index."""
    with pytest.raises(TypeError):
        ChunkMetadata()

def test_chunk_metadata_type_checking():
    """Test type checking of ChunkMetadata attributes."""
    metadata = ChunkMetadata(chunk_index=1)
    
    metadata.book_number = "2"  # Should still work due to Python's dynamic typing
    metadata.chapter_number = 3.0  # Should still work due to Python's dynamic typing
    
    assert isinstance(metadata.book_number, str)
    assert isinstance(metadata.chapter_number, float)

def test_get_chunks_basic(mock_raptor: Raptor, mock_book_data: dict[str, MockBook]):
    """Test basic functionality of _get_chunks_from_filepath."""
    with patch.object(mock_raptor.chunker, 'read_text_files', return_value=mock_book_data):
        with patch.object(mock_raptor.chunker, 'sentence_splitter', return_value=['Chunk 1', 'Chunk 2']):
            result = mock_raptor._get_chunks_from_filepath("dummy_path")
            
            #assert isinstance(result, OrderedDict)
            assert len(result) == 2  # Two books
            assert len(result['book1.txt']) == 2  # Two chapters
            assert isinstance(result['book1.txt'][0][0], Chunk)
            assert result['book1.txt'][0][0].text == 'Chunk 1'

def test_get_chunks_metadata(mock_raptor: Raptor, mock_book_data: dict[str, MockBook]):
    """Test metadata handling in chunks."""
    with patch.object(mock_raptor.chunker, 'read_text_files', return_value=mock_book_data):
        with patch.object(mock_raptor.chunker, 'sentence_splitter', return_value=['Chunk text']):
            result = mock_raptor._get_chunks_from_filepath("dummy_path")
            chunk = result['book1.txt'][0][0]  # First chunk of first chapter of first book
            
            assert chunk.metadata.book_number == 1
            assert chunk.metadata.chapter_number == 0
            assert chunk.metadata.level == 1
            assert chunk.metadata.chunk_index == 0

def test_get_chunks_empty_file(mock_raptor: Raptor):
    """Test handling of empty input."""
    with patch.object(mock_raptor.chunker, 'read_text_files', return_value={}):
        with pytest.raises(ValueError, match="No text found in the provided file path"):
            mock_raptor._get_chunks_from_filepath("dummy_path")

def test_get_chunks_chunk_creation(mock_raptor: Raptor, mock_book_data: dict[str, MockBook]):
    """Test proper chunk creation and properties."""
    with patch.object(mock_raptor.chunker, 'read_text_files', return_value=mock_book_data):
        with patch.object(mock_raptor.chunker, 'sentence_splitter', return_value=['Chunk 1', 'Chunk 2']):
            result = mock_raptor._get_chunks_from_filepath("dummy_path")
            chunk = result['book1.txt'][0][0]  # First chunk
            
            assert hasattr(chunk, 'text')
            assert hasattr(chunk, 'metadata')
            assert hasattr(chunk, 'chunk_key')
            assert hasattr(chunk, 'parents')
            assert hasattr(chunk, 'children')
            assert chunk.parents == []
            assert chunk.children == []

def test_get_chunks_ordering(mock_raptor: Raptor):
    """Test that books and chapters are ordered correctly."""
    mock_data = {
        'book2.txt': Mock(book_number=2, chapters={1: Mock(full_text='text')}),
        'book1.txt': Mock(book_number=1, chapters={1: Mock(full_text='text')})
    }
    
    with patch.object(mock_raptor.chunker, 'read_text_files', return_value=mock_data):
        with patch.object(mock_raptor.chunker, 'sentence_splitter', return_value=['Chunk']):
            result = mock_raptor._get_chunks_from_filepath("dummy_path")
            books = list(result.keys())
            assert books == ['book1.txt', 'book2.txt']  # Should be sorted
