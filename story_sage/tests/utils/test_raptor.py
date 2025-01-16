import pytest
from story_sage.utils.raptor import *

def test_chunk_metadata_init():
    # Test full initialization
    metadata = ChunkMetadata(
        chunk_index=1,
        book_number=2,
        chapter_number=3,
        level=4
    )
    
    assert metadata.chunk_index == 1
    assert metadata.book_number == 2
    assert metadata.chapter_number == 3
    assert metadata.level == 4

    # Test minimal initialization
    metadata = ChunkMetadata(chunk_index=1)
    assert metadata.chunk_index == 1
    assert metadata.book_number is None
    assert metadata.chapter_number is None
    assert metadata.level is None

def test_chunk_metadata_to_dict():
    metadata = ChunkMetadata(
        chunk_index=1,
        book_number=2,
        chapter_number=3,
        level=4
    )
    
    metadata_dict = metadata.__to_dict__()
    
    assert isinstance(metadata_dict, dict)
    assert metadata_dict == {
        "chunk_index": 1,
        "book_number": 2,
        "chapter_number": 3,
        "level": 4
    }

    # Test with minimal data
    metadata = ChunkMetadata(chunk_index=1)
    metadata_dict = metadata.__to_dict__()
    
    assert metadata_dict == {
        "chunk_index": 1,
        "book_number": None,
        "chapter_number": None,
        "level": None
    }

def test_chunk_init():
    # Test with ChunkMetadata object
    metadata = ChunkMetadata(
        chunk_index=1,
        book_number=2,
        chapter_number=3,
        level=4
    )
    
    chunk = Chunk(
        text="Test text",
        metadata=metadata,
        is_summary=True,
        embedding=[0.1, 0.2, 0.3]
    )
    
    assert chunk.text == "Test text"
    assert chunk.is_summary == True
    assert chunk.embedding == [0.1, 0.2, 0.3]
    assert isinstance(chunk.metadata, ChunkMetadata)
    assert chunk.chunk_key == "book_2|chapter_3|level_4|chunk_1"
    assert chunk.parents == []
    assert chunk.children == []

    # Test with dictionary metadata
    chunk = Chunk(
        text="Test text",
        metadata={
            "chunk_index": 1,
            "book_number": 2,
            "chapter_number": 3,
            "level": 4
        }
    )
    
    assert isinstance(chunk.metadata, ChunkMetadata)
    assert chunk.chunk_key == "book_2|chapter_3|level_4|chunk_1"

def test_chunk_create_key():
    # Test full key generation
    chunk = Chunk(
        text="Test",
        metadata=ChunkMetadata(
            chunk_index=1,
            book_number=2,
            chapter_number=3,
            level=4
        )
    )
    assert chunk.chunk_key == "book_2|chapter_3|level_4|chunk_1"

    # Test minimal key generation
    chunk = Chunk(
        text="Test",
        metadata=ChunkMetadata(chunk_index=1)
    )
    assert chunk.chunk_key == "chunk_1"

def test_chunk_to_json():
    chunk = Chunk(
        text="Test text",
        metadata=ChunkMetadata(
            chunk_index=1,
            book_number=2,
            chapter_number=3,
            level=4
        )
    )
    
    json_data = chunk.__json__()
    
    assert isinstance(json_data, dict)
    assert json_data["text"] == "Test text"
    assert json_data["chunk_key"] == "book_2|chapter_3|level_4|chunk_1"
    assert json_data["parents"] == []
    assert json_data["children"] == []
    assert json_data["metadata"] == {
        "chunk_index": 1,
        "book_number": 2,
        "chapter_number": 3,
        "level": 4
    }

def test_chunk_invalid_metadata():
    with pytest.raises(ValueError, match="metadata must be a dictionary or ChunkMetadata object"):
        Chunk(
            text="Test",
            metadata="invalid"  # Invalid metadata type
        )
