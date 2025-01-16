import pytest
from story_sage.models.story_sage_character import CharacterSummary, CharacterCollection

@pytest.fixture
def mock_character_summary():
    return {
        "character_name": "Harry Potter",
        "character_aliases": ["harry", "potter"],
        "character_id": "3f0253390ac5fa4923a56cfd1358219d",
        "chunk_summaries": {
            1: {
                1: ["The young wizard discovers his true identity"],
                2: ["Receives his Hogwarts letter"]
            }
        }
    }

def test_character_summary_initialization(mock_character_summary):
    character = CharacterSummary(**mock_character_summary)
    
    assert character.character_name == "Harry Potter"
    assert "harry" in character.character_aliases
    assert character.character_id == "3f0253390ac5fa4923a56cfd1358219d"
    assert character.chunk_summaries[1][1] == ["The young wizard discovers his true identity"]

def test_character_summary_default_values():
    character = CharacterSummary("Harry Potter")
    
    assert character.character_name == "Harry Potter"
    assert character.character_aliases == ["Harry Potter"]
    assert isinstance(character.character_id, str)
    assert character.chunk_summaries == {}

def test_character_collection_add_character():
    collection = CharacterCollection()
    character = collection.add_character("Voldemort", ["dark lord", "he who must not be named"])
    
    assert len(collection.characters) == 1
    assert character.character_name == "Voldemort"
    assert "dark lord" in character.character_aliases
    assert "he who must not be named" in character.character_aliases

def test_character_collection_add_summary():
    collection = CharacterCollection()
    collection.add_summary_to_character(1, 1, {
        "character_name": "Harry Potter",
        "summary": "The young wizard discovers his true identity"
    })
    
    character = collection.find_character("Harry Potter")
    assert character is not None
    assert character.chunk_summaries[1][1] == ["The young wizard discovers his true identity"]

def test_character_collection_find_character():
    collection = CharacterCollection()
    collection.add_character("Voldemort", ["dark lord", "he who must not be named"])
    
    # Test all aliases
    for name in ["Voldemort", "dark lord", "he who must not be named"]:
        character = collection.find_character(name)
        assert character is not None
        assert character.character_name == "Voldemort"
    
    assert collection.find_character("non-existent") is None

def test_character_collection_add_alias():
    collection = CharacterCollection()
    collection.add_character("Harry Potter")
    collection.add_alias_to_character("Harry Potter", "The Boy Who Lived")
    
    character = collection.find_character("Harry Potter")
    assert "The Boy Who Lived" in character.character_aliases

def test_case_insensitive_character_search():
    collection = CharacterCollection()
    collection.add_character("Harry Potter", ["The Boy Who Lived"])
    
    character1 = collection.find_character("harry potter")
    assert character1 is not None
    character2 = collection.find_character("THE BOY WHO LIVED")
    assert character2 is not None
