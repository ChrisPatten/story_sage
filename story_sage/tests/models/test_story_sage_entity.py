import pytest
from story_sage.models.story_sage_entity import (
    StorySageEntity,
    StorySageEntityGroup,
    StorySageEntityCollection
)

@pytest.fixture
def entity():
    return StorySageEntity(entity_name="John Doe", entity_type="character")

@pytest.fixture
def another_entity():
    return StorySageEntity(entity_name="Jane Smith", entity_type="character")

@pytest.fixture
def entity_group(entity, another_entity):
    return StorySageEntityGroup(entities=[entity, another_entity])

def test_story_sage_entity_initialization(entity):
    assert entity.entity_name == "John Doe"
    assert entity.entity_type == "character"
    assert entity.entity_group_id is None
    assert len(entity.entity_id) == 32  # MD5 hash length

def test_story_sage_entity_str(entity):
    assert str(entity) == "Entity: John Doe (character)"

def test_story_sage_entity_to_dict(entity):
    expected_dict = {
        'entity_name': "John Doe",
        'entity_type': "character",
        'entity_id': entity.entity_id,
        'entity_group_id': None
    }
    assert entity.to_dict() == expected_dict

def test_story_sage_entity_group_initialization(entity_group):
    assert len(entity_group) == 2
    assert entity_group.entity_group_id is not None
    for entity in entity_group:
        assert entity.entity_group_id == entity_group.entity_group_id

def test_story_sage_entity_group_add_entity(entity_group, another_entity):
    new_entity = StorySageEntity(entity_name="Alice Wonderland", entity_type="character")
    entity_group.add_entity(new_entity)
    assert len(entity_group) == 3
    assert new_entity.entity_group_id == entity_group.entity_group_id

def test_story_sage_entity_group_remove_entity(entity_group):
    entity_id_to_remove = entity_group.entities[0].entity_id
    entity_group.remove_entity_by_id(entity_id_to_remove)
    assert len(entity_group) == 1
    assert all(e.entity_id != entity_id_to_remove for e in entity_group.entities)

def test_story_sage_entity_group_get_names(entity_group):
    names = entity_group.get_names()
    assert "John Doe" in names
    assert "Jane Smith" in names

def test_story_sage_entity_group_to_json(entity_group):
    group_json = entity_group.to_json()
    assert isinstance(group_json, dict)
    assert group_json['entity_group_id'] == entity_group.entity_group_id
    assert len(group_json['entities']) == 2

def test_story_sage_entity_collection_initialization():
    collection = StorySageEntityCollection()
    assert len(collection) == 0

def test_story_sage_entity_collection_add_group(entity_group):
    collection = StorySageEntityCollection()
    collection.add_entity_group(entity_group)
    assert len(collection) == 1
    assert collection.entity_groups[0] == entity_group

def test_story_sage_entity_collection_remove_group(entity_group):
    collection = StorySageEntityCollection(entity_groups=[entity_group])
    collection.remove_entity_group(entity_group.entity_group_id)
    assert len(collection) == 0

def test_story_sage_entity_collection_get_names_by_group_id(entity_group):
    collection = StorySageEntityCollection(entity_groups=[entity_group])
    names_by_id = collection.get_names_by_group_id()
    assert entity_group.entity_group_id in names_by_id
    assert "John Doe" in names_by_id[entity_group.entity_group_id]
    assert "Jane Smith" in names_by_id[entity_group.entity_group_id]

def test_story_sage_entity_collection_merge_groups(entity_group):
    collection = StorySageEntityCollection()
    collection.add_entity_group(entity_group)
    
    new_entity = StorySageEntity(entity_name="Alice Wonderland", entity_type="character")
    new_group = StorySageEntityGroup(entities=[new_entity])
    collection.add_entity_group(new_group)
    
    merged_group = StorySageEntityCollection.merge_groups(entity_group, new_group)
    assert len(merged_group) == 3
    assert new_entity.entity_group_id == entity_group.entity_group_id
    assert merged_group in collection.entity_groups

def test_story_sage_entity_collection_to_json(entity_group):
    collection = StorySageEntityCollection(entity_groups=[entity_group])
    json_str = collection.to_json()
    assert isinstance(json_str, str)

def test_story_sage_entity_collection_from_json(entity_group):
    collection = StorySageEntityCollection(entity_groups=[entity_group])
    json_str = collection.to_json()
    new_collection = StorySageEntityCollection.from_json(json_str)
    assert len(new_collection) == 1
    new_group = new_collection.entity_groups[0]
    assert new_group.entity_group_id == entity_group.entity_group_id
    assert len(new_group) == 2

def test_story_sage_entity_collection_get_all_entities(entity_group):
    collection = StorySageEntityCollection(entity_groups=[entity_group])
    all_entities = collection.get_all_entities()
    assert len(all_entities) == 2
    assert all_entities[0].entity_name == "John Doe"
    assert all_entities[1].entity_name == "Jane Smith"
