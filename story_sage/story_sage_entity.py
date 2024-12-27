import hashlib
import json
from typing import List, Tuple, Set
from scipy.sparse import spmatrix

# Type alias for grouping data structure containing centroid, vectors, and string representations
GroupType = Tuple[spmatrix, spmatrix, Set[str]]

class StorySageEntity():
    """A class representing a single entity in the StorySage system.

    This class handles the creation and management of individual entities, including
    generating unique identifiers and managing group associations.

    Attributes:
        entity_name (str): The name/label of the entity
        entity_type (str): The classification type of the entity
        entity_group_id (str): The ID of the group this entity belongs to
        entity_id (str): Unique identifier generated from name and type

    Example:
        >>> entity = StorySageEntity("John Smith", "character")
        >>> print(entity.entity_name)
        'John Smith'
    """

    def __init__(self, entity_name: str, entity_type: str = 'entity'):
        """Initializes a StorySageEntity object.

        Args:
            entity_name (str): The name/label for the entity
            entity_type (str, optional): The type classification for the entity. Defaults to 'entity'.
        """
        self.entity_name: str = entity_name
        self.entity_type: str = entity_type
        self.entity_group_id: str = None
        # Generate the entity_id as the md5 hash of entity_name and entity_type
        hash_input = (entity_name + entity_type).encode('utf-8')
        self.entity_id = hashlib.md5(hash_input).hexdigest()


class StorySageEntityGroup():
    """A class representing a group of related entities in the StorySage system.

    This class manages collections of related entities, generating a unique group ID
    and maintaining entity relationships.

    Attributes:
        entities (List[StorySageEntity]): List of entities in this group
        entity_group_id (str): Unique identifier for the group

    Example:
        >>> entity1 = StorySageEntity("John Smith", "character")
        >>> entity2 = StorySageEntity("Johnny", "character")
        >>> group = StorySageEntityGroup([entity1, entity2])
    """

    def __init__(self, entities: List[StorySageEntity], entity_group_id: str = None):
        """Initializes a StorySageEntityGroup object.

        Args:
            entities (List[StorySageEntity]): List of StorySageEntity objects to include in the group
            entity_group_id (str, optional): Existing group ID to use. If None, generates new ID. Defaults to None.
        """
        self.entities: List[StorySageEntity] = entities
        if not entity_group_id:
            # Generate the entity_group_id as the md5 hash of the concatenated entity_ids
            hash_input = ''.join([entity.entity_id for entity in self.entities])
            self.entity_group_id = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        else:
            self.entity_group_id = entity_group_id

        for entity in self.entities:
            entity.entity_group_id = self.entity_group_id

    def add_entity(self, entity: StorySageEntity):
        """Adds a new entity to the existing group and updates its group association.

        Args:
            entity (StorySageEntity): The entity object to add to this group
        """
        self.entities.append(entity)
        entity.entity_group_id = self.entity_group_id

class StorySageEntityCollection():
    """A class managing collections of entity groups in the StorySage system.

    This class provides methods for managing multiple entity groups, including
    serialization, deserialization, and various lookup operations.

    Attributes:
        entity_groups (List[StorySageEntityGroup]): List of entity groups in the collection

    Example:
        >>> collection = StorySageEntityCollection()
        >>> group = StorySageEntityGroup([StorySageEntity("John Smith", "character")])
        >>> collection.add_entity_group(group)
    """

    def __init__(self, entity_groups: List[StorySageEntityGroup] = []):
        """Initializes a StorySageEntityCollection object.

        Args:
            entity_groups (List[StorySageEntityGroup], optional): Initial list of entity groups. Defaults to empty list.
        """
        self.entity_groups: List[StorySageEntityGroup] = entity_groups

    def add_entity_group(self, entity_group: StorySageEntityGroup):
        """Adds a new entity group to the collection.

        Args:
            entity_group (StorySageEntityGroup): The entity group to add to the collection
        """
        self.entity_groups.append(entity_group)

    def get_names_by_group_id(self):
        """Creates a mapping of group IDs to lists of entity names.

        Returns:
            dict: Dictionary where keys are group IDs and values are lists of entity names in that group

        Example:
            >>> collection.get_names_by_group_id()
            {'group123': ['John Smith', 'Johnny']}
        """
        output = {}
        for group in self.entity_groups:
            output[group.entity_group_id] = [entity.entity_name for entity in group.entities]

        return output
    
    def get_group_ids_by_name(self):
        """Creates a mapping of entity names to their corresponding group IDs.

        Returns:
            dict: Dictionary where keys are entity names and values are their group IDs

        Example:
            >>> collection.get_group_ids_by_name()
            {'John Smith': 'group123', 'Johnny': 'group123'}
        """
        output = {}
        for group in self.entity_groups:
            for entity in group.entities:
                output[entity.entity_name] = group.entity_group_id

        return output
    
    def to_json(self):
        """Converts the entity collection to a JSON string representation.

        Returns:
            str: JSON formatted string containing all entity and group data

        Example:
            >>> json_str = collection.to_json()
            >>> print(type(json_str))
            <class 'str'>
        """

        def entity_to_dict(entity):
            return {
                'entity_name': entity.entity_name,
                'entity_type': entity.entity_type,
                'entity_id': entity.entity_id,
                'entity_group_id': entity.entity_group_id
            }

        def group_to_dict(group):
            return {
                'entity_group_id': group.entity_group_id,
                'entities': [entity_to_dict(entity) for entity in group.entities]
            }

        collection_dict = {
            'entity_groups': [group_to_dict(group) for group in self.entity_groups]
        }

        return json.dumps(collection_dict, indent=4)
    
    @classmethod
    def from_json(cls, json_str: str):
        """Creates a StorySageEntityCollection instance from a JSON string.

        Args:
            json_str (str): JSON string representing an entity collection

        Returns:
            StorySageEntityCollection: New instance populated with data from JSON

        Example:
            >>> json_str = '{"entity_groups": [...]}'
            >>> collection = StorySageEntityCollection.from_json(json_str)
        """
        data = json.loads(json_str)

        def dict_to_entity(entity_dict):
            return StorySageEntity(
                entity_name=entity_dict['entity_name'],
                entity_type=entity_dict['entity_type']
            )

        def dict_to_group(group_dict):
            entities = [dict_to_entity(entity_dict) for entity_dict in group_dict['entities']]
            return StorySageEntityGroup(entities, entity_group_id=group_dict['entity_group_id'])

        entity_groups = [dict_to_group(group_dict) for group_dict in data['entity_groups']]
        return cls(entity_groups)
    
    @classmethod
    def from_sets(cls, entity_list: List[GroupType]):
        """Creates a StorySageEntityCollection from a list of entity groups.

        Args:
            entity_list (List[GroupType]): List of tuples containing centroids, vectors, and string sets

        Returns:
            StorySageEntityCollection: New instance populated with entities from the input groups

        Example:
            >>> groups = [(centroid1, vectors1, {"John", "Johnny"})]
            >>> collection = StorySageEntityCollection.from_sets(groups)
        """
        entity_groups = []
        for entry in entity_list:
            entities = [StorySageEntity(entity_name=entity) for entity in entry[2]]
            entity_group = StorySageEntityGroup(entities)
            entity_groups.append(entity_group)
        return cls(entity_groups)
    
    def get_all_entities(self) -> List[StorySageEntity]:
        """Retrieves all entities from all groups in the collection.

        Returns:
            List[StorySageEntity]: Flat list of all entities across all groups

        Example:
            >>> entities = collection.get_all_entities()
            >>> print(len(entities))
            5
        """
        all_entities = []
        for group in self.entity_groups:
            all_entities.extend(group.entities)
        return all_entities
    
    def merge_groups(self, group_id1: str, group_id2: str):
        """Merges two groups into one, combining their entities.

        Args:
            group_id1 (str): The ID of the first group to merge
            group_id2 (str): The ID of the second group to merge

        Raises:
            ValueError: If either group ID is not found in the collection
        """
        group1 = next((group for group in self.entity_groups if group.entity_group_id == group_id1), None)
        group2 = next((group for group in self.entity_groups if group.entity_group_id == group_id2), None)

        if not group1 or not group2:
            raise ValueError("One or both group IDs not found in the collection")

        # Merge entities from group2 into group1
        for entity in group2.entities:
            group1.add_entity(entity)

        # Remove group2 from the collection
        self.entity_groups.remove(group2)