import hashlib
import json
from typing import List, Tuple, Set
from scipy.sparse import spmatrix

# Type alias for the complex grouping data structure
GroupType = Tuple[spmatrix, spmatrix, Set[str]]

class StorySageEntity():
    """A class representing a unique entity in the StorySage system.

    StorySageEntity represents atomic units of data like characters, locations, or items.
    Each entity has a unique identifier generated from its name and type, ensuring
    consistent identification across the system.

    Examples:
        >>> character = StorySageEntity("Luke Skywalker", "character")
        >>> print(character.entity_name)  # "Luke Skywalker"
        >>> print(character.entity_type)  # "character"
        >>> print(character.entity_id)    # "md5hash..."
        
        >>> location = StorySageEntity("Tatooine", "location")
        >>> print(location)  # "Entity: Tatooine (location)"

    Attributes:
        entity_name (str): The unique name or label of the entity.
        entity_type (str): The classification of the entity (e.g., "character", "location").
        entity_group_id (str): The ID of the group this entity belongs to (None if ungrouped).
        entity_id (str): A unique MD5 hash identifier generated from name and type.
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

    def __str__(self):
        """Returns a string representation of the StorySageEntity object."""
        return f"Entity: {self.entity_name} ({self.entity_type})"
    
    def to_dict(self) -> dict:
        """Converts the entity to a dictionary representation.

        Returns:
            dict: Dictionary containing entity data

        Example:
            >>> entity_dict = entity.to_dict()
            >>> print(type(entity_dict))
            <class 'dict'>
        """
        return {
            'entity_name': self.entity_name,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'entity_group_id': self.entity_group_id
        }


class StorySageEntityGroup():
    """A container class for managing related entities in the StorySage system.

    This class handles grouping of semantically similar or related entities. For example,
    different names referring to the same character ("Bob", "Robert", "Bobby") would be
    grouped together.

    Examples:
        >>> # Create a group of related character names
        >>> bob = StorySageEntity("Bob Smith", "character")
        >>> robert = StorySageEntity("Robert Smith", "character")
        >>> bobby = StorySageEntity("Bobby", "character")
        >>> group = StorySageEntityGroup([bob, robert, bobby])
        >>> print(len(group))  # 3
        >>> print(group.get_names())  # ['Bob Smith', 'Robert Smith', 'Bobby']
        
        >>> # Add a new alias to the group
        >>> rob = StorySageEntity("Rob", "character")
        >>> group.add_entity(rob)
        >>> print(len(group))  # 4

    Attributes:
        entities (List[StorySageEntity]): The list of related entities in this group.
        entity_group_id (str): A unique MD5 hash identifier for the entire group.
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

    def remove_entity_by_id(self, entity_id: str):
        """Removes an entity from the group based on its ID.

        Args:
            entity_id (str): The ID of the entity to remove

        Raises:
            ValueError: If the entity ID is not found in the group
        """
        entity = next((entity for entity in self.entities if entity.entity_id == entity_id), None)

        if not entity:
            raise ValueError("Entity ID not found in the group")

        self.entities.remove(entity)


    def get_names(self):
        """Returns a list of entity names in the group.

        Returns:
            List[str]: List of entity names in the group

        Example:
            >>> group.get_names()
            ['John Smith', 'Johnny']
        """
        return [entity.entity_name for entity in self.entities]

    def __iter__(self):
        """Allows iteration over all entities in the group."""
        for entity in self.entities:
            yield entity

    def __len__(self):
        """Returns the total number of entities in the group."""
        return len(self.entities)
    
    def __getitem__(self, index):
        """Allows subscriptable access to entities in the group by index."""
        return self.entities[index]
    
    def to_json(self) -> json:
        """Converts the entity group to a JSON string representation.

        Returns:
            str: JSON formatted string containing all entity data

        Example:
            >>> json_str = group.to_json()
            >>> print(type(json_str))
            <class 'str'>
        """

        group_dict = {
            'entity_group_id': self.entity_group_id,
            'entities': [entity.to_dict() for entity in self.entities]
        }

        return group_dict

class StorySageEntityCollection():
    """A high-level manager class for organizing multiple entity groups.

    This class provides functionality for managing the entire entity ecosystem,
    including operations like merging groups, serialization, and lookups.
    It's particularly useful for maintaining complex entity relationships
    and performing bulk operations.

    Examples:
        >>> # Create and populate a collection
        >>> collection = StorySageEntityCollection()
        >>> bob_group = StorySageEntityGroup([
        ...     StorySageEntity("Bob Smith", "character"),
        ...     StorySageEntity("Robert Smith", "character")
        ... ])
        >>> collection.add_entity_group(bob_group)
        
        >>> # Serialize the collection
        >>> json_data = collection.to_json()
        >>> new_collection = StorySageEntityCollection.from_json(json_data)
        
        >>> # Look up group memberships
        >>> name_groups = collection.get_names_by_group_id()
        >>> print(name_groups)  # {'group_id': ['Bob Smith', 'Robert Smith']}

    Attributes:
        entity_groups (List[StorySageEntityGroup]): The managed list of entity groups.
    """

    def __init__(self, entity_groups: List[StorySageEntityGroup] = []):
        """Initializes a StorySageEntityCollection object.

        Args:
            entity_groups (List[StorySageEntityGroup], optional): Initial list of entity groups. Defaults to empty list.
        """
        # Validate that entity_groups is a list of StorySageEntityGroup objects
        for group in entity_groups:
            if not isinstance(group, StorySageEntityGroup):
                raise ValueError("All elements in entity_groups must be of type StorySageEntityGroup")

        self.entity_groups: List[StorySageEntityGroup] = entity_groups

    def add_entity_group(self, entity_group: StorySageEntityGroup):
        """Adds a new entity group to the collection.

        Args:
            entity_group (StorySageEntityGroup): The entity group to add to the collection
        """
        self.entity_groups.append(entity_group)

    def remove_entity_group(self, entity_group_id: str):
        """Removes an entity group from the collection based on its ID.

        Args:
            entity_group_id (str): The ID of the group to remove

        Raises:
            ValueError: If the group ID is not found in the collection
        """
        group = next((group for group in self.entity_groups if group.entity_group_id == entity_group_id), None)

        if not group:
            raise ValueError("Group ID not found in the collection")

        self.entity_groups.remove(group)

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
        entity_groups = [cls._dict_to_group(group_dict) for group_dict in data['entity_groups']]
        return cls(entity_groups)
    
    @classmethod
    def from_dict(cls, data: dict):
        """Creates a StorySageEntityCollection instance from a dictionary.

        Args:
            data (dict): Dictionary containing entity group data

        Returns:
            StorySageEntityCollection: New instance populated with data from the dictionary

        Example:
            >>> data = {'entity_groups': [...]}
            >>> collection = StorySageEntityCollection.from_dict(data)
        """

        entity_groups = [cls._dict_to_group(group_dict) for group_dict in data['entity_groups']]
        return cls(entity_groups)
    
    @classmethod
    def from_sets(cls, entity_list: List[List[str]]):
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
            entities = [StorySageEntity(entity_name=entity) for entity in entry]
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
    
    @classmethod
    def merge_groups(cls, group_1: StorySageEntityGroup, group_2: StorySageEntityGroup) -> StorySageEntityGroup:
        """Combines two entity groups into one, preserving unique entities.

        Merges all entities from group_2 into group_1, avoiding duplicates based on
        entity names. This is useful for combining groups when new relationships
        are discovered.

        Args:
            group_1 (StorySageEntityGroup): The primary group that will contain all entities
            group_2 (StorySageEntityGroup): The secondary group whose entities will be merged into group_1

        Returns:
            StorySageEntityGroup: The merged group containing all unique entities

        Examples:
            >>> group1 = StorySageEntityGroup([StorySageEntity("Bob", "character")])
            >>> group2 = StorySageEntityGroup([StorySageEntity("Robert", "character")])
            >>> merged = StorySageEntityCollection.merge_groups(group1, group2)
            >>> print(len(merged))  # 2
        """

        # Merge entities from group2 into group1
        for entity in group_2.entities:
            if entity.entity_name not in [e.entity_name for e in group_1.entities]:
                group_1.add_entity(entity)
        
        return group_1

    @classmethod
    def _dict_to_entity(cls, entity_dict):
        return StorySageEntity(
            entity_name=entity_dict['entity_name'],
            entity_type=entity_dict['entity_type']
        )
    
    @classmethod
    def _dict_to_group(cls, group_dict):
        entities = [cls._dict_to_entity(entity_dict) for entity_dict in group_dict['entities']]
        return StorySageEntityGroup(entities, entity_group_id=group_dict['entity_group_id'])
    
    def __iter__(self):
        """Allows iteration over all entities in the collection."""
        for group in self.entity_groups:
            yield group

    def __len__(self):
        """Returns the total number of groups in the collection."""
        return len(self.entity_groups)
    
    def to_dict(self):
        """Converts the entity collection to a dictionary representation.

        Returns:
            dict: Dictionary containing all entity and group data

        Example:
            >>> data = collection.to_dict()
            >>> print(type(data))
            <class 'dict'>
        """
        return json.loads(self.to_json())