from uuid import uuid4
import json

class CharacterSummary:
    """A class to store and manage character information and their chapter summaries.

    This class maintains character details including their name, aliases, and summaries
    organized by book and chapter.

    Attributes:
        character_id (str): Unique identifier for the character.
        character_name (str): The primary name of the character.
        character_aliases (list[str]): Alternative names or nicknames for the character.
        chunk_summaries (dict): Nested dictionary storing character summaries by book and chapter.
    """
    
    def __init__(self, character_name: str, character_aliases: list[str] = None, 
                 character_id: str = None, chunk_summaries: dict[int, dict[int, list[str]]] = None):
        """Initialize a new CharacterSummary instance.

        Args:
            character_name (str): The primary name of the character.
            character_aliases (list[str], optional): List of alternative names. Defaults to None.
            character_id (str, optional): Unique identifier. Defaults to None (generates new UUID).
            chunk_summaries (dict, optional): Existing summaries to load. Defaults to None.
        """
        self.character_id = character_id or str(uuid4())
        self.character_name: str = character_name
        self.character_aliases: list[str] = character_aliases or [ character_name ]
        self.chunk_summaries: dict[int, dict[int, list[str]]] = chunk_summaries or {}

class CharacterCollection:
    """A collection manager for character summaries.

    This class manages multiple CharacterSummary instances and provides methods for
    adding, finding, and managing character information.

    Example:
        >>> collection = CharacterCollection()
        >>> collection.add_character("Harry Potter", ["The Boy Who Lived"])
        >>> collection.add_summary_to_character(
        ...     1, 1, 
        ...     {"character_name": "Harry Potter", 
        ...      "summary": "Discovers he's a wizard"}
        ... )
    """

    def __init__(self):
        """Initialize an empty character collection."""
        self.characters: dict[str, CharacterSummary] = {}

    def add_character(self, character_name: str, character_aliases: list[str] = None) -> CharacterSummary:
        """Add a new character to the collection.

        Args:
            character_name (str): The primary name of the character.
            character_aliases (list[str], optional): List of alternative names. Defaults to None.

        Returns:
            CharacterSummary: The newly created character instance.

        Example:
            >>> collection = CharacterCollection()
            >>> harry = collection.add_character("Harry Potter", ["The Boy Who Lived"])
        """
        new_character = CharacterSummary(character_name)
        self.characters[new_character.character_id] = new_character
        if character_aliases is not None:
            for alias in character_aliases:
                self.add_alias_to_character(character_name, alias)
        return new_character

    def add_summary_to_character(self, book_id: int, chapter_id: int, character_summary: dict[str, str]):
        """Add a summary for a character in a specific book and chapter.

        Args:
            book_id (int): The identifier for the book.
            chapter_id (int): The identifier for the chapter.
            character_summary (dict): Dictionary containing character_name and summary.

        Example:
            >>> collection.add_summary_to_character(
            ...     1, 1, 
            ...     {"character_name": "Harry Potter", 
            ...      "summary": "Learns about Hogwarts"}
            ... )
        """
        # Find or create the character
        target_character = self.find_character(character_summary['character_name'])
        if target_character is None:
            self.add_character(character_summary['character_name'])
            target_character = self.find_character(character_summary['character_name'])
            
        # Initialize nested dictionaries if they don't exist
        if book_id not in target_character.chunk_summaries:
            target_character.chunk_summaries[book_id] = {}
        if chapter_id not in target_character.chunk_summaries[book_id]:
            target_character.chunk_summaries[book_id][chapter_id] = []
            
        # Add the new summary
        target_character.chunk_summaries[book_id][chapter_id].append(character_summary['summary'])

    def add_alias_to_character(self, character_name: str, character_alias: str):
        """Add an alternative name for a character.

        Args:
            character_name (str): The primary name of the character.
            character_alias (str): The alternative name to add.
        """
        target_character = self.find_character(character_name)
        if target_character is None:
            self.add_character(character_name)
            target_character = self.find_character(character_name)
        target_character.character_aliases.append(character_alias)

    def find_character(self, character_name: str):
        """Find a character by their name or alias.

        Args:
            character_name (str): The name or alias to search for.

        Returns:
            CharacterSummary: The matching character or None if not found.

        Example:
            >>> character = collection.find_character("The Boy Who Lived")
            >>> print(character.character_name)
            'Harry Potter'
        """
        for _, character in self.characters.items():
            if str.lower(character.character_name) == str.lower(character_name):
                return character
            if str.lower(character_name) in [str.lower(alias) for alias in character.character_aliases]:
                return character
        return None
    
    def to_json(self):
        """Convert the collection to a JSON string.

        Returns:
            str: JSON representation of the collection.
        """
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    
    @staticmethod
    def from_json(json_string: str):
        """Create a CharacterCollection from a JSON string.

        Args:
            json_string (str): JSON string representing a CharacterCollection.

        Returns:
            CharacterCollection: New instance populated with the JSON data.

        Example:
            >>> json_str = collection.to_json()
            >>> new_collection = CharacterCollection.from_json(json_str)
        """
        dict_obj = json.loads(json_string)
        new_collection = CharacterCollection()
        for character_id, character_dict in dict_obj['characters'].items():
            new_collection.characters[character_id] = CharacterSummary(**character_dict)
        return new_collection
