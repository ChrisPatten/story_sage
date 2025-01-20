"""Module for managing book series and their metadata in the StorySage application.

This module provides data classes and utilities for managing book series, including
serialization and deserialization of book metadata, series settings, and collections.

Classes:
    Book: Represents a single book within a series
    EntitySettings: Configuration for entity extraction from books
    StorySageSeries: Represents a complete book series with metadata
    StorySageSeriesCollection: Manages multiple series collections

Example:
    >>> # Create a new book series with entity settings
    >>> entity_settings = EntitySettings(
    ...     names_to_skip=["Mr.", "Mrs.", "Dr."],
    ...     person_titles=["King", "Queen", "Prince"]
    ... )
    >>> 
    >>> # Create books for the series
    >>> book1 = Book(
    ...     number_in_series=1,
    ...     title="The Beginning",
    ...     book_metadata_name="the_beginning",
    ...     number_of_chapters=12,
    ...     cover_image="beginning.jpg"
    ... )
    >>> book2 = Book(
    ...     number_in_series=2,
    ...     title="The Middle",
    ...     book_metadata_name="the_middle",
    ...     number_of_chapters=15
    ... )
    >>> 
    >>> # Create a series and add it to a collection
    >>> series = StorySageSeries(
    ...     series_id=1,
    ...     series_name="Epic Fantasy",
    ...     series_metadata_name="epic_fantasy",
    ...     entity_settings=entity_settings,
    ...     books=[book1, book2]
    ... )
    >>> 
    >>> collection = StorySageSeriesCollection()
    >>> collection.add_series(series)
    >>> 
    >>> # Convert to JSON format
    >>> json_data = collection.to_metadata_json()
    >>> print(json_data)
    {
        'series_list': [{
            'series_id': 1,
            'series_name': 'Epic Fantasy',
            'series_metadata_name': 'epic_fantasy',
            'books': [
                {
                    'number_in_series': 1,
                    'title': 'The Beginning',
                    'book_metadata_name': 'the_beginning',
                    'number_of_chapters': 12,
                    'cover_image': 'beginning.jpg'
                },
                {
                    'number_in_series': 2,
                    'title': 'The Middle',
                    'book_metadata_name': 'the_middle',
                    'number_of_chapters': 15,
                    'cover_image': None
                }
            ]
        }]
    }
"""

from dataclasses import dataclass
from typing import List

@dataclass
class Book:
    """Represents a single book within a series.

    Args:
        number_in_series (int): The book's position in the series order.
        title (str): The title of the book.
        book_metadata_name (str): Slug-style name for metadata files.
        number_of_chapters (int): Total number of chapters in the book.
        cover_image (str, optional): Path or URL to book cover image. Defaults to None.

    Example:
        >>> book = Book(
        ...     number_in_series=1,
        ...     title="First Adventure",
        ...     book_metadata_name="first_adventure",
        ...     number_of_chapters=10,
        ...     cover_image="cover.jpg"
        ... )
    """
    number_in_series: int
    title: str
    book_metadata_name: str
    number_of_chapters: int
    cover_image: str = None

    def to_json(self):
        """
        Converts the book to a JSON dictionary.

        Returns:
            dict: A dictionary representation of the book.
        """
        return {
            'number_in_series': self.number_in_series,
            'title': self.title,
            'book_metadata_name': self.book_metadata_name,
            'number_of_chapters': self.number_of_chapters,
            'cover_image': self.cover_image
        }

@dataclass
class EntitySettings:
    """Configuration settings for entity extraction from book content.

    Args:
        names_to_skip (List[str]): Titles or prefixes to ignore when extracting names.
        person_titles (List[str]): Character titles to be removed from extracted names.

    Example:
        >>> settings = EntitySettings(
        ...     names_to_skip=["Mr.", "Mrs.", "Dr."],
        ...     person_titles=["King", "Queen", "Prince"]
        ... )
    """
    names_to_skip: List[str]
    person_titles: List[str]

@dataclass
class StorySageSeries:
    """Represents a complete book series with its metadata and settings.

    Args:
        series_id (int): Unique identifier for the series.
        series_name (str): Display name of the series.
        series_metadata_name (str): Slug-style name for metadata files.
        entity_settings (EntitySettings): Configuration for entity extraction.
        books (List[Book]): Ordered list of books in the series.

    Example:
        >>> series = StorySageSeries(
        ...     series_id=1,
        ...     series_name="Epic Fantasy",
        ...     series_metadata_name="epic_fantasy",
        ...     entity_settings=EntitySettings(
        ...         names_to_skip=["Mr."],
        ...         person_titles=["King"]
        ...     ),
        ...     books=[Book(1, "Book One", "book_one", 10)]
        ... )
    """
    series_id: int
    series_name: str
    series_metadata_name: str
    entity_settings: EntitySettings
    books: List[Book]
    
    @classmethod
    def from_dict(cls, data: dict) -> 'StorySageSeries':
        """Creates a StorySageSeries instance from a dictionary.

        Args:
            data (dict): Dictionary containing series data with keys matching class attributes.

        Returns:
            StorySageSeries: New instance populated with the provided data.

        Example:
            >>> data = {
            ...     'series_id': 1,
            ...     'series_name': 'Epic Fantasy',
            ...     'series_metadata_name': 'epic_fantasy',
            ...     'entity_settings': {
            ...         'names_to_skip': ['Mr.'],
            ...         'person_titles': ['King']
            ...     },
            ...     'books': [{
            ...         'number_in_series': 1,
            ...         'title': 'Book One',
            ...         'book_metadata_name': 'book_one',
            ...         'number_of_chapters': 10
            ...     }]
            ... }
            >>> series = StorySageSeries.from_dict(data)
        """
        if 'entity_settings' in data:
            entity_settings = EntitySettings(**data['entity_settings'])
        else:
            entity_settings = EntitySettings(names_to_skip=[], person_titles=[])
        books = [Book(**book) for book in data['books']]
        return cls(
            series_id=data['series_id'],
            series_name=data['series_name'],
            series_metadata_name=data['series_metadata_name'],
            entity_settings=entity_settings,
            books=books
        )
    
    def to_metadata_json(self) -> dict:
        """
        Converts the series to a metadata JSON dictionary.

        Returns:
            dict: A dictionary representation of the series metadata.
        """
        return {
            'series_id': self.series_id,
            'series_name': self.series_name,
            'series_metadata_name': self.series_metadata_name,
            'books': [book.to_json() for book in self.books]
        }
    
class StorySageSeriesCollection:
    """Manages a collection of multiple book series.

    This class provides methods for adding, retrieving, and serializing collections
    of StorySageSeries instances.

    Example:
        >>> collection = StorySageSeriesCollection()
        >>> series = StorySageSeries(...)  # Create a series
        >>> collection.add_series(series)
        >>> json_data = collection.to_metadata_json()
    """
    def __init__(self):
        self.series_list: List[StorySageSeries] = []
    
    def add_series(self, series: StorySageSeries):
        """
        Adds a series to the collection.

        Args:
            series (StorySageSeries): The series to add.
        """
        self.series_list.append(series)
    
    def get_series_by_name(self, series_name: str) -> StorySageSeries:
        """
        Retrieves a series by name.

        Args:
            series_name (str): The name of the series to retrieve.

        Returns:
            StorySageSeries: The series with the given name.
        """
        for series in self.series_list:
            if series.series_name == series_name:
                return series
        return None
    
    def to_metadata_json(self) -> dict:
        """
        Converts the collection to a metadata JSON dictionary.

        Returns:
            dict: A dictionary representation of the series collection metadata.
        """
        return {
            'series_list': [series.to_metadata_json() for series in self.series_list]
        }
    
    @classmethod
    def from_metadata_json(cls, metadata: dict) -> 'StorySageSeriesCollection':
        """
        Creates an instance of StorySageSeriesCollection from a metadata JSON dictionary.

        Args:
            metadata (dict): A dictionary containing series collection metadata.

        Returns:
            StorySageSeriesCollection: An instance of StorySageSeriesCollection.
        """
        series_list = [StorySageSeries.from_dict(series) for series in metadata['series_list']]
        collection = cls()
        collection.series_list = series_list
        return collection
