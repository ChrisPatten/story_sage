"""Module for managing book series and their metadata in the StorySage application.

This module provides data classes for representing books, series, and collections of series,
along with functionality for JSON serialization and deserialization.

Example:
    Creating and using a StorySageSeries:
    
    >>> book = Book(
    ...     number_in_series=1,
    ...     title="The First Book",
    ...     book_metadata_name="first_book",
    ...     number_of_chapters=10
    ... )
    >>> entity_settings = EntitySettings(
    ...     names_to_skip=["Mr.", "Mrs."],
    ...     person_titles=["King", "Queen"]
    ... )
    >>> series = StorySageSeries(
    ...     series_id=1,
    ...     series_name="Amazing Series",
    ...     series_metadata_name="amazing_series",
    ...     entity_settings=entity_settings,
    ...     books=[book]
    ... )
    >>> metadata = series.to_metadata_json()
    >>> # Result:
    >>> # {
    >>> #     'series_id': 1,
    >>> #     'series_name': 'Amazing Series',
    >>> #     'series_metadata_name': 'amazing_series',
    >>> #     'books': [{
    >>> #         'number_in_series': 1,
    >>> #         'title': 'The First Book',
    >>> #         'book_metadata_name': 'first_book',
    >>> #         'number_of_chapters': 10
    >>> #     }]
    >>> # }
"""

from dataclasses import dataclass
from typing import List

@dataclass
class Book:
    """
    Represents a book in the series.

    Attributes:
        number_in_series (int): The book's position in the series.
        title (str): The title of the book.
        book_metadata_name (str): Metadata name associated with the book.
        number_of_chapters (int): Number of chapters in the book.
    """
    number_in_series: int
    title: str
    book_metadata_name: str
    number_of_chapters: int

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
            'number_of_chapters': self.number_of_chapters
        }

@dataclass
class EntitySettings:
    """
    Settings for entity extraction.

    Attributes:
        names_to_skip (List[str]): List of names to skip during extraction.
        person_titles (List[str]): List of person titles to remove.
    """
    names_to_skip: List[str]
    person_titles: List[str]

@dataclass
class StorySageSeries:
    """
    Represents a series of books.

    Attributes:
        series_id (int): Unique identifier for the series.
        series_name (str): Name of the series.
        series_metadata_name (str): Metadata name associated with the series.
        entity_settings (EntitySettings): Settings for entity extraction.
        books (List[Book]): List of books in the series.
    """
    series_id: int
    series_name: str
    series_metadata_name: str
    entity_settings: EntitySettings
    books: List[Book]
    
    @classmethod
    def from_dict(cls, data: dict) -> 'StorySageSeries':
        """
        Creates an instance of StorySageSeries from a dictionary.

        Args:
            data (dict): A dictionary containing series data.

        Returns:
            StorySageSeries: An instance of StorySageSeries.
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
    
class StorySageSeriesCollection():
    """
    Represents a collection of StorySageSeries.

    Attributes:
        series_list (List[StorySageSeries]): List of StorySageSeries instances.
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
