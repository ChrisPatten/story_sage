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
        entity_settings = EntitySettings(**data['entity_settings'])
        books = [Book(**book) for book in data['books']]
        return cls(
            series_id=data['series_id'],
            series_name=data['series_name'],
            series_metadata_name=data['series_metadata_name'],
            entity_settings=entity_settings,
            books=books
        )
