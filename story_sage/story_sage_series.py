from .story_sage_book import StorySageBook

class StorySageSeries:
    """
    A class to represent the metadata of a story series.

    Attributes:
        series_id (int): The unique identifier for the series.
        series_name (str): The name of the series.
        series_metadata_name (str): The metadata name used for the series.
        books (list[StorySageBook]): A list of StorySageBook objects, each representing a book in the series.
        entities (dict): A dictionary containing entity information for the series.
    """

    def __init__(self, series_id: str, series_name: str, series_metadata_name: str, books: list[StorySageBook], entities: dict):
        """
        Initialize the StorySageSeries with the given metadata.

        Args:
            series_id (str): The unique identifier for the series.
            series_name (str): The name of the series.
            series_metadata_name (str): The metadata name used for the series.
            books (list[StorySageBook]): A list of StorySageBook objects, each representing a book in the series.
            entities (dict): A dictionary containing entity information for the series.
        """
        self.series_id = str(series_id)
        self.series_name = series_name
        self.series_metadata_name = series_metadata_name
        self.books = books
        self.entities = entities
