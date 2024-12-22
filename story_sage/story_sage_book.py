class StorySageBook:
    """
    A class to represent the metadata of a book in a story series.

    Attributes:
        number_in_series (int): The position of the book in the series.
        title (str): The title of the book.
        book_metadata_name (str): The metadata name used for the book.
        number_of_chapters (int): The number of chapters in the book.
    """

    def __init__(self, number_in_series: int, title: str, book_metadata_name: str, number_of_chapters: int):
        """
        Initialize the StorySageBook with the given metadata.

        Args:
            number_in_series (int): The position of the book in the series.
            title (str): The title of the book.
            book_metadata_name (str): The metadata name used for the book.
            number_of_chapters (int): The number of chapters in the book.
        """
        self.number_in_series = number_in_series
        self.title = title
        self.book_metadata_name = book_metadata_name
        self.number_of_chapters = number_of_chapters
        self.chapter_summaries = [None] * (number_of_chapters + 1)
