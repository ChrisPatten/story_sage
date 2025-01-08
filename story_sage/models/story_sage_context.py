from dataclasses import dataclass

@dataclass
class StorySageContext:
    """A class representing contextual information from a book for story generation.

    This class holds information about specific chunks of text from books, including 
    identifiers for the chunk, book number, chapter number, and the actual text content.

    Attributes:
        chunk_id (str): Unique identifier for the text chunk.
        book_number (int): The book number in a series or collection.
        chapter_number (int): The chapter number within the book.
        chunk (str): The actual text content from the book.

    Example:
        >>> context = StorySageContext(
        ...     chunk_id="b1_ch1_001",
        ...     book_number=1,
        ...     chapter_number=1,
        ...     chunk="It was the best of times, it was the worst of times..."
        ... )
        >>> print(context.format_for_llm())
        'Book 1, Chapter 1: It was the best of times, it was the worst of times...'
    """
    chunk_id: str
    book_number: int
    chapter_number: int
    chunk: str

    @classmethod
    def from_dict(cls, data: dict) -> 'StorySageContext':
        """Creates a StorySageContext instance from a dictionary.

        Args:
            data (dict): A dictionary containing the required fields:
                - chunk_id (str)
                - book_number (int)
                - chapter_number (int)
                - chunk (str)

        Returns:
            StorySageContext: A new instance populated with the dictionary data.

        Example:
            >>> data = {
            ...     'chunk_id': 'b1_ch1_001',
            ...     'book_number': 1,
            ...     'chapter_number': 1,
            ...     'chunk': 'In a hole in the ground there lived a hobbit.'
            ... }
            >>> context = StorySageContext.from_dict(data)
        """
        return cls(
            chunk_id=data['chunk_id'],
            book_number=data['book_number'],
            chapter_number=data['chapter_number'],
            chunk=data['chunk']
        )

    def format_for_llm(self) -> str:
        """Formats the context for input to a Language Learning Model (LLM).

        Creates a formatted string containing the book number, chapter number,
        and the text chunk in a standardized format suitable for LLM input.

        Returns:
            str: A formatted string in the format "Book X, Chapter Y: [text chunk]"

        Example:
            >>> context = StorySageContext("b1_ch1_001", 1, 1, "Once upon a time...")
            >>> print(context.format_for_llm())
            'Book 1, Chapter 1: Once upon a time...'
        """
        return f"Book {self.book_number}, Chapter {self.chapter_number}: {self.chunk}"