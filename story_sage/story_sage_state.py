from typing import List, TypedDict

class StorySageState(TypedDict):
    """
    A typed dictionary representing the state of the Story Sage system.

    This class holds all the necessary information required during the processing
    of a user's question, including the question itself, context, and various
    metadata such as book and chapter numbers.
    """
    question: str
    """The user's question."""

    context: List[str]
    """A list of context strings retrieved based on the question and filters."""

    answer: str
    """The answer generated by the language model."""

    book_number: int
    """The number of the book in the series for context filtering."""

    chapter_number: int
    """The chapter number within the book for context filtering."""

    entities: List[str]
    """List of entity IDs extracted from the question."""

    series_id: int
    """The ID of the book series for context filtering."""