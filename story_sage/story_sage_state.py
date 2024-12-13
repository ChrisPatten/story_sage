from typing import List, TypedDict

class StorySageState(TypedDict):
    question: str
    context: List[str]
    answer: str
    book_number: int
    chapter_number: int
    n_chunks: int
    characters: List[str]