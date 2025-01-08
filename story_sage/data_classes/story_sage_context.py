from dataclasses import dataclass

@dataclass
class StorySageContext:
    """Chunks that serve as context for generation of answers"""
    chunk_id: str
    book_number: int
    chapter_number: int
    chunk: str

    def __init__(self, data) -> 'StorySageContext':
        self.chunk_id = data['chunk_id']
        self.book_number = data['book_number']
        self.chapter_number = data['chapter_number']
        self.chunk = data['chunk']

    def format_for_llm(self) -> str:
        """Formats the context for input to the LLM"""
        return f"Book {self.book_number}, Chapter {self.chapter_number}: {self.chunk}"