from dataclasses import dataclass, field
from typing import List, TypedDict, Literal, Tuple
from ..story_sage_conversation import StorySageConversation

@dataclass
class StorySageState():

    question: str
    book_number: int
    chapter_number: int
    series_id: int
    context_filters: dict = field(default_factory=dict)
    initial_context: List[dict] = field(default_factory=list)
    target_ids: List[str] = field(default_factory=list)
    # List of chunk IDs to use as context
    context: List[Tuple[str, str, str, str]] = field(default_factory=list)
    # List of tuples containing (chunk_id, book_number, chapter_number, full_chunk)
    answer: str = None
    entities: List[str] = field(default_factory=list)
    conversation: StorySageConversation = None
    node_history: List[str] = field(default_factory=list)
    tokens_used: int = 0