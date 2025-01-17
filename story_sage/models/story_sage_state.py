from dataclasses import dataclass, field
from typing import List, Tuple
from .story_sage_conversation import StorySageConversation, StorySageContext


INPUT_TOKENS_CPM = 0.15
OUTPUT_TOKENS_CPM = 0.6
@dataclass
class StorySageState():
    """Represents the current state of a StorySage query session.
    
    This class maintains the state of a question-answering session, including the
    input question, context information, and generated results. It tracks both
    the query parameters and the response data.

    Attributes:
        question (str): The user's input question or query.
        book_number (int): The book number being queried.
        chapter_number (int): The chapter number being queried.
        series_id (int): Identifier for the book series.
        context_filters (dict): Filters to narrow down context search. Defaults to empty dict.
        initial_context (List[dict]): Initial context provided for the query. Defaults to empty list.
        target_ids (List[str]): List of specific chunk IDs to target. Defaults to empty list.
        context (List[Tuple[str, str, str, str]]): List of context tuples containing 
            (chunk_id, book_number, chapter_number, full_chunk). Defaults to empty list.
        answer (str): The generated answer to the question. Defaults to None.
        entities (List[str]): List of relevant entities extracted from the context. Defaults to empty list.
        conversation (StorySageConversation): Conversation history object. Defaults to None.
        node_history (List[str]): History of traversed nodes. Defaults to empty list.
        tokens_used (int): Count of tokens used in the current session. Defaults to 0.
        sort_order (str): How to sort retrieved chunks:
            - 'chronological': Sort by book/chapter ascending
            - 'reverse_chronological': Sort by book/chapter descending
            - None: Sort by similarity score (default)

    Example:
        >>> state = StorySageState(
        ...     question="Who is Harry Potter?",
        ...     book_number=1,
        ...     chapter_number=1,
        ...     series_id=42
        ... )
        >>> state.context_filters = {"character": "Harry Potter"}
        >>> state.tokens_used = 150
        >>> state.answer = "Harry Potter is a young wizard..."
    """

    question: str
    book_number: int
    chapter_number: int
    series_id: int
    summary_chunks: List[dict] = field(default_factory=list)  # RAPTOR top level summaries
    context_filters: dict = field(default_factory=dict)  # Filters for context retrieval
    initial_context: List[dict] = field(default_factory=list)  # Initial list of chunk IDs based on search of summaries
    target_ids: List[str] = field(default_factory=list)  # Specific chunks to get full text from
    context: List[StorySageContext] = field(default_factory=list)  # Retrieved context chunks
    answer: str = None  # Generated response
    search_query: str = None
    entities: List[str] = field(default_factory=list)  # Extracted relevant entities
    conversation: StorySageConversation = None  # Conversation history
    node_history: List[str] = field(default_factory=list)  # Track traversed nodes
    tokens_used: Tuple[int, int] = (0, 0)  # Token usage tracking
    needs_overview: bool = False  # Flag to indicate if an overview is needed
    sort_order: str = None  # Add this field for temporal query sorting

    def get_cost(self) -> str:
        cost = (self.tokens_used[0]/1000000*INPUT_TOKENS_CPM) + (self.tokens_used[1]/1000000*OUTPUT_TOKENS_CPM)
        self.tokens_used = (0, 0)
        return f"${cost:.4f}"