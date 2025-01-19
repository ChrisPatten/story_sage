from pydantic import BaseModel
from typing import List, Dict, Optional, Union

class ResponseData(BaseModel):
    response: str
    has_answer: bool
    follow_up: str

class ChunkEvaluationResult(BaseModel):
    chunk_scores: List[Dict[str, Union[str, int]]]
    secondary_query: Optional[str]

class KeywordsResult(BaseModel):
    keywords: List[str]

class QueryResult(BaseModel):
    query: str
    needs_overview: bool

class RefinedQuestionResult(BaseModel):
    refined_question: str

class TemporalQueryResult(BaseModel):
    is_temporal: bool
    query_type: str  # 'current', 'first', 'specific_point'
    book_number: Optional[int]
    chapter_number: Optional[int]
    book_position: Optional[int] 
    relative_position: Optional[str]  # 'before', 'after', 'at'

class SearchStrategyResult(BaseModel):
    strategy: str # 'exact', 'phrase', 'proximity', 'fuzzy'
    needs_similarity_search: bool

class SearchEvaluationResult(BaseModel):
    use_similarity: bool
    reasoning: str
