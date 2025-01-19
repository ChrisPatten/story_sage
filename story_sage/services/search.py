import re
from typing import Dict, List, Optional, Tuple, TypeAlias, Generator, Union
import logging
from dataclasses import dataclass
from enum import Enum
from ..models.chunk import Chunk, ChunkMetadata

@dataclass
class SearchResult:
    chunk_id: str
    content: str
    score: float
    metadata: Dict

class SearchStrategy(Enum):
    EXACT = "exact"
    PHRASE = "phrase"
    PROXIMITY = "proximity"
    FUZZY = "fuzzy"

class StorySageSearch:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def search(self, text: str, chunks: List[Chunk], strategy: SearchStrategy = SearchStrategy.EXACT) -> List[Chunk]:
        self.logger.debug(f'Searching for "{text}" using strategy {strategy.value}')
        if strategy == SearchStrategy.EXACT:
            return self._exact_search(text, chunks)
        elif strategy == SearchStrategy.PHRASE:
            return self._phrase_search(text, chunks)
        elif strategy == SearchStrategy.PROXIMITY:
            return self._proximity_search(text, chunks)
        elif strategy == SearchStrategy.FUZZY:
            return self._fuzzy_search(text, chunks)
        
        return self._exact_search(text, chunks)

    def _exact_search(self, text: str, chunks: List[Chunk]) -> List[Chunk]:
        results = []
        keywords = set(text.lower().split())
        self.logger.debug(f"Exact search keywords: {keywords}")
        
        for chunk in chunks:
            chunk_words = set(chunk.text.lower().split())
            overlap = len(keywords.intersection(chunk_words))
            if overlap > 0:
                score = overlap / len(keywords)
                chunk.metadata.match_type = "exact"
                chunk.metadata.matched_terms = overlap
                results.append((chunk, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [result[0] for result in results]

    def _phrase_search(self, text: str, chunks: List[Chunk]) -> List[Chunk]:
        results = []
        phrases = self._extract_phrases(text)
        self.logger.debug(f"Phrase search phrases: {phrases}")
        
        for chunk in chunks:
            content_lower = chunk.text.lower()
            matches = 0
            for phrase in phrases:
                if phrase.lower() in content_lower:
                    matches += 1
            
            if matches > 0:
                score = matches / len(phrases)
                chunk.metadata.match_type = "phrase"
                chunk.metadata.matched_phrases = matches
                results.append((chunk, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [result[0] for result in results]

    def _proximity_search(self, text: str, chunks: List[Chunk], window_size: int = 50) -> List[Chunk]:
        results = []
        keywords = text.lower().split()
        self.logger.debug(f"Proximity search keywords: {keywords}")
        
        for chunk in chunks:
            words = chunk.text.lower().split()
            max_score = 0
            
            for i in range(len(words) - window_size + 1):
                window = words[i:i + window_size]
                found_keywords = [k for k in keywords if k in window]
                score = len(found_keywords) / len(keywords)
                max_score = max(max_score, score)
            
            if max_score > 0:
                chunk.metadata.match_type = "proximity"
                chunk.metadata.window_size = window_size
                results.append((chunk, max_score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [result[0] for result in results]

    def _fuzzy_search(self, text: str, chunks: List[Chunk], threshold: float = 0.8) -> List[Chunk]:
        results = []
        keywords = text.lower().split()
        self.logger.debug(f"Fuzzy search keywords: {keywords}")
        
        for chunk in chunks:
            chunk_words = chunk.text.lower().split()
            matches = 0
            
            for keyword in keywords:
                for chunk_word in chunk_words:
                    if self._similar_enough(keyword, chunk_word, threshold):
                        matches += 1
                        break
            
            if matches > 0:
                score = matches / len(keywords)
                chunk.metadata.match_type = "fuzzy"
                chunk.metadata.threshold = threshold
                results.append((chunk, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [result[0] for result in results]

    def _extract_phrases(self, text: str) -> List[str]:
        # Simple phrase extraction - could be enhanced with NLP
        return [p.strip() for p in re.findall(r'"([^"]*)"|\b\w+(?:\s+\w+){1,3}', text)]

    def _similar_enough(self, s1: str, s2: str, threshold: float) -> bool:
        # Simple similarity check - could be replaced with proper Levenshtein
        if len(s1) == 0 or len(s2) == 0:
            return False
        return s1 in s2 or s2 in s1 or (
            abs(len(s1) - len(s2)) <= 2 and
            sum(a == b for a, b in zip(s1, s2)) / max(len(s1), len(s2)) >= threshold
        )
    
    def _get_ids_and_docs(self, chunks: List[Chunk]) -> Generator[Tuple[str, str, Dict[str, Union[str, int]]], None, None]:
        yield zip(chunks['ids'][0], chunks['documents'][0], chunks['metadatas'][0])
