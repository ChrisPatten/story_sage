"""Search module for finding relevant text chunks using multiple search strategies.

This module implements various text search algorithms including exact matching,
phrase matching, proximity search, and fuzzy search. It supports parallelized 
processing for improved performance on large document collections.

The search strategies are:
- EXACT: Direct word-for-word matching
- PHRASE: Multi-word phrase matching  
- PROXIMITY: Sliding window word co-occurrence
- FUZZY: Approximate string matching using Levenshtein distance

Example:
    ```python
    search = StorySageSearch(threshold=0.25, max_results=20)
    chunks = [Chunk("some text...", metadata={})]
    
    # Perform exact match search
    results = search.search("search text", chunks, strategy=SearchStrategy.EXACT)
    
    # Try phrase matching
    results = search.search("specific phrase", chunks, strategy=SearchStrategy.PHRASE)
    ```
"""

import re
from typing import Dict, List, Optional, Tuple, Set, Generator, Union
import logging
from dataclasses import dataclass
from enum import Enum
from ..models.chunk import Chunk, ChunkMetadata
from functools import lru_cache
from multiprocessing import Pool
import numpy as np

@dataclass
class SearchResult:
    """Container for search result data.
    
    Attributes:
        chunk_id: Unique identifier for the chunk
        content: Text content of the matched chunk
        score: Relevance score from 0-1
        metadata: Additional chunk metadata
    """
    chunk_id: str
    content: str
    score: float
    metadata: Dict

class SearchStrategy(Enum):
    """Enumeration of available search strategies.
    
    Values:
        EXACT: Direct word matching
        PHRASE: Multi-word phrase matching
        PROXIMITY: Sliding window co-occurrence
        FUZZY: Approximate string matching
    """
    EXACT = "exact"
    PHRASE = "phrase"
    PROXIMITY = "proximity"
    FUZZY = "fuzzy"

class StorySageSearch:
    """Search engine implementing multiple text search strategies.

    This class provides methods for searching text chunks using different
    matching algorithms. It supports exact matching, phrase matching,
    proximity search with sliding windows, and fuzzy string matching.

    Results are filtered by a relevance threshold and limited to a maximum
    count. If a strategy finds no results, it can fall back to fuzzy search.

    Attributes:
        threshold (float): Minimum relevance score (0-1) for results
        max_results (int): Maximum number of results to return
        logger: Logger instance for this class

    Example:
        ```python
        search = StorySageSearch(threshold=0.25, max_results=20)
        chunks = [Chunk("example text", metadata={})]
        results = search.search("query", chunks, strategy=SearchStrategy.EXACT)
        ```
    """

    def __init__(self, threshold: float = 0.25, max_results: int = 20):
        """Initialize search engine with configurable parameters.

        Args:
            threshold: Minimum relevance score (0-1) for results. Default 0.25
            max_results: Maximum results to return. Default 20
        """
        self.logger = logging.getLogger(__name__)
        self.threshold = threshold
        self.max_results = max_results

    def search(self, text: str, chunks: List[Chunk], 
               strategy: SearchStrategy = SearchStrategy.EXACT) -> List[Chunk]:
        """Search chunks using specified strategy.

        Executes search using selected strategy and falls back to fuzzy search
        if no results are found (unless fuzzy was the original strategy).

        Args:
            text: Search query text
            chunks: List of Chunk objects to search
            strategy: Search strategy to use. Default EXACT

        Returns:
            List[Chunk]: Matching chunks ordered by relevance score,
                        limited by max_results
        """
        self.logger.info(f'Searching for "{text}" using strategy {strategy.value}')
        result: List[Chunk] = []
        if strategy == SearchStrategy.EXACT:
            result = self._exact_search(text, chunks)
        elif strategy == SearchStrategy.PHRASE:
            result = self._phrase_search(text, chunks)
        elif strategy == SearchStrategy.PROXIMITY:
            result = self._proximity_search(text, chunks)
        elif strategy == SearchStrategy.FUZZY:
            result = self._fuzzy_search(text, chunks)
        
        if len(result) == 0:
            if strategy != SearchStrategy.FUZZY:
                self.logger.warning(f"No results found using {strategy.value} search. Trying fuzzy search...")
                result = self._fuzzy_search(text, chunks)
        else:
            self.logger.info(f"Found {len(result)} results using {strategy.value} search")
        
        return result[:self.max_results]

    def _exact_search(self, text: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        Performs an exact keyword matching search on a list of text chunks.
        This method splits both the search text and chunk texts into individual words,
        and finds matches based on word overlap. The matching is case-insensitive.
        Args:
            text (str): The search query text to match against chunks
            chunks (List[Chunk]): List of Chunk objects to search through
        Returns:
            List[Chunk]: A sorted list of matching Chunk objects, ordered by relevance score.
                         Score is calculated as (number of matching words) / (total keywords in search)
        Note:
            - The search is case-insensitive
            - Chunks are scored based on the proportion of search keywords they contain
            - Matching chunks have their metadata updated with:
                - match_type: "exact"
                - matched_terms: number of matching words
        """
        results = []
        keywords = set(text.lower().split())
        self.logger.info(f"Exact search keywords: {keywords}")
        
        for chunk in chunks:
            chunk_words = set(chunk.text.lower().split())
            overlap = len(keywords.intersection(chunk_words))
            if overlap > 0:
                score = overlap / len(keywords)
                chunk.metadata.match_type = "exact"
                chunk.metadata.matched_terms = overlap
                results.append((chunk, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        result = [result[0] for result in results if result[1] > self.threshold]
        return result

    def _phrase_search(self, text: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        Performs phrase-based search on a list of text chunks.
        This method searches for specific phrases within each chunk of text and ranks the results
        based on the number of phrase matches found. The search is case-insensitive.
        Args:
            text (str): The search query text containing phrases to search for.
            chunks (List[Chunk]): A list of Chunk objects to search through.
        Returns:
            List[Chunk]: A sorted list of Chunk objects that contain any of the search phrases,
                ordered by relevance score (highest to lowest). Each returned chunk has its
                metadata updated with:
                - match_type: Set to "phrase"
                - matched_phrases: Number of phrases found in the chunk
        The relevance score is calculated as: (number of matched phrases) / (total number of search phrases)
        """
        results = []
        phrases = self._extract_phrases(text)
        self.logger.info(f"Phrase search phrases: {phrases}")
        
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
        result = [result[0] for result in results if result[1] > self.threshold]
        return result

    def _proximity_search(self, text: str, chunks: List[Chunk], window_size: int = 50) -> List[Chunk]:
        """
        Performs a proximity-based search on a list of text chunks.
        This method searches for keywords within a sliding window of text, ranking results based on
        how many keywords appear close together within the window. It's particularly useful for
        finding passages where search terms occur near each other.
        Parameters:
        ----------
        text : str
            The search query text containing keywords to search for
        chunks : List[Chunk]
            List of Chunk objects to search through, where each chunk contains text and metadata
        window_size : int, optional
            Size of the sliding window in words (default is 50)
        Returns:
        -------
        List[Chunk]
            A sorted list of Chunk objects that contain matches, ordered by relevance score.
            Each returned chunk includes metadata with:
            - match_type: "proximity"
            - window_size: size of window used for search
        Notes:
        -----
        - The search is case-insensitive
        - Scoring is based on the proportion of query keywords found within the window
        - Each chunk's score is determined by its best-scoring window
        - Results are sorted in descending order by score
        """
        results = []
        keywords = text.lower().split()
        self.logger.info(f"Proximity search keywords: {keywords}")
        
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
        result = [result[0] for result in results if result[1] > self.threshold]
        return result

    def _fuzzy_search(self, text: str, chunks: List[Chunk], threshold: float = 0.8) -> List[Chunk]:
        """Perform fuzzy string matching search.

        Uses parallel processing to match words using Levenshtein distance.
        Each chunk is scored based on ratio of matching keywords.

        Args:
            text: Search query text
            chunks: List of Chunk objects to search
            threshold: Minimum string similarity (0-1). Default 0.8

        Returns:
            List[Chunk]: Matching chunks ordered by score, filtered by
                class threshold and limited by max_results
        """
        if not text or not chunks:
            return []
            
        keywords = self._tokenize_and_normalize(text)
        
        # Prepare arguments for multiprocessing
        chunk_args = [
            (chunk, keywords, threshold, self._similar_enough) 
            for chunk in chunks
        ]
        
        # Process chunks in parallel
        with Pool() as pool:
            results = pool.map(self._process_chunk, chunk_args)
        
        # Filter and sort results
        valid_results = [(chunk, score) for chunk, score in results if score > self.threshold]
        valid_results.sort(key=lambda x: x[1], reverse=True)
        
        return [result[0] for result in valid_results]

    @staticmethod
    def _extract_phrases(text: str) -> List[str]:
        """Extract multi-word phrases from text.

        Finds quoted phrases and word groups of 2-4 words.

        Args:
            text: Input text to extract phrases from

        Returns:
            List[str]: Extracted phrases
        """
        return [p.strip() for p in re.findall(r'"([^"]*)"|\b\w+(?:\s+\w+){1,3}', text)]

    def _similar_enough(self, s1: str, s2: str, threshold: float) -> bool:
        """Check if strings are similar using Levenshtein distance.

        Args:
            s1: First string to compare
            s2: Second string to compare  
            threshold: Minimum similarity score (0-1)

        Returns:
            bool: True if normalized distance is >= threshold
        """
        if len(s1) == 0 or len(s2) == 0:
            return False
            
        # Calculate Levenshtein distance
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(dp[i-1][j] + 1,        # deletion
                              dp[i][j-1] + 1,        # insertion
                              dp[i-1][j-1] + cost)   # substitution
                
        distance = dp[m][n]
        max_len = max(len(s1), len(s2))
        similarity = 1 - (distance / max_len)
        
        return similarity >= threshold
    
    def _get_ids_and_docs(self, chunks: List[Chunk]) -> Generator[Tuple[str, str, Dict[str, Union[str, int]]], None, None]:
        """Generate tuples of chunk data.

        Args:
            chunks: List of Chunk objects

        Yields:
            Tuple containing (id, document, metadata) for each chunk
        """
        yield zip(chunks['ids'][0], chunks['documents'][0], chunks['metadatas'][0])

    @lru_cache(maxsize=1000)
    def _tokenize_and_normalize(self, text: str) -> Set[str]:
        """Tokenize and normalize text with caching.

        Splits text into lowercase words.

        Args:
            text: Input text to tokenize

        Returns:
            Set[str]: Normalized tokens
        """
        return set(text.lower().split())
    
    def _process_chunk(self, args: tuple) -> Tuple[Chunk, float]:
        """Process single chunk for parallel fuzzy search.

        Args:
            args: Tuple of (chunk, keywords, threshold, similar_enough_func)

        Returns:
            Tuple[Chunk, float]: Processed chunk and its relevance score
        """
        chunk, keywords, threshold, similar_enough_func = args
        
        # Get normalized text from chunk
        chunk_text = self._tokenize_and_normalize(chunk.text)
        
        # Count matching keywords
        matches = sum(1 for keyword in keywords 
                    if any(similar_enough_func(keyword, word, threshold) 
                        for word in chunk_text))
        
        # Calculate score as ratio of matches to total keywords
        score = matches / len(keywords) if keywords else 0.0
        
        # Set metadata
        chunk.metadata.update({
            'match_type': 'fuzzy',
            'threshold': threshold,
            'score': score
        })
        
        return chunk, score