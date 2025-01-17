# Import necessary libraries and modules
import logging
import regex as re
from typing import List, Optional, Dict, Any, Tuple

from chromadb import QueryResult
from ..models import StorySageConfig, StorySageState, StorySageContext
from . import StorySageLLM, StorySageRetriever
from .story_sage_llm import _UsageType, ResponseData, ChunkEvaluationResult, KeywordsResult, QueryResult, RefinedQuestionResult
from ..utils import flatten_nested_list

# Constants
VALID_CHUNK_ID_PATTERN = re.compile(r'series_\d+\|book_\d+\|chapter_\d+\|level_\d+\|chunk_\d+')
RELEVANCE_SCORE_THRESHOLD = 6

"""
StorySageChain orchestrates the question-answering process for literary text analysis.
It manages the flow of retrieving relevant context and generating responses using LLM.

Example usage:
    config = StorySageConfig(
        chroma_path="path/to/db",
        chroma_collection="summaries",
        chroma_full_text_collection="full_text",
        n_chunks=3
    )
    
    state = StorySageState(
        question="Who is Sherlock Holmes?",
        series_id="sherlock_holmes",
        book_number=1
    )
    
    chain = StorySageChain(config, state)
    result_state = chain.invoke()
    print(result_state.answer)
    # Output: "Sherlock Holmes is a fictional detective created by Sir Arthur Conan Doyle..."
"""

class StorySageChain:
    """A chain that orchestrates the retrieval and response generation process.
    
    Args:
        config (StorySageConfig): Configuration object containing settings for the chain
        state (StorySageState): Current state of the conversation and context
        log_level (int): Logging level to use. Defaults to logging.WARN
        print_logs_to_console (bool): Whether to print logs to console. Defaults to False
    """
    logger: logging.Logger
    config: StorySageConfig
    entities: List[str]
    series_list: List[str]
    prompts: Dict[str, str]
    raptor_retriever: StorySageRetriever
    llm: StorySageLLM
    state: StorySageState
    needs_clarification: bool
    secondary_query: Optional[str]

    def __init__(self, config: StorySageConfig, state: StorySageState, 
                 log_level: int = logging.WARN, print_logs_to_console: bool = False):
        self.logger = self._setup_logger(log_level, print_logs_to_console)
        self.config = config
        self.entities = config.entities
        self.series_list = config.series
        self.prompts = config.prompts
        self.raptor_retriever = StorySageRetriever(
            chroma_path=config.chroma_path, 
            chroma_collection_name=config.raptor_collection, 
            n_chunks=config.n_chunks,
            logger=self.logger
        )
        self.llm = StorySageLLM(config, log_level=log_level)
        self.state = state
        self.needs_clarification = False
        self.secondary_query = None
        self._chunk_cache = {}  # Add LRU cache for frequently accessed chunks
        self._cache_size = 1000  # Maximum number of cached chunks

    def invoke(self) -> StorySageState:
        """Execute the chain to generate a response to the question in the current state."""
        self.logger.info(f"Invoking StorySageChain with question: '{self.state.question}'")
        self.state.node_history = []

        try:
            # Early exit for simple questions that don't need context
            if self._is_simple_question(self.state.question):
                self.logger.info('Question is simple and does not require context')
                if self._generate_simple_response():
                    return self.state

            # Get context filters and handle followup questions
            self._get_context_filters()
            if self._is_followup_question():
                if self._handle_followup_question():
                    return self.state

            # Main question processing pipeline
            if not self._process_main_question():
                self.state.answer = "I'm sorry, I couldn't find an answer to that question."

            return self.state

        except Exception as e:
            self.logger.error(f"Error in chain execution: {e}", exc_info=True)
            self.state.answer = "I encountered an error while processing your question."
            return self.state

    def _process_main_question(self) -> bool:
        """Process the main question through the retrieval and generation pipeline."""
        # Generate optimized search query
        if not self._generate_search_query():
            self.state.search_query = self.state.question

        # Get initial context and evaluate chunks in batches
        if not self._get_initial_context():
            return self._fallback_to_keyword_search()

        relevant_chunks = self._process_chunks_in_batches(self.state.summary_chunks)
        if not relevant_chunks:
            return self._fallback_to_keyword_search()

        # Retrieve and generate from relevant chunks
        self.state.target_ids = relevant_chunks
        if self._retrieve_by_ids() and self._generate():
            return True

        return self._fallback_to_keyword_search()

    def _process_chunks_in_batches(self, chunks: Dict[str, str], batch_size: int = 10) -> List[str]:
        """Process chunks in batches for more efficient evaluation."""
        relevant_chunks = []
        chunk_items = list(chunks.items())
        
        for i in range(0, len(chunk_items), batch_size):
            batch = dict(chunk_items[i:i + batch_size])
            batch_results, _ = self._evaluate_chunks(batch)
            relevant_chunks.extend(batch_results)
            
            # Early exit if we have enough highly relevant chunks
            if len(relevant_chunks) >= self.config.n_chunks:
                break

        return relevant_chunks

    def _handle_followup_question(self) -> bool:
        """Handle followup questions with context from previous interactions."""
        if self._expand_followup_question():
            # Check cache first
            cache_key = f"{self.state.series_id}_{self.state.question}"
            if cache_key in self._chunk_cache:
                self.state.context = self._chunk_cache[cache_key]
                return self._generate()

            if self._retrieve_from_followup_query():
                # Cache the results
                self._cache_chunks(cache_key, self.state.context)
                return self._generate()
        return False

    def _cache_chunks(self, key: str, chunks: List[StorySageContext]) -> None:
        """Cache chunks with LRU eviction."""
        if len(self._chunk_cache) >= self._cache_size:
            # Remove oldest item
            self._chunk_cache.pop(next(iter(self._chunk_cache)))
        self._chunk_cache[key] = chunks

    def _is_simple_question(self, question: str) -> bool:
        """Determine if a question can be answered without context."""
        simple_patterns = [
            r"^what (is|are|was|were) the",
            r"^(can|could) you explain",
            r"^tell me (about|what)",
        ]
        return any(re.match(pattern, question.lower()) for pattern in simple_patterns)

    def _generate_simple_response(self) -> bool:
        """Generate a response for simple questions without context."""
        try:
            response = self.llm.generate_response(
                context=[],
                question=self.state.question,
                conversation=self.state.conversation
            )
            if response[0].has_answer:
                self.state.answer = response[0].response
                return True
        except Exception as e:
            self.logger.error(f"Error generating simple response: {e}")
        return False

    def _fallback_to_keyword_search(self) -> bool:
        """Fallback strategy using keyword search."""
        self.logger.warning('Falling back to keyword search')
        return self._try_keyword_retrieval() and self._generate()

    # Internal methods for steps in the chain
    def _get_context_filters(self) -> None:
        """Initialize context filters based on current state for document retrieval."""
        self.logger.debug('Initializing context filters')
        self.state.node_history.append('GetContextFilters')
        self.state.context_filters = {
            'entities': self.state.entities,
            'series_id': self.state.series_id,
            'book_number': self.state.book_number,
            'chapter_number': self.state.chapter_number
        }
    
    def _get_initial_context(self) -> bool:
        """Retrieve context from the summary collection."""
        self.logger.debug('Retrieving summary context')
        self.state.node_history.append('GetSummaryContext')
        context_filters = self.state.context_filters.copy()
        context_filters['top_level_only'] = self.state.needs_overview
        try:
            self.state.summary_chunks = self.raptor_retriever.retrieve_chunks(
                query_str=self.state.search_query,
                context_filters=context_filters,
                n_results=25 if self.state.needs_overview else 10
            )
            self.logger.debug(f'Summary chunks retrieved: {len(self.state.summary_chunks)}')
            if len(self.state.summary_chunks) < 1:
                return False
            return True
        except Exception as e:
            self.logger.error(f'Error retrieving summary context: {e}')
            self.state.summary_chunks = []
            return False
        
    def _evaluate_chunks(self, context_chunks: Dict[str, str]) -> Tuple[List[str], Optional[str]]:
        """Combined chunk evaluation and selection."""
        result: ChunkEvaluationResult
        result, usage = self.llm.evaluate_chunks(
            context=context_chunks,
            question=self.state.question,
            conversation=self.state.conversation
        )
        
        # Filter chunks meeting threshold
        relevant_chunks = []
        for chunk_score in result.chunk_scores:
            chunk_id = chunk_score['chunk_id']
            score = chunk_score['score']
            if score >= RELEVANCE_SCORE_THRESHOLD and re.match(VALID_CHUNK_ID_PATTERN, chunk_id):
                relevant_chunks.append(chunk_id)
        self._add_usage(usage)
        return relevant_chunks, result.secondary_query

    def _retrieve_by_ids(self) -> bool:
        """Retrieve context using specific chunk IDs.
        
        Returns:
            bool: True if context was successfully retrieved and populated, False if no valid chunks found
        """
        self.logger.debug('Retrieving context by IDs')
        self.state.node_history.append('GetContextByIDs')
        try:
            results = self.raptor_retriever.retrieve_from_hierarchy(self.state.target_ids)
            if not results:
                self.logger.debug('No context found in hierarchy')
                return False
        except Exception as e:
            self.logger.error(f'Error retrieving context by IDs: {e}')
            return False
        
        if len(results['ids']) < 1:
            self.logger.debug('No context found by IDs')
            return False
        
        self.state.context = self._get_context_from_result(results)
        return True

    def _retrieve_from_query(self, exclude_summaries: bool = False) -> bool:
        """Retrieve context from either summary or full text collection.
        
        Args:
            exclude_summaries: Whether to exclude summary chunks from the search
            
        Returns:
            bool: True if context was successfully retrieved, False if no chunks found
        """
        self.logger.debug('Retrieving context from chunks')
        self.state.node_history.append('GetContextFromChunks')
        context_filters = self.state.context_filters.copy()
        if exclude_summaries:
            context_filters['exclude_summaries'] = True

        try:
            results = self.raptor_retriever.retrieve_chunks(
                query_str=self.state.search_query,
                context_filters=context_filters
            )
            self.logger.debug(f'Chunks retrieved: {results["ids"]}')
        except Exception as e:
            self.logger.error(f'Error retrieving context from chunks: {e}')
            return False

        if len(results['ids'][0]) < 1:
            self.logger.debug('No chunks found')
            return False
        
        self.state.context = self._get_context_from_result(results)
        return True

    def _get_context_from_keywords(self, keywords: List[str] = None) -> bool:
        """Retrieve context by searching for specific keywords.
        
        Args:
            keywords: List of keywords to search for. If None, uses words from question
            
        Returns:
            bool: True if context was successfully retrieved, False if no chunks found
        """
        self.logger.debug('Retrieving context from keywords')
        self.state.node_history.append('GetContextFromKeywords')
        if not keywords:
            keywords = self.state.question.split()
        try:
            results = self.raptor_retriever.get_by_keyword(
                keywords=keywords,
                context_filters=self.state.context_filters
            )
            self.logger.debug(f'Context retrieved by keywords: {results["ids"]}')
        except Exception as e:
            self.logger.error(f'Error retrieving context from keywords: {e}')
            return False

        if len(results['ids']) < 1:
            self.logger.debug('No context found by keywords')
            return False
        
        self.state.context = self._get_context_from_result(results)
        return True

    def _generate(self) -> bool:
        """Generate response using LLM based on current context and question.
        
        Returns:
            bool: True if generation was successful
        """
        self.logger.debug('Generating response')
        self.state.node_history.append('Generate')
        try:
            response_data: ResponseData
            response_data, tokens = self.llm.generate_response(
                context=self.state.context,
                question=self.state.question,
                conversation=self.state.conversation
            )
            self._add_usage(tokens)
            self.logger.debug(f'Response generated: {response_data.response}')
            if not response_data.has_answer and response_data.follow_up:
                self.needs_clarification = True
                self.state.answer = response_data.response + '\n\n\n' + response_data.follow_up
                self.logger.info('Clarifying question generated')
                return True
            
            self.needs_clarification = False
            self.state.answer = (response_data.response + '\n\n\n' + response_data.follow_up) if response_data.follow_up else response_data.response
            return response_data.has_answer
        except Exception as e:
            self.logger.error(f'Error generating response: {e}')
            return False

    def _get_context_from_result(self, results: Dict[str, List[Any]]) -> List[StorySageContext]:
        """Convert query results into StorySageContext objects.
        
        Args:
            results: Dictionary containing query results with 'ids' and 'metadatas' keys
            
        Returns:
            List[StorySageContext]: List of context objects sorted by chunk ID
        """
        self.logger.debug(f'Getting context from result: {results["ids"]}')
        try:
            results['ids'] = flatten_nested_list(results['ids'])
            results['metadatas'] = flatten_nested_list(results['metadatas'])
            
            for id, m in zip(results['ids'], results['metadatas']):
                m['id'] = id
            
            results = sorted(results['metadatas'], key=lambda x: (x['id']))

            return [
                StorySageContext.from_dict(
                    data = {
                        'chunk_id': m['id'], 
                        'book_number': m['book_number'], 
                        'chapter_number': m['chapter_number'], 
                        'chunk': m['full_chunk']
                    } 
                ) for m in results
            ]
        except KeyError as e:
            self.logger.error(f'Tried to access a key that doesn\'t exist in results[\'metadatas\']: {e}')
            return []
        except Exception as e:
            self.logger.error(f'Unexpected error processing results: {e}')
            return []
    
    def _add_usage(self, tokens: Tuple[int, int]) -> None:
        """Add token usage to the state object.
        
        Args:
            tokens: Tuple of (prompt_tokens, completion_tokens) counts
        """
        self.logger.debug(f'Adding token usage: {tokens}')
        self.state.tokens_used = (self.state.tokens_used[0] + tokens[0], self.state.tokens_used[1] + tokens[1])

    def _is_followup_question(self) -> bool:
        """Check if current question is a followup based on conversation history."""
        self.logger.debug('Checking if question is a followup')
        if self.state.conversation:
            return len(self.state.conversation.get_history()) > 0
        else:
            return False

    def _retrieve_from_followup_query(self) -> bool:
        """Get context using followup query based on conversation history.
        
        Returns:
            bool: True if context was found, False otherwise
        """
        self.logger.debug('Getting context from followup query')
        self.state.node_history.append('GetContextFromFollowup')
        try:
            query_result: QueryResult
            query_result, tokens = self.llm.generate_followup_query(
                question=self.state.question,
                conversation=self.state.conversation
            )
            self._add_usage(tokens)
            self.logger.debug(f'Generated followup query: {query_result.query}')
            
            result = self.raptor_retriever.retrieve_chunks(
                query_str=query_result.query,
                context_filters=self.state.context_filters
            )
            if not result:
                self.logger.debug('No chunks found from followup query')
                return False
                
            self.state.context = self._get_context_from_result(result)
            return True
        except Exception as e:
            self.logger.error(f'Error in followup query: {e}')
            return False

    def _expand_followup_question(self) -> bool:
        """Refine the follow-up question using LLM based on conversation history.
        
        Returns:
            bool: True if refinement was successful
        """
        self.logger.debug('Refining followup question')
        self.state.node_history.append('RefineFollowupQuestion')
        try:
            refined_question_result: RefinedQuestionResult
            refined_question_result, tokens = self.llm.refine_followup_question(
                question=self.state.question,
                conversation=self.state.conversation
            )
            self._add_usage(tokens)
            self.state.question = refined_question_result.refined_question
            self.logger.debug(f'Refined follow-up question: {refined_question_result.refined_question}')
            return True
        except Exception as e:
            self.logger.error(f'Error refining follow-up question: {e}')
            return False

    # Helper methods
    def _setup_logger(self, log_level: int, print_logs_to_console: bool) -> logging.Logger:
        """Configure and return a logger instance."""
        logger = logging.getLogger(__name__)
        logger.level = log_level
        if print_logs_to_console:
            logger.addHandler(logging.StreamHandler())
        return logger

    def _generate_search_query(self) -> bool:
        """Generate an optimized search query using LLM.
        
        Returns:
            bool: True if query generation was successful
        """
        self.logger.debug('Generating search query')
        try:
            query_result: QueryResult
            query_result, tokens = self.llm.generate_query(
                question=self.state.question,
                conversation=self.state.conversation
            )
            self.state.search_query = query_result.query
            self.state.needs_overview = query_result.needs_overview
            self._add_usage(tokens)
            return True
        except Exception as e:
            self.logger.error(f'Error generating search query: {e}')
            return False

    def _try_keyword_retrieval(self) -> bool:
        """Attempt to retrieve context using keyword-based search.
        
        Returns:
            bool: True if context was successfully retrieved using keywords
        """
        self.logger.debug('Attempting keyword retrieval')
        try:
            keywords_result: KeywordsResult
            keywords_result, tokens = self.llm.get_keywords_from_question(
                question=self.state.question,
                conversation=self.state.conversation
            )
            self._add_usage(tokens)
            self.logger.debug(f'Keywords for retrieval: {keywords_result.keywords}')
            return self._get_context_from_keywords(keywords=keywords_result.keywords)
        except Exception as e:
            self.logger.error(f'Error in keyword retrieval: {e}')
            return False