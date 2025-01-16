# Import necessary libraries and modules
import logging
import regex as re
from typing import List

from chromadb import QueryResult
from ..models import StorySageConfig, StorySageState, StorySageContext
from . import StorySageLLM, StorySageRetriever
from .story_sage_llm import _UsageType
from ..utils import flatten_nested_list

# Constants
VALID_CHUNK_ID_PATTERN = re.compile(r'series_\d+\|book_\d+\|chapter_\d+\|level_\d\+|chunk_\d+')

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

    def invoke(self) -> StorySageState:
        """Execute the chain to generate a response to the question in the current state.
        
        Returns:
            StorySageState: Updated state containing the answer and context
        """
        self.logger.debug(f'Invoking StorySageChain with question: {self.state.question}')
        self.state.node_history = []

        # Get the initial context filters
        self._get_context_filters()
        self.logger.debug(f'Context filters: {self.state.context_filters}')

        if self._is_followup_question():
            self.logger.debug('Question is a followup')
            if self._refine_followup_question():
                if self._get_context_from_followup():
                    if self._generate():
                        return self.state

        # Generate an optimized query for vector search
        if not self._generate_search_query():
            self.state.search_query = self.state.question
        self.logger.debug(f'Search query: {self.state.search_query}')

       #self.state.search_query = self.state.question

        # Get the initial context based on the generated search query
        self._get_summary_context()
        #self.logger.debug(f'Summary chunks: {self.state.summary_chunks["ids"]}')
        
        if self._get_relevant_chunks():
            self.logger.debug(f'Target IDs: {self.state.target_ids}')
            self._get_context_by_ids()
        elif self._get_context_from_chunks(exclude_summaries=False):
            self.logger.debug(f'Got context from full text')

        self.state.summary_chunks = None

        if self._generate():
            if self.needs_clarification:
                self.logger.debug('Generated clarifying question')
                return self.state
            #self.logger.debug(f'Generated response: {self.state.answer}')
            return self.state

        self.logger.debug('Generating response from semantic search failed. Trying keyword search.')
        if self._try_keyword_retrieval():
            return self.state

        if not self.state.answer:
            self.state.answer = "I'm sorry, I couldn't find an answer to that question."
        
        self.logger.debug(f'Final answer: {self.state.answer}')
        return self.state or StorySageState(answer="I'm sorry, I couldn't find an answer to that question.")

    # Internal methods for steps in the chain
    def _get_context_filters(self) -> None:
        """Initialize context filters based on current state for document retrieval."""
        self.state.node_history.append('GetContextFilters')
        self.state.context_filters = {
            'entities': self.state.entities,
            'series_id': self.state.series_id,
            'book_number': self.state.book_number,
            'chapter_number': self.state.chapter_number
        }
    
    def _get_summary_context(self) -> None:
        """Retrieve context from the summary collection."""
        self.state.node_history.append('GetSummaryContext')
        context_filters = self.state.context_filters.copy()
        context_filters['top_level_only'] = False
        try:
            self.state.summary_chunks = self.raptor_retriever.retrieve_chunks(
                query_str=self.state.search_query,
                context_filters=context_filters,
                n_results=15
            )
        except Exception as e:
            self.logger.error(f'Error retrieving summary context: {e}')
            self.state.summary_chunks = []

    def _identify_relevant_chunks(self) -> bool:
        """Use LLM to identify relevant chunks from initial context.
        
        Returns:
            bool: True if relevant chunks were identified, False otherwise
        """
        self.state.node_history.append('IdentifyRelevantChunks')
        try:
            relevant_chunks, secondary_query, tokens = self.llm.identify_relevant_chunks(
                question=self.state.question,
                context=self.state.summary_chunks,
                conversation=self.state.conversation
            )
            self._add_usage(tokens)
        except Exception as e:
            self.logger.error(f'Error identifying relevant chunks: {e}')
            return False
        if len(relevant_chunks) < 1:
            self.secondary_query = secondary_query
            return False
        elif not all(re.match(VALID_CHUNK_ID_PATTERN, chunk_id) for chunk_id in relevant_chunks):
            self.secondary_query = secondary_query
            return False
        else:
            self.state.target_ids = relevant_chunks
            return True

    def _get_context_by_ids(self) -> bool:
        """Retrieve context using specific chunk IDs.
        
        Returns:
            bool: True if context was successfully retrieved, False if no chunks found
        """
        self.state.node_history.append('GetContextByIDs')
        try:
            results = self.raptor_retriever.retrieve_from_hierarchy(self.state.target_ids)
        except Exception as e:
            self.logger.error(f'Error retrieving context by IDs: {e}')
            return False
        
        if len(results['ids']) < 1:
            return False
        
        self.state.context = self._get_context_from_result(results)
        return True

    def _get_context_from_chunks(self, exclude_summaries: bool = False) -> bool:
        """Retrieve context from either summary or full text collection.
        
        Args:
            exclude_summaries: Whether to exclude summary chunks from the search
            
        Returns:
            bool: True if context was successfully retrieved, False if no chunks found
        """
        self.state.node_history.append('GetContextFromChunks')
        context_filters = self.state.context_filters.copy()
        if exclude_summaries:
            context_filters['exclude_summaries'] = True

        try:
            results = self.raptor_retriever.retrieve_chunks(
                query_str=self.state.search_query,
                context_filters=context_filters
            )
        except Exception as e:
            self.logger.error(f'Error retrieving context from chunks: {e}')
            return False

        if len(results['ids'][0]) < 1:
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
        self.state.node_history.append('GetContextFromKeywords')
        if not keywords:
            keywords = self.state.question.split()
        try:
            results = self.raptor_retriever.get_by_keyword(
                keywords=keywords,
                context_filters=self.state.context_filters
            )
        except Exception as e:
            self.logger.error(f'Error retrieving context from keywords: {e}')
            return False

        if len(results['ids']) < 1:
            return False
        
        self.state.context = self._get_context_from_result(results)
        return True
    
    def _get_relevant_chunks(self, threshold: int = 6) -> bool:
        """Retrieve relevant chunks based on the current context.
        
        Args:
            threshold: Number of chunks to retrieve
            
        Returns:
            bool: True if relevant chunks were found, False otherwise
        """
        self.state.node_history.append('GetRelevantChunks')
        try:
            relevant_chunks: dict[str, int] = None
            relevant_chunks, tokens = self.llm.evaluate_chunks(
                question=self.state.question,
                context=self.state.context,
                conversation=self.state.conversation
            )
            # Remove any chunks from context that aren't in in_scope_chunks
            self._add_usage(tokens)
        except Exception as e:
            self.logger.error(f'Error getting relevant chunks: {e}')
            return False
        
        if not relevant_chunks:
            return False
        
        try:
            in_scope_chunks: list[str] = []
            for chunk_key, score in relevant_chunks.items():
                if score >= threshold and re.match(VALID_CHUNK_ID_PATTERN, chunk_key):
                    in_scope_chunks.append(chunk_key)
        except:
            self.logger.error('Error filtering relevant chunks. Keys:', relevant_chunks)
            return False
    
        self.state.target_ids = in_scope_chunks
        return True

    def _generate(self) -> bool:
        """Generate response using LLM based on current context and question.
        
        Returns:
            bool: True if generation was successful
        """
        self.state.node_history.append('Generate')
        try:
            response, has_answer, follow_up, tokens = self.llm.generate_response(
                context=self.state.context,
                question=self.state.question,
                conversation=self.state.conversation
            )
            self._add_usage(tokens)
            if not has_answer and follow_up:
                self.needs_clarification = True
                self.state.answer = response + '\n\n\n' + follow_up
                return True
            
            self.needs_clarification = False
            self.state.answer = (response + '\n\n\n' + follow_up) if follow_up else response
            return has_answer
        except Exception as e:
            self.logger.error(f'Error generating response: {e}')
            return False

    def _get_context_from_result(self, results: QueryResult) -> List[StorySageContext]:
        """Convert query results into StorySageContext objects.
        
        Args:
            results: Query results from ChromaDB
            
        Returns:
            List<StorySageContext]: List of context objects sorted by chunk ID
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
    
    def _add_usage(self, tokens: _UsageType) -> None:
        """Add token usage to the state object.
        
        Args:
            tokens: Tuple of token counts for the current operation
        """
        self.state.tokens_used = (self.state.tokens_used[0] + tokens[0], self.state.tokens_used[1] + tokens[1])

    def _is_followup_question(self) -> bool:
        """Check if current question is a followup based on conversation history."""
        if self.state.conversation:
            return len(self.state.conversation.get_history()) > 0
        else:
            return False

    def _get_context_from_followup(self) -> bool:
        """Get context using followup query based on conversation history.
        
        Returns:
            bool: True if context was found, False otherwise
        """
        self.state.node_history.append('GetContextFromFollowup')
        try:
            query, tokens = self.llm.generate_followup_query(
                question=self.state.question,
                conversation=self.state.conversation
            )
            self._add_usage(tokens)
            self.logger.debug(f'Generated followup query: {query}')
            
            result: QueryResult = self.raptor_retriever.retrieve_chunks(
                query_str=query,
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

    def _refine_followup_question(self) -> bool:
        """Refine the follow-up question using LLM based on conversation history.
        
        Returns:
            bool: True if refinement was successful
        """
        self.state.node_history.append('RefineFollowupQuestion')
        try:
            refined_question, tokens = self.llm.refine_followup_question(
                question=self.state.question,
                conversation=self.state.conversation
            )
            self._add_usage(tokens)
            self.state.question = refined_question
            self.logger.debug(f'Refined follow-up question: {refined_question}')
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
        try:
            search_query, tokens = self.llm.generate_query(
                question=self.state.question,
                conversation=self.state.conversation
            )
            self.state.search_query = search_query
            self._add_usage(tokens)
            return True
        except Exception as e:
            self.logger.error(f'Error generating search query: {e}')
            return False

    def _process_summary_chunks(self) -> bool:
        """Process summary chunks and retrieve detailed context if relevant.
        
        Returns:
            bool: True if processing was successful
        """
        if len(self.state.summary_chunks) > 0:
            if not self._identify_relevant_chunks():
                if self.secondary_query:
                    return self._handle_secondary_query()
                return False
            return self._get_context_by_ids() and self._generate()
        return False

    def _handle_secondary_query(self) -> bool:
        """Handle secondary query to get more context.
        
        Returns:
            bool: True if context was successfully retrieved
        """
        self.logger.debug(f'Secondary query: {self.secondary_query}')
        if self.state.answer:
            response = self.state.answer + '\n\n' + self.secondary_query
            self.state.conversation.add_turn(
                question=self.state.question, 
                detected_entities=self.state.entities, 
                context=self.state.context, 
                response=response
            )
            self.state.search_query = self.secondary_query
            return self._get_context_from_chunks(exclude_summaries=False)
        return False

    def _try_keyword_retrieval(self) -> bool:
        """Attempt to retrieve context using keyword-based search.
        
        Returns:
            bool: True if retrieval was successful
        """
        try:
            keywords, tokens = self.llm.get_keywords_from_question(
                question=self.state.question,
                conversation=self.state.conversation
            )
            self._add_usage(tokens)
            return self._get_context_from_keywords(keywords=keywords)
        except Exception as e:
            self.logger.error(f'Error in keyword retrieval: {e}')
            return False