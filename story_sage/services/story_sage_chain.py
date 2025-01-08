# Import necessary libraries and modules
import logging
from typing import List, Literal
from chromadb import QueryResult
from ..models import StorySageConfig, StorySageState, StorySageContext
from . import StorySageLLM, StorySageRetriever
from ..utils import flatten_nested_list

# These are helpers for development and should both be False before committing
PRINT_STATE = False
LOG_STATE = False

def print_state(step_name: str, state: StorySageState):
    if PRINT_STATE:
        print(f'Step: {step_name}')
        print(f'State: {state}')
        print('🥦' * 20)
    if LOG_STATE:
        logging.debug(f'Step: {step_name}')
        logging.debug(f'State: {state}')

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
    """

    def __init__(self, config: StorySageConfig, state: StorySageState, 
                 log_level: int = logging.WARN) -> 'StorySageChain':
        self.config = config

        self.entities = config.entities
        self.series_list = config.series
        self.summary_retriever = StorySageRetriever(config.chroma_path, config.chroma_collection, config.n_chunks)
        self.full_retriever = StorySageRetriever(config.chroma_path, config.chroma_full_text_collection, round(config.n_chunks / 3))
        self.prompts = config.prompts
        self.llm = StorySageLLM(config, log_level=log_level)

        self.logger = logging.getLogger(__name__)
        self.logger.level = log_level

        self.state = state

    def invoke(self) -> StorySageState:
        """Execute the chain to generate a response to the question in the current state.
        
        Returns:
            StorySageState: Updated state containing the answer and context
        """
        # Get the initial context filters
        self._get_context_filters()

        # Get the initial context
        self._get_initial_context()

        # Try to get context from vector store
        while True:
            if self._identify_relevant_chunks():
                if self._get_context_by_ids():
                    break
                else:
                    # handle scenario where it made up IDs
                    pass # For now, let this go on to contents from summary chunks
            elif self._get_context_from_chunks('summary'):
                break
            elif self._get_context_from_chunks('full'):
                break

        if not self._generate():
            try:
                keywords = self.llm.get_keywords_from_question(question=self.state.question,
                                                            conversation=self.state.conversation)
                self.logger.debug(f'Keywords: {keywords}')
                self._get_context_from_keywords(keywords=keywords)
            except Exception as e:
                self.logger.error(f'Error getting keywords: {e}')
                self.state.answer = "I'm sorry, I couldn't find an answer to that question."
                return self.state


        self._generate()

        return self.state


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
        return


    def _get_initial_context(self) -> None:
        """Retrieve initial context using summary retriever."""
        self.state.node_history.append('GetInitialContext')
        self.initial_context = self.summary_retriever.first_pass_query(
            query_str=self.state.question,
            context_filters=self.state.context_filters
        )
        return
    

    def _identify_relevant_chunks(self) -> bool:
        """Use LLM to identify relevant chunks from initial context.
        
        Returns:
            bool: True if relevant chunks were identified, False otherwise
        """
        self.state.node_history.append('IdentifyRelevantChunks')
        try:
            relevant_chunks = self.llm.identify_relevant_chunks(
                question=self.state.question,
                context=self.initial_context,
                conversation=self.state.conversation
            )
        except Exception as e:
            self.logger.error(f'Error identifying relevant chunks: {e}')
            return False
        if len(relevant_chunks[0]) < 1:
            self.secondary_query = relevant_chunks[1]
            return False
        else:
            self.state.target_ids = relevant_chunks[0]
            return True

    def _get_context_by_ids(self) -> bool:
        """Retrieve context using specific chunk IDs.
        
        Returns:
            bool: True if context was successfully retrieved, False if no chunks found
        """
        self.state.node_history.append('GetContextByIDs')
        results = self.summary_retriever.get_by_ids(self.state.target_ids)
        
        if len(results['ids']) < 1:
            return False
        
        self.state.context = self._get_context_from_result(results)

        return True

    def _get_context_from_chunks(self, collection: Literal['summary', 'full']) -> bool:
        """Retrieve context from either summary or full text collection.
        
        Args:
            collection: Either 'summary' or 'full' to specify which collection to query
            
        Returns:
            bool: True if context was successfully retrieved, False if no chunks found
            
        Raises:
            ValueError: If collection is neither 'summary' nor 'full'
        """
        self.state.node_history.append('GetContextFromChunks')
        if collection == 'summary':
            retriever = self.summary_retriever
        elif collection == 'full':
            retriever = self.full_retriever
        else:
            raise ValueError('Collection must be "summary" or "full"')
        
        results = retriever.retrieve_chunks(
            query_str=self.state.question,
            context_filters=self.state.context_filters
        )

        if len(results['ids'][0]) < 1:
            return False
        
        self.state.context = self._get_context_from_result(results)
        
        return True

    def _get_context_from_keywords(self, keywords: List[str] = None) -> None:
        """Retrieve context by searching for specific keywords.
        
        Args:
            keywords: List of keywords to search for. If None, uses words from question
            
        Returns:
            bool: True if context was successfully retrieved, False if no chunks found
        """
        self.state.node_history.append('GetContextFromKeywords')
        if not keywords:
            keywords = self.state.question.split()
        # Get context from the full text retriever
        results = self.full_retriever.get_by_keyword(keywords=keywords,
                                                     context_filters=self.state.context_filters)

        if len(results['ids']) < 1:
            return False
        
        self.state.context = self._get_context_from_result(results)

        return True

    def _generate(self) -> None:
        """Generate response using LLM based on current context and question.
        
        Returns:
            bool: True if generation was successful
        """
        self.state.node_history.append('Generate')
        try:
            response = self.llm.generate_response(
                context=self.state.context,
                question=self.state.question,
                conversation=self.state.conversation
            )
            self.state.answer = response[0]
            return response[1]
        except Exception as e:
            self.logger.error(f'Error generating response: {e}')
            return False
    

    def _get_context_from_result(self, results: QueryResult) -> List[StorySageContext]:
        """Convert query results into StorySageContext objects.
        
        Args:
            results: Query results from ChromaDB
            
        Returns:
            List[StorySageContext]: List of context objects sorted by chunk ID
        """
        results['ids'] = flatten_nested_list(results['ids'])
        results['metadatas'] = flatten_nested_list(results['metadatas'])
        
        # Add the chunk ID to the metadata so we can sort by it
        # Chunk ID is in the form <series metadata name>_<book number>_<chapter number>_<chunk index>
        #   e.g. "sherlock_holmes_1_1_0"
        for id, m in zip(results['ids'], results['metadatas']):
            m['id'] = id
        
        results = sorted(
            results['metadatas'],
            key=lambda x: (x['id'])
        )

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