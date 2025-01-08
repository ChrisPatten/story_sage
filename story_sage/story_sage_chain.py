# Import necessary libraries and modules
import logging
from typing import Optional, List, Literal
from chromadb import QueryResult
from .types import StorySageConfig, StorySageState, StorySageContext, StorySageLLM, StorySageRetriever
from .utils import flatten_nested_list

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

class StorySageChain:

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
            keywords = self.llm.get_keywords_from_question(question=self.state.question,
                                                           conversation=self.state.conversation)
            self.logger.debug(f'Keywords: {keywords}')
            self._get_context_from_keywords(keywords=keywords)


        self._generate()

        return self.state


    # Internal methods for steps in the chain
    def _get_context_filters(self) -> None:
        self.state.node_history.append('GetContextFilters')
        self.state.context_filters = {
            'entities': self.state.entities,
            'series_id': self.state.series_id,
            'book_number': self.state.book_number,
            'chapter_number': self.state.chapter_number
        }
        return


    def _get_initial_context(self) -> None:
        self.state.node_history.append('GetInitialContext')
        self.initial_context = self.summary_retriever.first_pass_query(
            query_str=self.state.question,
            context_filters=self.state.context_filters
        )
        return
    

    def _identify_relevant_chunks(self) -> bool:
        self.state.node_history.append('IdentifyRelevantChunks')
        relevant_chunks = self.llm.identify_relevant_chunks(
            question=self.state.question,
            context=self.initial_context,
            conversation=self.state.conversation
        )
        if len(relevant_chunks[0]) < 1:
            self.secondary_query = relevant_chunks[1]
            return False
        else:
            self.state.target_ids = relevant_chunks[0]
            return True

    def _get_context_by_ids(self) -> bool:
        self.state.node_history.append('GetContextByIDs')
        results = self.summary_retriever.get_by_ids(self.state.target_ids)
        
        if len(results['ids']) < 1:
            return False
        
        self.state.context = self._get_context_from_result(results)

        return True

    def _get_context_from_chunks(self, collection: Literal['summary', 'full']) -> bool:
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
        self.state.node_history.append('Generate')
        response = self.llm.generate_response(
            context=self.state.context,
            question=self.state.question,
            conversation=self.state.conversation
        )
        self.state.answer = response[0]
        return response[1]
    

    def _get_context_from_result(self, results: QueryResult) -> StorySageContext:
        
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
            StorySageContext(
                data = {
                    'chunk_id': m['id'], 
                    'book_number': m['book_number'], 
                    'chapter_number': m['chapter_number'], 
                    'chunk': m['full_chunk']
                } 
            ) for m in results
        ]