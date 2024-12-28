# Import necessary libraries and modules
import logging
from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from .data_classes.story_sage_state import StorySageState
from .vector_store import StorySageRetriever
from .story_sage_stepback import StorySageStepback
from .story_sage_entity import StorySageEntityCollection, StorySageEntity
from .data_classes.story_sage_series import StorySageSeries
import httpx
from typing import Optional, List


class StorySageChain(StateGraph):
    """Defines the chain of operations for the Story Sage system."""

    def __init__(self, api_key: str, entities: dict[str, StorySageEntityCollection], 
                 series_list: List[StorySageSeries], retriever: StorySageRetriever, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the StorySageChain instance.

        Args:
            api_key (str): The API key for the language model.
            entities (dict[str, StorySageEntityCollection]): Dictionary containing character information.
            retriever (StorySageRetriever): The retriever instance for fetching context.
            logger (Optional[logging.Logger]): Logger instance for logging.
        """

        # Validate that entities is a dictionary with keys of type str and values of type StorySageEntityCollection
        if not isinstance(entities, dict) or not all(isinstance(k, str) and isinstance(v, StorySageEntityCollection) for k, v in entities.items()):
            raise ValueError("entities must be a dictionary with keys of type str and values of type StorySageEntityCollection")
        # Validate that series_list is a list of StorySageSeries objects
        if not isinstance(series_list, list) or not all(isinstance(series, StorySageSeries) for series in series_list):
            raise ValueError("series_list must be a list of StorySageSeries objects")
        
        
        # Store entities and retriever for later use
        self.entities = entities
        self.series_list = series_list
        self.retriever = retriever
        # Initialize the Stepback module to optimize queries
        self.stepback = StorySageStepback(api_key=api_key)
        # Set up the OpenAI language model with the provided API key
        self.llm = ChatOpenAI(api_key=api_key, model='gpt-4o-mini', http_client=httpx.Client(verify=False))
        # Define the prompt template for generating responses
        self.prompt = PromptTemplate(
            input_variables=['question', 'context'],
            template="""
                HUMAN

                You are an assistant to help a reader keep track of people, places, and plot points in books.
                The attached pieces of retrieved context are excerpts from the books related to the reader's question. Use them to generate your response.

                Guidelines for the response:
                * If you don't know the answer or aren't sure, just say that you don't know. 
                * Don't provide any irrelevant information. Most importantly: DO NOT PROVIDE INFORMATION FROM OUTSIDE THE CONTEXT.
                * Use bullet points to provide excerpts from the context that support your answer. Reference the book and chapter whenever you include an excerpt.
                * If there is no context, you can say that you don't have enough information to answer the question.

                Question: {question} 
                Context: {context} 
                Answer:
            """
        )
        # Initialize the logger
        if logger:
            self.logger = logger
            self.logger.debug('Logger initialized from parent.')
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.debug('Logger initialized locally.')

        # Build the state graph for the chain's workflow
        graph_builder = StateGraph(StorySageState)
        graph_builder.add_node("GetCharacters", self.get_characters)
        graph_builder.add_node("GetContext", self.get_context)
        graph_builder.add_node("Generate", self.generate)
        # Define the sequence of operations
        graph_builder.add_edge(START, "GetCharacters")
        graph_builder.add_edge("GetCharacters", "GetContext")
        graph_builder.add_edge("GetContext", "Generate")
        graph_builder.add_edge("Generate", END)
        # Compile the graph
        self.graph = graph_builder.compile()
  
    def get_characters(self, state: StorySageState) -> dict:
        """
        Extract entities mentioned in the user's question, filtered by series.

        Args:
            state (StorySageState): The current state of the system.

        Returns:
            dict: A dictionary with lists of entities found in the question.
        """
        self.logger.debug("Extracting characters from question.")
        # Initialize sets to collect entities
        entities_in_question = set()
        # Preprocess question for entity search
        question_text_search = ''.join(c for c in str.lower(state['question']) if c.isalpha() or c.isspace())
        if state.get('series_id'):
            self.logger.debug("Series ID found in state.")
            series_info = next((series for series in self.series_list if series.series_id == state['series_id']), None)
            # Convert the question to lowercase for case-insensitive search
            question_text_search = state['question'].lower()
            series_entities = self.entities.get(series_info.series_metadata_name)
            if series_entities:
                entities = series_entities.get_all_entities()
                # Check if any entity names are mentioned in the question
                for entity in entities:
                    if entity.entity_name in question_text_search:
                        entities_in_question.add(entity.entity_group_id)
                self.logger.debug(f'Found entities for series: {series_info.series_name}')
        # Return the entities as lists
        return {
            'entities': list(entities_in_question),
        }
  
    def get_context(self, state: StorySageState) -> dict:
        """
        Retrieve relevant context based on the user's question and extracted entities.

        Args:
            state (StorySageState): The current state of the system.

        Returns:
            dict: A dictionary containing the retrieved context.
        """
        self.logger.debug("Retrieving context based on the question and entities.")
        # Set up filters for context retrieval
        context_filters = {
            'entities': state['entities'],
            'series_id': state['series_id'],
            'book_number': state['book_number'],
            'chapter_number': state['chapter_number']
        }
        # Retrieve chunks of context based on the query and filters
        retrieved_docs = self.retriever.retrieve_chunks(
            query_str=state['question'],
            context_filters=context_filters
        )

        if not retrieved_docs:
            self.logger.debug("No context retrieved. Rerun without character filters.")
            context_filters['entities'] = []
            retrieved_docs = self.retriever.retrieve_chunks(
                query_str=state['question'],
                context_filters=context_filters
            )

        # Format the retrieved context for the prompt
        context = [
            f"Book {meta['book_number']}, Chapter {meta['chapter_number']}: {doc}"
            for meta, doc in zip(retrieved_docs['metadatas'][0], retrieved_docs['documents'][0])
        ]
        self.logger.debug(f"Context retrieved: {context}")
        # Return the context
        return {'context': context}
  
    def generate(self, state: StorySageState) -> dict:
        """
        Generate an answer using the language model based on the question and context.

        Args:
            state (StorySageState): The current state of the system.

        Returns:
            dict: A dictionary containing the generated answer.
        """
        # Combine context excerpts into a single string
        docs_content = '\n\n'.join(state['context'])
        # Invoke the prompt with the question and context
        messages = self.prompt.invoke(
            {'question': state['question'], 'context': docs_content}
        )
        self.logger.debug(f"LLM prompt: {messages}")
        # Get the response from the language model
        response = self.llm.invoke(messages)
        # Return the generated answer
        return {'answer': response.content}