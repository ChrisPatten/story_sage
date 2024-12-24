# Import necessary libraries and modules
import logging
from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from .story_sage_state import StorySageState
from .story_sage_retriever import StorySageRetriever
from .story_sage_stepback import StorySageStepback
import httpx
from typing import Optional
import spacy

class StorySageChain(StateGraph):
    """Defines the chain of operations for the Story Sage system."""

    def __init__(self, api_key: str, entities: dict, retriever: StorySageRetriever, logger: Optional[logging.Logger] = None):
        """
        Initialize the StorySageChain instance.

        Args:
            api_key (str): The API key for the language model.
            entities (dict): Dictionary containing character information.
            retriever (StorySageRetriever): The retriever instance for fetching context.
            logger (Optional[logging.Logger]): Logger instance for logging.
        """
        # Store entities and retriever for later use
        self.entities = entities
        self.retriever = retriever

        # Load the spaCy NLP model
        self.nlp = spacy.load("en_core_web_sm")

        # Initialize the Stepback module to optimize queries
        self.stepback = StorySageStepback(api_key=api_key)
        # Set up the OpenAI language model with the provided API key
        self.llm = ChatOpenAI(api_key=api_key, model='gpt-4o-mini', http_client=httpx.Client(verify=False))
        # Define the prompt template for generating responses
        self.prompt = PromptTemplate(
            input_variables=['question', 'context', 'book_number', 'chapter_number'],
            template="""
                HUMAN

                You are an assistant to help a reader keep track of people, places, and plot points in books.
                The attached pieces of retrieved context are excerpts from the books related to the reader's question. Use them to generate your response.

                Guidelines for the response:
                * If you don't know the answer or aren't sure, just say that you don't know. 
                * Don't provide any irrelevant information.
                * Use bullet points to provide excerpts from the context that support your answer. Reference the book and chapter whenever you include an excerpt.
                * If there is no context, you can say that you don't have enough information to answer the question.

                The reader is currently in Book {book_number}, Chapter {chapter_number}. Don't limit your responses to just this book. Answer the reader's question
                taking into account all the context provided.

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
        graph_builder.add_node("RouterFunction", self.router_function)
        graph_builder.add_node("GetCharacters", self.get_characters)
        graph_builder.add_node("GetContext", self.get_context)
        graph_builder.add_node("Generate", self.generate)
        # Define the sequence of operations
        graph_builder.add_edge(START, "RouterFunction")
        graph_builder.add_edge("RouterFunction", "GetCharacters")
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
        series_info = None
        if state.get('series_id'):
            # Convert the question to lowercase for case-insensitive search
            question_text_search = state['question'].lower()
            series_id = int(state.get('series_id'))
            # Retrieve series information based on series ID
            for series_id_str, series in self.entities['series'].items():
                if series['series_id'] == series_id:
                    series_info = series['series_entities']
                    self.logger.debug(f'Series info found.')
                    break
           
            if not series_info:
                self.logger.error(f'Series info not found in entities!')
                self.logger.debug(f"Entities dict: {self.entities}")
                raise ValueError('Series info not found in entities.')

            all_entities_by_name = {**series_info['people_by_name'], **series_info['entity_by_name']}
            all_entities_by_id = {**series_info['people_by_id'], **series_info['entity_by_id']}
            # Check if any entity names are mentioned in the question
            for name, id in all_entities_by_name.items():
                if name in question_text_search:
                    entities_in_question.add(id)

        # Log the entities found
        entities_strs = ' | '.join([f"{id}: {all_entities_by_id[id]}" for id in entities_in_question])
        self.logger.debug(f'Entities: {entities_strs}')
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
        self.logger.debug(f"Context filters: {context_filters}")
        # Optimize the query or fall back to the original question
        optimized_query = self.stepback.optimize_query(state['question'])
        if not optimized_query:
            self.logger.debug("Failed to optimize query. Using original question.")
            optimized_query = state['question']
        # Retrieve chunks of context based on the query and filters
        self.logger.debug(f"order by: {state['order_by']}")
        retrieved_docs = self.retriever.retrieve_chunks(
            query_str=optimized_query,
            context_filters=context_filters,
            order_direction=state['order_by']
        )

        if not retrieved_docs:
            self.logger.debug("No context retrieved. Rerun without character filters.")
            context_filters['entities'] = []
            retrieved_docs = self.retriever.retrieve_chunks(
                query_str=optimized_query,
                context_filters=context_filters
            )

        # Format the retrieved context for the prompt
        context = [
            f"Book {book_number}, Chapter {chapter_number}: {doc}"
            for book_number, chapter_number, doc in retrieved_docs
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
            {'question': state['question'], 'context': docs_content, 'book_number': state['book_number'], 'chapter_number': state['chapter_number']}
        )
        self.logger.debug(f"LLM prompt: {messages}")
        # Get the response from the language model
        response = self.llm.invoke(messages)
        # Return the generated answer
        return {'answer': response.content}

    def router_function(self, state: StorySageState) -> StorySageState:
        """
        Router function to determine if the question is about the past or the present
        and set the order_by property accordingly.

        Args:
            state (StorySageState): The current state of the system.

        Returns:
            StorySageState: The updated state with the order_by property set.
        """
        question = state['question'].lower()

        # Use spaCy NLP model to analyze the question
        doc = self.nlp(question)

        # Default to 'most_recent' unless past tense is detected
        tense = 'most_recent'

        # Check for past tense verbs in the question
        for token in doc:
            if token.tag_ in ['VBD', 'VBN']:
                tense = 'earliest'
                break

        # Set the order_by property based on the detected tense
        state['order_by'] = tense
        return state
