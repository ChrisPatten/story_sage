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
from typing import Optional, List, Literal
import spacy
from pydantic import BaseModel
from openai import OpenAI

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

class StorySageChain(StateGraph):
    """Defines a chain of operations for the Story Sage system.

    This class manages a workflow that extracts characters, retrieves context,
    and generates user-facing answers based on retrieved text. The workflow is
    represented by a compiled StateGraph.

    Attributes:
        entities (dict[str, StorySageEntityCollection]): A mapping of series metadata names to entity collections.
        series_list (List[StorySageSeries]): A list of series information.
        retriever (StorySageRetriever): Used to fetch relevant document context chunks.
        stepback (StorySageStepback): Provides query optimization functionality.
        llm (ChatOpenAI): Language model interface.
        prompt (PromptTemplate): Defines the text prompt structure for responses.
        logger (logging.Logger): Logger for debug and info messages.

    Example:
        >>> # Initialize the chain with necessary components
        >>> chain_instance = StorySageChain(
        ...     api_key="YOUR_API_KEY",
        ...     entities={"my_series": StorySageEntityCollection([])},
        ...     series_list=[StorySageSeries(series_id=1, series_name="My Series", series_metadata_name="my_series")],
        ...     retriever=StorySageRetriever("path/to/db", "collection", 5),
        ...     logger=logging.getLogger("story_sage_chain")
        ... )
        >>> # Invoke some state-handling methods via:
        >>> # chain_instance.graph.invoke(StorySageState(...))

    Example Results:
        The chain will generate structured results with an answer and context 
        details based on the user's question and the available series data.
    """

    def __init__(self, api_key: str, entities: dict[str, StorySageEntityCollection], 
                 series_list: List[StorySageSeries], retriever: StorySageRetriever, 
                 logger: Optional[logging.Logger] = None):
        """Initializes a StorySageChain instance.

        Args:
            api_key (str): The API key for the language model.
            entities (dict[str, StorySageEntityCollection]): A dictionary of entity collections keyed by series metadata name.
            series_list (List[StorySageSeries]): A list containing information about different series.
            retriever (StorySageRetriever): The retriever used to query relevant text chunks.
            logger (Optional[logging.Logger]): An optional logger instance for debug and info logs.

        Returns:
            StorySageChain: An initialized StorySageChain object with a compiled processing graph.

        Example:
            >>> chain_instance = StorySageChain(
            ...     api_key="YOUR_API_KEY",
            ...     entities={"series_key": StorySageEntityCollection([])},
            ...     series_list=[StorySageSeries(...)] ,
            ...     retriever=StorySageRetriever(...),
            ...     logger=logging.getLogger(__name__)
            ... )

        Example Results:
            The returned StorySageChain object includes a compiled StateGraph
            that can be used to process user questions and retrieve contextual answers.
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

        # Load the spaCy NLP model
        self.nlp = spacy.load("en_core_web_sm")

        # Initialize the Stepback module to optimize queries
        self.stepback = StorySageStepback(api_key=api_key)
        # Set up the OpenAI language model with the provided API key
        self.llm = ChatOpenAI(api_key=api_key, model='gpt-4o-mini', http_client=httpx.Client(verify=False))
        # Define the prompt template for generating responses
        self.prompt = PromptTemplate(
            input_variables=['question', 'context', 'book_number', 'chapter_number', 'conversation'],
            template="""
                HUMAN

                You are an assistant to help a reader keep track of people, places, and plot points in books.
                The attached pieces of retrieved context are excerpts from the books related to the reader's question. Use them to generate your response.

                Guidelines for the response:
                * If you don't know the answer or aren't sure, just say that you don't know. 
                * Don't provide any irrelevant information. Most importantly: DO NOT PROVIDE INFORMATION FROM OUTSIDE THE CONTEXT.
                * Use bullet points to provide excerpts from the context that support your answer. Reference the book and chapter whenever you include an excerpt.
                * If there is no context, you can say that you don't have enough information to answer the question.

                The reader is currently in Book {book_number}, Chapter {chapter_number}. Don't limit your responses to just this book. Answer the reader's question
                taking into account all the context provided.

                ---------------------
                Question: {question} 
                ---------------------
                Previous Conversation: {conversation}
                ---------------------
                Context: {context} 
                ---------------------
                Answer:
            """
        )

        # Set up the OpenAI Client to be able to use other APIs besides LangChain
        self.client = OpenAI(api_key=api_key, http_client=httpx.Client(verify=False))


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
        graph_builder.add_node("GetContextFilters", self.get_context_filters)
        graph_builder.add_node("GetInitialContext", self.get_initial_context)
        graph_builder.add_node("IdentifyRelevantChunks", self.identify_relevant_chunks)
        graph_builder.add_node("GetContextByIDs", self.get_context_by_ids)
        graph_builder.add_node("GetContext", self.get_context)
        graph_builder.add_node("Generate", self.generate)
        # Define the sequence of operations
        graph_builder.add_edge(START, "RouterFunction")
        graph_builder.add_edge("RouterFunction", "GetCharacters")
        graph_builder.add_edge("GetCharacters", "GetContextFilters")
        graph_builder.add_edge("GetContextFilters", "GetInitialContext")
        graph_builder.add_edge("GetInitialContext", "IdentifyRelevantChunks")
        graph_builder.add_conditional_edges(
            "IdentifyRelevantChunks", 
            self.check_for_target_ids,
            {
                'by_id': "GetContextByIDs",
                'full_search': "GetContext"
            }
        )
        graph_builder.add_edge("GetContextByIDs", "Generate")
        graph_builder.add_edge("GetContext", "Generate")
        graph_builder.add_edge("Generate", END)
        # Compile the graph
        self.graph = graph_builder.compile()
  
    def get_characters(self, state: StorySageState) -> dict:
        """Identifies entities mentioned in the user's question, filtered by series.

        Args:
            state (StorySageState): The current state of the system, including user query details and context parameters.

        Returns:
            dict: A dictionary containing a list of unique entity group IDs found in the question text.

        Example:
            >>> state = StorySageState(question="What does Rand do?", series_id=1)
            >>> result = chain_instance.get_characters(state)
            >>> print(result)
            {'entities': ['<group_id_1>', '<group_id_2>']}

        Example Results:
            If certain entity names appear in the user's question, they will be
            mapped to their respective entity_group_id values.
        """
        print_state('get_characters', state)
        self.logger.debug("Extracting characters from question.")
        # Initialize sets to collect entities
        entities_in_question = set()
        # Preprocess question for entity search
        question_text_search = ''.join(c for c in str.lower(state['question']) if c.isalpha() or c.isspace())
        series_info = None
        if state.get('series_id'):
            self.logger.debug("Series ID found in state.")
            series_info = next((series for series in self.series_list if series.series_id == int(state['series_id'])), None)
            if not series_info:
                print(self.series_list)
                raise ValueError(f"Series ID {state['series_id']} not found in series list.")
            # Convert the question to lowercase for case-insensitive search
            question_text_search = state['question'].lower()
            series_entities = self.entities.get(series_info.series_metadata_name)
            if series_entities:
                entities = series_entities.get_all_entities()
                # Check if any entity names are mentioned in the question
                for entity in entities:
                    if entity.entity_name in question_text_search:
                        entities_in_question.add(entity.entity_group_id)
                self.logger.debug(f'Found entities: {entities_in_question}')
        # Return the entities as lists
        new_node_history = state['node_history'].append('GetCharacters') if state['node_history'] else ['GetCharacters']
        return {
            'entities': list(entities_in_question),
            'node_history': new_node_history
        }
    
    def get_initial_context(self, state: StorySageState) -> dict:
        """Retrieves the initial context based on the user's question and entities.

        This method queries the document store based on the question and entities
        to retrieve relevant text chunks for further processing.

        Args:
            state (StorySageState): The current state containing the user's question and entity IDs.

        Returns:
            dict: A dictionary containing the 'initial_context' field with a list of relevant text excerpts.

        Example:
            >>> state = StorySageState(question="Where does John meet Sarah?", entities=["<group_id_1>", "<group_id_2>"])
            >>> context_data = chain_instance.get_initial_context(state)
            >>> print(context_data['initial_context'])
            ['Book 2, Chapter 5: John travels to the docks...', ...]

        Example Results:
            Returns selected text chunks that match the provided entities and can
            be used to generate a final answer.
        """
        print_state('get_initial_context', state)
        self.logger.debug("Retrieving initial context based on the question and entities.")
        # Query the document store for relevant text chunks
        doc_dict = self.retriever.first_pass_query(
            query_str=state['question'],
            context_filters=state['context_filters']
        )

        new_node_history = state['node_history'].append('GetInitialContext') if state['node_history'] else ['GetInitialContext']
        return { 
            'initial_context': doc_dict,
            'node_history': new_node_history
        }
  
    def identify_relevant_chunks(self, state: StorySageState) -> dict:
        """
        A placeholder for the chunk relevance logic, returning a class
        with a 'chunk_ids' attribute for filtering.
        """
        print_state('identify_relevant_chunks', state)
        if 'initial_context' not in state or len(state['initial_context']) < 1:
            return { 'target_ids': [], 'secondary_query': None }
        class RelevantChunks(BaseModel):
            chunk_ids: list[str]
            secondary_query: str
        
        summaries = "\n".join([f"- {id}: {doc}" for id, doc in state['initial_context'].items()])
        chat_completion = self.client.beta.chat.completions.parse(
            messages=[
                {
                    "role": "system",
                    "content": f"""
                        You will receive a list of summaries of chunked passages from bookes along with their IDs.
                        Based on the provided summaries, please identify the IDs of the chunks that are most relevant to the input text.
                        Make sure to return at least some IDs, even if you are not sure about their relevance.
                        The summaries are in the format:

                        - <chunk_id>: <summary>

                        If the answer to the input text is not likely to be present based on the summaries, 
                            please provide a secondary query to send to the vector store that would help in identifying more relevant chunks.
                        For example, if the input text is about someone with a relationship to someone else,
                            write a query that would help in identifying who the other people in the relationship are.

                        Input text: {state['question']}
                    """
                },
                {
                    "role": "user",
                    "content": summaries
                }
            ],
            model="gpt-4o-mini",
            response_format=RelevantChunks
        )
        result = chat_completion.choices[0].message.parsed
        new_node_history = state['node_history'].append('IdentifyRelevantChunks') if state['node_history'] else ['IdentifyRelevantChunks']
        return { 
            'target_ids': result.chunk_ids, 
            'secondary_query': result.secondary_query,
            'tokens_used': state['tokens_used'] + chat_completion.usage.total_tokens,
            'node_history': new_node_history
        }

    def get_context_filters(self, state: StorySageState) -> dict:
        # Set up filters for context retrieval
        print_state(f"get_context_filters", state)
        new_node_history = state['node_history'].append('GetContextFilters') if state['node_history'] else ['GetContextFilters']
        return { 
            'context_filters' : {
                'entities': state['entities'],
                'series_id': state['series_id'],
                'book_number': state['book_number'],
                'chapter_number': state['chapter_number']
            },
            'node_history': new_node_history
        }
        
    def get_context_by_ids(self, state: StorySageState) -> dict:
        print_state(f"get_context_by_ids", state)

        results = self.retriever.get_by_ids(state['target_ids'])
        
        # Sort the results by book_number and chapter_number
        sorted_results = sorted(
            results['metadatas'],
            key=lambda x: (x['book_number'], x['chapter_number'])
        )

        context = [ 
            f"Book {m['book_number']}, Chapter {m['chapter_number']}: {m['full_chunk']}" for m in sorted_results
        ]

        new_node_history = state['node_history'].append('GetContextByIds') if state['node_history'] else ['GetContextByIds']

        return {
            'context': context,
            'node_history': new_node_history
        }

    def get_context(self, state: StorySageState) -> dict:
        """Retrieves relevant contextual excerpts for the user's question.

        This method queries the underlying document store based on the question
        and filtered entities/book/chapter details, then provides the matched text.

        Args:
            state (StorySageState): The current state containing question text and optional filters.

        Returns:
            dict: A dictionary containing a 'context' key with a list of relevant text excerpts.

        Example:
            >>> state = StorySageState(question="Where does John meet Sarah?", book_number=2, chapter_number=5)
            >>> context_data = chain_instance.get_context(state)
            >>> print(context_data['context'])
            ['Book 2, Chapter 5: John travels to the docks...', ...]

        Example Results:
            Returns selected text chunks that match the provided filters and can
            be used to generate a final answer.
        """
        print_state('get_context', state)

        results = self.retriever.retrieve_chunks(query_str=state['question'], context_filters=state['context_filters'])
        
        docs_with_metadata = zip(results['docs'][0], results['metadatas'][0])

        # Sort the results by book_number and chapter_number
        sorted_results = sorted(
           docs_with_metadata,
            key=lambda x: (x[1]['book_number'], x[1]['chapter_number'])
        )

        context = [ 
            f"Book {m[1]['book_number']}, Chapter {m[1]['chapter_number']}: {m[0]}" for m in sorted_results
        ]

        new_node_history = state['node_history'].append('GetContext') if state['node_history'] else ['GetContext']
        return {
            'context': context,
            'node_history': new_node_history
        }
  
    def generate(self, state: StorySageState) -> dict:
        """Generates the final answer using the language model.

        This method assembles the user's question and retrieved context into
        a prompt and then invokes the language model to produce a concise answer.

        Args:
            state (StorySageState): The current state containing the user's question and context.

        Returns:
            dict: A dictionary containing the generated 'answer' field.

        Example:
            >>> state = StorySageState(question="Who is the main character?",
            ...                        context=["Book 1, Chapter 1: The introduction of Rand al'Thor..."])
            >>> result = chain_instance.generate(state)
            >>> print(result['answer'])
            "The main character appears to be Rand al'Thor..."

        Example Results:
            The final answer is typically a human-readable text that references
            the relevant context while adhering to the guidelines (no outside info).
        """
        print_state(f"generate", state)
        # Combine context excerpts into a single string
        docs_content = '\n\n'.join(state['context'])
        # Invoke the prompt with the question and context
        messages = self.prompt.invoke(
            {'question': state['question'], 'context': docs_content, 
             'book_number': state['book_number'], 'chapter_number': state['chapter_number'],
             'conversation': state['conversation']}
        )
        self.logger.debug(f"LLM prompt: {messages}")
        # Get the response from the language model
        response = self.llm.invoke(messages)
        print(response)
        # Return the generated answer

        new_node_history = state['node_history'].append('Generate') if state['node_history'] else ['Generate']
        return {
            'answer': response.content,
            'node_history': new_node_history
        }

    def router_function(self, state: StorySageState) -> dict:
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
        new_node_history = state['node_history'].append('RouterFunction') if state['node_history'] else ['RouterFunction']
        return { 
            'order_by': tense,
            'node_history': new_node_history
        }
    
    def check_for_target_ids(self, state: StorySageState) -> Literal['by_id', 'full_search']:
        """
        Check if the state contains target_ids for chunk filtering.

        Args:
            state (StorySageState): The current state of the system.

        Returns:
            bool: True if target_ids are present, False otherwise.
        """
        if ('target_ids' in state and len(state['target_ids']) > 0):
            return 'by_id'
        else:
            return 'full_search'  
