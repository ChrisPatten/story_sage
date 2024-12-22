# Import necessary libraries and modules
import logging
from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from .story_sage_state import StorySageState
from .story_sage_retriever import StorySageRetriever
from .story_sage_config import StorySageConfig
from .story_sage_series import StorySageSeries
from .story_sage_entities import StorySageEntities
import httpx
from typing import Optional, List


class StorySageChain:
    """
    A class to manage the processing chain for the Story Sage system.

    Attributes:
        config (StorySageConfig): Configuration object.
        series_collection (list): List of StorySageSeries objects.
        retriever (StorySageRetriever): Retriever object for fetching story elements.
        logger (logging.LoggerAdapter): Logger for logging messages.
    """

    def __init__(self, config: StorySageConfig, series_collection: list[StorySageSeries], retriever: StorySageRetriever, logger: logging.LoggerAdapter):
        """
        Initialize the StorySageChain with the given configuration and series collection.

        Args:
            config (StorySageConfig): Configuration object.
            series_collection (list): List of StorySageSeries objects.
            retriever (StorySageRetriever): Retriever object for fetching story elements.
            logger (logging.LoggerAdapter): Logger for logging messages.
        """
        self.config = config
        self.series_collection = series_collection
        self.retriever = retriever
        self.logger = logger

        self.llm = ChatOpenAI(api_key=self.config.openai_api_key, model='gpt-4o-mini', http_client=httpx.Client(verify=False))
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
            series_id_str = str(state['series_id'])
            # Convert the question to lowercase for case-insensitive search
            question_text_search = state['question'].lower()
            # Retrieve series information based on series ID
            series: StorySageSeries = self.series_collection[series_id_str]
            series_entities: StorySageEntities = series.entities if series else None
            if series_entities is not None:
                all_entities_by_name = {**series_entities.people_by_name, **series_entities.entity_by_name}
                all_entities_by_id = {**series_entities.people_by_id, **series_entities.entity_by_id}
            else:
                self.logger.debug(f"No entities found for series ID {series_id}.")
                all_entities_by_name = {}
                all_entities_by_id = {}
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
    
    def get_candidate_chapters(self, state: StorySageState) -> List[dict]:
        """
        Get candidate chapters that are likely to contain the answer to the user's question.

        Args:
            query_str (str): The user's query.
            context_filters (dict): Dictionary containing context filters such as series, book, and chapter details.

        Returns:
            List[dict]: List of dictionaries containing book number and chapter number of candidate chapters.
        """
        # Retrieve summary chunks relevant to the query
        filters = {'series_id': state['series_id'], 'book_number': state['book_number'], 'chapter_number': state['chapter_number']}
        summary_chunks = self.retriever.retrieve_summary_chunks(state['question'], filters)

        # Log the retrieved summary chunks for debugging purposes
        self.logger.debug(f"Retrieved summary chunks: {summary_chunks}")

        # Prepare the data to send to the LLM
        candidate_chapters = []
        for summary in summary_chunks:
            for document in summary['documents']:
                metadata = document['metadata']
                candidate_chapters.append({
                    'book_number': metadata['book_number'],
                    'chapter_number': metadata['chapter_number'],
                    'summary': document['document']
                })

        # Send the candidate chapters along with the user's question to the LLM
        # Assuming there's a function `evaluate_candidates_with_llm` to interact with the LLM
        likely_chapters = self._evaluate_candidates_with_llm(state['question'], candidate_chapters)

        # Log the likely chapters for debugging purposes
        self.logger.debug(f"Likely chapters: {likely_chapters}")

        return { 'likely_chapters': likely_chapters }

    def _evaluate_candidates_with_llm(self, query_str, candidate_chapters):
        """
        Evaluate candidate chapters with the LLM to determine which chapters are likely to contain the answer.

        Args:
            query_str (str): The user's query.
            candidate_chapters (list): List of candidate chapters with their summaries.

        Returns:
            List[dict]: List of dictionaries containing book number and chapter number of likely chapters.
        """
        evaluation_prompt_templat = f"""
You are a literary analysis expert helping to determine which chapters of a book are most relevant to answering a user's question. Your task is to evaluate each chapter summary and rate its likelihood of containing information relevant to the question. Do not answer the question itself.

Consider the following when analyzing relevance:
1. Direct mentions of key terms or characters from the question
2. Related events or themes that might provide context
3. Character interactions or developments relevant to the query
4. World-building or background information that could inform the answer
5. Chronological relevance to the question's context

For each chapter summary, provide:
- Relevance Score (0-10)
- Brief explanation of why this chapter might be relevant
- Key elements that match the question
- Confidence level in the assessment (Low/Medium/High)

Question: <user_question>
Reading Progress: Book <X>, Chapter <Y>

Chapter Summaries to Analyze:
<list_of_chapter_summaries>

Format your response as follows for each chapter:

Chapter <N>:
Relevance Score: [0-10]
Confidence: [Low/Medium/High]
Reasoning: [1-2 sentences explaining relevance]
Key Elements: [bullet points of specific matching elements]
---

Only include chapters with a relevance score of 3 or higher in your response. Sort results by relevance score in descending order.

Remember:
- Do not reveal plot points beyond the user's reading progress
- Focus on identifying relevant information, not answering the question
- Consider both explicit and implicit connections to the question
- Account for context that might be necessary for a complete answer

        """
        return candidate_chapters
  
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