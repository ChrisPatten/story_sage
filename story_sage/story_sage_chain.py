from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from .story_sage_state import StorySageState
from .story_sage_retriever import StorySageRetriever
import httpx


class StorySageChain(StateGraph):
    """Defines the chain of operations for the Story Sage system."""

    def __init__(self, api_key: str, character_dict: dict, retriever: StorySageRetriever):
        """
        Initialize the StorySageChain instance.

        Args:
            api_key (str): The API key for the language model.
            character_dict (dict): Dictionary containing character information.
            retriever (StorySageRetriever): The retriever instance for fetching context.
        """
        self.character_dict = character_dict
        self.retriever = retriever
        self.llm = ChatOpenAI(api_key=api_key, model='gpt-4o-mini', http_client = httpx.Client(verify=False))
        self.prompt = PromptTemplate(
            input_variables=['question', 'context'],
            template="""
                HUMAN

                You are an assistant to help a reader keep track of people, places, and plot points in books.
                The attached pieces of retrieved context are excerpts from the books related to the reader's question. Use them to generate your response.

                Guidelines for the response:
                * If you don't know the answer or aren't sure, just say that you don't know. 
                * Don't provide any irrelevant information.
                * Use bullet points to provide excerpts from the context that support your answer. Reference the book and chapter whenever you include an excerpt.

                Question: {question} 
                Context: {context} 
                Answer:
            """
        )

        graph_builder = StateGraph(StorySageState)
        graph_builder.add_node("GetCharacters", self.get_characters)
        graph_builder.add_node("GetContext", self.get_context)
        graph_builder.add_node("Generate", self.generate)
        graph_builder.add_edge(START, "GetCharacters")
        graph_builder.add_edge("GetCharacters", "GetContext")
        graph_builder.add_edge("GetContext", "Generate")
        graph_builder.add_edge("Generate", END)
        self.graph = graph_builder.compile()
  
    def get_characters(self, state: StorySageState) -> dict:
        """
        Extract characters mentioned in the user's question.

        Args:
            state (StorySageState): The current state of the system.

        Returns:
            dict: A dictionary with the list of characters found.
        """
        characters_in_question = set()
        for character in self.character_dict:
            if str.lower(character) in str.lower(state['question']):
                characters_in_question.add(character)
        return {'characters': list(characters_in_question)}
  
    def get_context(self, state: StorySageState) -> dict:
        """
        Retrieve context based on the user's question and extracted characters.

        Args:
            state (StorySageState): The current state of the system.

        Returns:
            dict: A dictionary containing the retrieved context.
        """
        retrieved_docs = self.retriever.retrieve_chunks(
            query_str=state['question'],
            book_number=state['book_number'],
            chapter_number=state['chapter_number'],
            characters=state['characters']
        )
        context = [
            f"book number: {meta['book_number']}, chapter: {meta['chapter_number']}, excerpt: {doc}"
            for meta, doc in zip(retrieved_docs['metadatas'][0], retrieved_docs['documents'][0])
        ]
        return {'context': context}
  
    def generate(self, state: StorySageState) -> dict:
        """
        Generate an answer based on the user's question and the retrieved context.

        Args:
            state (StorySageState): The current state of the system.

        Returns:
            dict: A dictionary containing the generated answer.
        """
        docs_content = '\n\n'.join(state['context'])
        messages = self.prompt.invoke(
            {'question': state['question'], 'context': docs_content}
        )
        response = self.llm.invoke(messages)
        return {'answer': response.content}