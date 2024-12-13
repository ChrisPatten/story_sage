
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from .story_sage_state import StorySageState
from .story_sage_retriever import StorySageRetriever
import httpx


class StorySageChain(CompiledStateGraph):
  def __init__(self, api_key: str, state: StorySageState, character_dict: dict, retriever: StorySageRetriever):
    self.state = state
    self.character_dict = character_dict
    self.retriever = retriever
    self.llm = ChatOpenAI(api_key=api_key, model='gpt-4o-mini', http_client = httpx.Client(verify=False))
    self.prompt = PromptTemplate(
      input_variables=['question', 'context'],
      template="""
        HUMAN

        You are an assistant to help a reader keep track of people, places, and plot points in books.
        The following pieces of retrieved context are excerpts from the books related to the reader's question. Use them to generate your response.

        Guidelines for the response:
        * If you don't know the answer, just say that you don't know. 
        * If you're not sure about something, you can say that you're not sure.
        * Take as much time as you need to answer the question.
        * Use as many words as you need to answer the question completely, but don't provide any irrelevant information.
        * Use bullet points to provide examples from the context that support your answer.

        Question: {question} 
        Context: {context} 
        Answer:
      """
    )


    graph_builder = StateGraph(StorySageState).add_sequence([self._get_characters, self._get_context, self._generate])
    graph_builder.add_edge(START, self._get_characters)
    self.graph = graph_builder.compile()
  
  def _get_characters(self) -> dict:
    characters_in_question = set()
    for character in self.character_dict:
      if str.lower(character) in str.lower(self.state['question']):
        characters_in_question.add(character)
    return {'characters': list(characters_in_question)}
  
  def _get_context(self) -> dict:
    retrieved_docs = self.retriever.retrieve_chunks(
      query_str=self.state['question'],
      book_number=self.state['book_number'],
      chapter_number=self.state['chapter_number'],
      characters=self.state['characters']
    )
    return {'context': retrieved_docs}
  
  def _generate(self) -> dict:
    docs_content = '\n\n'.join(doc for doc in self.state['context']['documents'][0])
    messages = self.prompt.invoke(
      {'question': self.state['question'], 'context': docs_content}
    )
    response = self.llm.invoke(messages)
    return {'answer': response.content}