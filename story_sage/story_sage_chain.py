
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from .story_sage_state import StorySageState
from .story_sage_retriever import StorySageRetriever
import httpx


class StorySageChain(StateGraph):
  def __init__(self, api_key: str, character_dict: dict, retriever: StorySageRetriever):
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
    characters_in_question = set()
    for character in self.character_dict:
      if str.lower(character) in str.lower(state['question']):
        characters_in_question.add(character)
    return {'characters': list(characters_in_question)}
  
  def get_context(self, state: StorySageState) -> dict:
    retrieved_docs = self.retriever.retrieve_chunks(
      query_str=state['question'],
      book_number=state['book_number'],
      chapter_number=state['chapter_number'],
      characters=state['characters']
    )
    return {'context': retrieved_docs}
  
  def generate(self, state: StorySageState) -> dict:
    docs_content = '\n\n'.join(doc for doc in state['context']['documents'][0])
    messages = self.prompt.invoke(
      {'question': state['question'], 'context': docs_content}
    )
    response = self.llm.invoke(messages)
    return {'answer': response.content}