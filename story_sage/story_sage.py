# Imports
import os
from langchain_openai import ChatOpenAI
from langchain import hub, PromptTemplate
from langgraph.graph import START, StateGraph, CompiledStateGraph
from typing_extensions import List, TypedDict
from sentence_transformers import SentenceTransformer
import yaml
import httpx
import torch
import pickle
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

# Constants
CHROMA_PATH = '../chroma_data'
CHROMA_COLLECTION_NAME = 'wheel_of_time'
CHARACTER_DICT_PATH = '../merged_characters.pkl'


# Classes
class StorySageEmbedder(EmbeddingFunction):
  def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
    self.model = SentenceTransformer(model_name)
    self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    self.model = self.model.to(self.device)

  def __call__(self, input: Documents) -> Embeddings:
      return self.model.encode(input).tolist()
  
  def embed_documents(self, documents: Documents) -> Embeddings:
      embedded_documents = []
      for document in documents:
        embedded_document = self.model.encode(document)
        embedded_documents.append(embedded_document)
      return embedded_documents

class StorySageState(TypedDict):
    question: str
    context: List[str]
    answer: str
    book_number: int
    chapter_number: int
    n_chunks: int
    characters: List[str]

class StorySage:
  def __init__(self, chroma_path: str = CHROMA_PATH, 
               chroma_collection_name: str = CHROMA_COLLECTION_NAME,
               character_dict_path: str = CHARACTER_DICT_PATH,
               n_chunks: int = 5, llm: ChatOpenAI = None):
    self.embedder = StorySageEmbedder()
    self.chroma_client = chromadb.PersistentClient(path=chroma_path)
    self.vector_store = self.chroma_client.get_collection(
       name=chroma_collection_name,
       embedding_function=self.embedder
    )
    self.character_dict = self._load_character_dict(character_dict_path)
    self.llm = llm
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
    self.n_chunks = n_chunks

  def retrieve_chunks(self, query: str, book_number: int = None, 
                      chapter_number: int = None, characters: List[str] = []) -> List[str]:
    book_chapter_filter = {
        '$or': [
            {'book_number': {'$lt': book_number}},
            {'$and': [
                {'book_number': book_number},
                {'chapter_number': {'$lt': chapter_number}}
            ]}
        ]
    }

    if characters:
      characters_filter = []
      for character in characters:
        characters_filter.append({f'character_{character}': True})
      if len(characters_filter) == 1:
        characters_filter = characters_filter[0]
      else:
        characters_filter = {'$or': characters_filter}
      query_filter = {
        '$and': [
            characters_filter,
            book_chapter_filter
        ]
      }
    else:
        query_filter = book_chapter_filter

    query_result = self.vector_store.query(
      query_texts=[query], 
      n_results = self.n_chunks,
      include=['metadatas', 'documents'],
      where=query_filter
    )

    return query_result


  def _load_character_dict(self, character_dict_path: str) -> dict:
    with open(character_dict_path, 'rb') as f:
      character_dict = pickle.load(f)
    return character_dict
    
class StorySageGraph(CompiledStateGraph):
  def __init__(self, story_sage: StorySage, state: StorySageState, character_dict: dict):
    self.state = state
    self.character_dict = character_dict
    self.story_sage = story_sage
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
    retrieved_docs = self.story_sage.retrieve_chunks(
      query=self.state['question'],
      book_number=self.state['book_number'],
      chapter_number=self.state['chapter_number'],
      characters=self.state['characters']
    )
    return {'context': retrieved_docs}
  
  def _generate(self) -> dict:
    docs_content = '\n\n'.join(doc for doc in self.state['context']['documents'][0])
    messages = self.story_sage.prompt.invoke(
      {'question': self.state['question'], 'context': docs_content}
    )
    response = self.story_sage.llm.invoke(messages)
    return {'answer': response.content}

