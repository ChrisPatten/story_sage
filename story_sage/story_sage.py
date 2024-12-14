import chromadb
from .story_sage_state import StorySageState
from .story_sage_retriever import StorySageRetriever
from .story_sage_chain import StorySageChain
class StorySage():
    def __init__(self, api_key: str, chroma_path: str, chroma_collection_name: str, character_dict: dict, n_chunks: int = 5):
        self.state = StorySageState()
        self.retriever = StorySageRetriever(chroma_path, chroma_collection_name, character_dict, n_chunks)
        self.chain = StorySageChain(api_key, character_dict, self.retriever)

    def invoke(self, question: str, book_number: int = 100, 
               chapter_number: int = 0, series_id: int = 0):
        self.state['question'] = question
        self.state['book_number'] = book_number
        self.state['chapter_number'] = chapter_number
        self.state['context'] = None
        self.state['answer'] = None
        result = self.chain.graph.invoke(self.state)
        self.state['context'] = result['context']
        self.state['answer'] = result['answer']
        return self.state['answer'], self.state['context']
