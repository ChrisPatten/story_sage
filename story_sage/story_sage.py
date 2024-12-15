import chromadb
from .story_sage_state import StorySageState
from .story_sage_retriever import StorySageRetriever
from .story_sage_chain import StorySageChain

class StorySage():
    """Class that handles the invocation of the Story Sage system."""

    def __init__(self, api_key: str, chroma_path: str, chroma_collection_name: str, character_dict: dict, n_chunks: int = 5):
        """
        Initialize the StorySage instance.

        Args:
            api_key (str): The API key for the language model.
            chroma_path (str): File path to the Chroma database.
            chroma_collection_name (str): Name of the Chroma collection.
            character_dict (dict): Dictionary containing character information.
            n_chunks (int, optional): Number of chunks to retrieve. Defaults to 5.
        """
        self.state = StorySageState()
        self.retriever = StorySageRetriever(chroma_path, chroma_collection_name, character_dict, n_chunks)
        self.chain = StorySageChain(api_key, character_dict, self.retriever)

    def invoke(self, question: str, book_number: int = 100, 
               chapter_number: int = 0, series_name: str = None) -> tuple:
        """
        Invoke the Story Sage system with a question and context parameters.

        Args:
            question (str): The user's question.
            book_number (int, optional): Book number for context filtering. Defaults to 100.
            chapter_number (int, optional): Chapter number for context filtering. Defaults to 0.
            series_id (int, optional): Series ID for context filtering. Defaults to 0.

        Returns:
            tuple: A tuple containing the answer and context.
        """
        self.state['question'] = question
        self.state['book_number'] = book_number
        self.state['chapter_number'] = chapter_number
        self.state['context'] = None
        self.state['answer'] = None
        self.state['series_name'] = series_name
        result = self.chain.graph.invoke(self.state)
        self.state['context'] = result['context']
        self.state['answer'] = result['answer']
        return self.state['answer'], self.state['context']
