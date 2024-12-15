from .story_sage_embedder import StorySageEmbedder
from typing import List
import chromadb


class StorySageRetriever:
    """Class responsible for retrieving relevant chunks of text based on the user's query."""

    def __init__(self, chroma_path: str, chroma_collection_name: str,
                 character_dict: dict, n_chunks: int = 5):
        """
        Initialize the StorySageRetriever instance.

        Args:
            chroma_path (str): Path to the Chroma database.
            chroma_collection_name (str): Name of the Chroma collection.
            character_dict (dict): Dictionary containing character information.
            n_chunks (int, optional): Number of chunks to retrieve. Defaults to 5.
        """
        
        self.embedder = StorySageEmbedder()
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.vector_store = self.chroma_client.get_collection(
            name=chroma_collection_name,
            embedding_function=self.embedder
        )
        self.character_dict = character_dict
        self.n_chunks = n_chunks

    def retrieve_chunks(self, query_str, book_number: int = None,
                        chapter_number: int = None, characters: List[str] = [],
                        series_name: str = None) -> List[str]:
        """
        Retrieve chunks of text relevant to the query and filtering parameters.

        Args:
            query_str (str): The user's query.
            book_number (int, optional): Book number for context filtering.
            chapter_number (int, optional): Chapter number for context filtering.
            characters (List[str], optional): List of characters to filter by.
            series_name (str, optional): Name of the series to filter by.

        Returns:
            List[str]: Retrieved documents containing relevant context.
        """
        
        # configure series filter
        if series_name:
            series_filter = {'series_name': series_name}
            combined_filter = series_filter
            # configure book and chapter filter
            book_chapter_filter = {
                '$or': [
                    {'book_number': {'$lt': book_number}},
                    {'$and': [
                        {'book_number': book_number},
                        {'chapter_number': {'$lt': chapter_number}}
                    ]}
                ]
            }
            combined_filter = {
                '$and': [
                    combined_filter,
                    book_chapter_filter
                ]
            }
        else:
            # configure book and chapter filter
            book_chapter_filter = {
                '$or': [
                    {'book_number': {'$lt': book_number}},
                    {'$and': [
                        {'book_number': book_number},
                        {'chapter_number': {'$lt': chapter_number}}
                    ]}
                ]
            }
            combined_filter = book_chapter_filter
        
        # configure character filter
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
                    combined_filter
                ]
            }
        else:
            query_filter = combined_filter

        # query the vector store
        query_result = self.vector_store.query(
            query_texts=[query_str],
            n_results=self.n_chunks,
            include=['metadatas', 'documents'],
            where=query_filter
        )

        return query_result