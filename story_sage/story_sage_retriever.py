from .story_sage_embedder import StorySageEmbedder
from typing import List
import chromadb


class StorySageRetriever:
    def __init__(self, chroma_path: str, chroma_collection_name: str,
                 character_dict: dict, n_chunks: int = 5):
        
        self.embedder = StorySageEmbedder()
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.vector_store = self.chroma_client.get_collection(
            name=chroma_collection_name,
            embedding_function=self.embedder
        )
        self.character_dict = character_dict
        self.n_chunks = n_chunks

    def retrieve_chunks(self, query_str, book_number: int = None,
                        chapter_number: int = None, characters: List[str] = []) -> List[str]:
        
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
                    book_chapter_filter
                ]
            }
        else:
            query_filter = book_chapter_filter

        # query the vector store
        query_result = self.vector_store.query(
            query_texts=[query_str],
            n_results=self.n_chunks,
            include=['metadatas', 'documents'],
            where=query_filter
        )

        return query_result