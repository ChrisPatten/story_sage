from typing import List, Union
from numpy import ndarray
from chromadb import QueryResult, GetResult
import logging

class ChunkMetadata:
    """Container for metadata associated with a text chunk.
    
    Attributes:
        book_number (int): Sequential identifier for the book.
        chapter_number (int): Chapter number within the book.
        level (int): Hierarchy level (1=raw chunks, 2+=summaries).
        chunk_index (int): Sequential index within the level.
    """
    def __init__(self, 
                 chunk_index: int, 
                 book_number: int = None, 
                 chapter_number: int = None, 
                 level: int = None,
                 series_id: int = None,
                 book_position: float = None,
                 book_filename: str = None,
                 match_type: str = None,
                 matched_terms: int = None,
                 matched_phrases: int = None,
                 window_size: int = None,
                 threshold: float = None):
        self.series_id = series_id
        self.book_number = book_number
        self.chapter_number = chapter_number
        self.level = level
        self.chunk_index = chunk_index
        self.book_position = book_position
        self.book_filename = book_filename
        self.match_type = match_type
        self.matched_terms = matched_terms
        self.matched_phrases = matched_phrases
        self.window_size = window_size
        self.threshold = threshold

        self.logger = logging.getLogger(__name__)

    def __to_dict__(self) -> dict:
        return {
            "series_id": self.series_id,
            "book_number": self.book_number,
            "chapter_number": self.chapter_number,
            "level": self.level,
            "chunk_index": self.chunk_index,
            "book_position": self.book_position,
            "book_filename": self.book_filename,
            "match_type": self.match_type,
            "matched_terms": self.matched_terms,
            "matched_phrases": self.matched_phrases,
            "window_size": self.window_size,
            "threshold": self.threshold
        }

    def to_json(self) -> dict:
        """Convert metadata to JSON-serializable dictionary."""
        return {
            "series_id": self.series_id,
            "book_number": self.book_number,
            "chapter_number": self.chapter_number,
            "level": self.level,
            "chunk_index": self.chunk_index,
            "book_position": self.book_position,
            "book_filename": self.book_filename,
            "match_type": self.match_type,
            "matched_terms": self.matched_terms,
            "matched_phrases": self.matched_phrases,
            "window_size": self.window_size,
            "threshold": self.threshold
        }

class Chunk:
    """Container for text data with hierarchical relationships.
    
    A Chunk represents either an original text segment or a summary, storing both
    content and relationships to other chunks in the hierarchy.
    
    Attributes:
        text (str): The text content.
        metadata (ChunkMetadata): Associated metadata.
        is_summary (bool): True if this chunk is a summary of other chunks.
        embedding (np.ndarray): Vector representation of the text.
        chunk_key (str): Unique identifier for this chunk.
        parents (List[str]): Keys of parent chunks (summaries of this chunk).
        children (List[str]): Keys of child chunks (chunks this summarizes).
    """
    def __init__(self, 
                 text: str, 
                 metadata: Union[ChunkMetadata|dict], 
                 is_summary: bool=False,
                 embedding: List[ndarray]=None,
                 parents: List[str]=None,
                 children: List[str]=None):
        
        self.text = text
        self.is_summary = is_summary
        self.embedding = embedding
        if type(metadata) == dict:
            self.metadata = ChunkMetadata(**metadata)
        elif type(metadata) == ChunkMetadata:
            self.metadata = metadata
        else:
            raise ValueError("metadata must be a dictionary or ChunkMetadata object.")
        
        self.chunk_key = self._create_chunk_key()
        self.parents = parents if parents is not None else []
        self.children = children if children is not None else []

    def _create_chunk_key(self) -> str:
        """Generates a unique chunk key from the metadata in a consistent format."""
        parts = []
        if hasattr(self.metadata, 'series_id'):
            parts.append(f"series_{self.metadata.series_id}")
        if self.metadata.book_number is not None:
            parts.append(f"book_{self.metadata.book_number}")
        if self.metadata.chapter_number is not None:
            parts.append(f"chapter_{self.metadata.chapter_number}")
        if self.metadata.level is not None:
            parts.append(f"level_{self.metadata.level}")
        if self.metadata.chunk_index is not None:
            parts.append(f"chunk_{self.metadata.chunk_index}")
        return "|".join(parts)
    
    def __string__(self) -> str:
        return f"Chunk: {self.chunk_key} * Parents: {self.parents} * Children: {self.children}" 
    
    def __repr__(self) -> str:
        return self.__string__()
    
    def __json__(self) -> dict:
        return {
            "text": self.text,
            "metadata": self.metadata.__dict__,
            "chunk_key": self.chunk_key,
            "parents": self.parents,
            "children": self.children
        }

    @classmethod
    def from_data(cls, data: dict):
        """Create a Chunk instance from the given data format."""
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary. Got: {}".format(data))
        
        if 'ids' in data and isinstance(data['ids'][0], list):
            text = data['documents'][0][0]
            metadata_dict = data['metadatas'][0][0]
            embedding = data['embeddings'][0] if data['embeddings'] else None
        elif 'ids' in data:
            text = data['documents'][0]
            metadata_dict = data['metadatas'][0]
            embedding = data['embeddings'][0] if data['embeddings'] else None
        else:
            raise ValueError("Data format not recognized: {}".format(data))

        is_summary = metadata_dict.get('is_summary', False)
        parents = metadata_dict.get('parents', '').split(',') if metadata_dict.get('parents') else []
        children = metadata_dict.get('children', '').split(',') if metadata_dict.get('children') else []
        
        metadata = ChunkMetadata(
            chunk_index=metadata_dict['chunk_index'],
            book_number=metadata_dict.get('book_number'),
            chapter_number=metadata_dict.get('chapter_number'),
            level=metadata_dict.get('level'),
            series_id=metadata_dict.get('series_id'),
            book_filename=metadata_dict.get('book_filename'),
            match_type=metadata_dict.get('match_type'),
            matched_terms=metadata_dict.get('matched_terms'),
            matched_phrases=metadata_dict.get('matched_phrases'),
            window_size=metadata_dict.get('window_size'),
            threshold=metadata_dict.get('threshold')
        )
        
        return cls(text=text, metadata=metadata, is_summary=is_summary, embedding=embedding, parents=parents, children=children)

    def to_json(self) -> dict:
        """Convert chunk to JSON-serializable dictionary."""
        return {
            "text": self.text,
            "metadata": self.metadata.to_json(),
            "is_summary": self.is_summary,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "chunk_key": self.chunk_key,
            "parents": self.parents,
            "children": self.children
        }
    
    @staticmethod
    def from_chroma_results(results: Union[QueryResult, GetResult]) -> List['Chunk']:
        """Convert a list of search results to a list of Chunk objects."""
        logger = logging.getLogger(__name__)

        chunks = []
        if 'ids' in results:
            if len(results['ids']) == 0:
                logger.error("No results found.")
                return []
            if isinstance(results['ids'][0], list): # QueryResult
                for i in range(len(results['ids'][0])):
                    chunks.append(Chunk.from_data({
                        'ids': [results['ids'][0][i]],
                        'documents': [results['documents'][0][i]],
                        'metadatas': [results['metadatas'][0][i]],
                        'embeddings': [results['embeddings'][0][i]] if results.get('embeddings', False) else None
                    }))
            else:
                for i in range(len(results['ids'])): # GetResult
                    chunks.append(Chunk.from_data({
                        'ids': [results['ids'][i]],
                        'documents': [results['documents'][i]],
                        'metadatas': [results['metadatas'][i]],
                        'embeddings': [results['embeddings'][i]] if results.get('embeddings', False) else None
                    }))
            return chunks
        else:
            logger.error("No results found.")
            return []
        