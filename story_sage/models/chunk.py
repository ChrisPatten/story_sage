from typing import List, Union
from numpy import ndarray
from chromadb import QueryResult, GetResult
import logging
from .context import StorySageContext

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
                 full_chunk: str = None,
                 match_type: str = None,
                 matched_terms: int = None,
                 matched_phrases: int = None,
                 window_size: int = None,
                 threshold: float = None,
                 is_summary: bool = False):
        self.series_id = series_id
        self.book_number = book_number
        self.chapter_number = chapter_number
        self.level = level
        self.chunk_index = chunk_index
        self.book_position = book_position
        self.book_filename = book_filename
        self.full_chunk = full_chunk
        self.match_type = match_type
        self.matched_terms = matched_terms
        self.matched_phrases = matched_phrases
        self.window_size = window_size
        self.threshold = threshold
        self.is_summary = is_summary

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
            "full_chunk": self.full_chunk,
            "match_type": self.match_type,
            "matched_terms": self.matched_terms,
            "matched_phrases": self.matched_phrases,
            "window_size": self.window_size,
            "threshold": self.threshold,
            "is_summary": self.is_summary
        }

    def update(self, metadata: dict) -> None:
        """Update metadata attributes from a dictionary.
        
        Args:
            metadata (dict): Dictionary of attributes to update.
        """
        for key, value in metadata.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_json(self) -> dict:
        """Convert metadata to JSON-serializable dictionary."""
        return self.__to_dict__()

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
        parents (List[str]): Keys of parent chunk keys (summaries of this chunk).
        children (List[str]): Keys of child chunk keys (chunks this summarizes).
    """
    def __init__(self, 
                 text: str, 
                 metadata: Union[ChunkMetadata|dict], 
                 embedding: List[ndarray]=None,
                 parents: List[str]=[],
                 children: List[str]=[]):
        
        
        self.text = text
        self.embedding = embedding
        if type(metadata) == dict:
            parents = metadata.pop('parents', parents)
            children = metadata.pop('children', children)
            if len(parents) > 0:
                if ',' in parents[0]:
                    parents = parents[0].split(',')
            if len(children) > 0:
                if ',' in children[0]:
                    children = children[0].split(',')
            try:
                self.metadata = ChunkMetadata(**metadata)
            except TypeError as e:
                raise ValueError(f"metadata dictionary invalid! {metadata}")
        elif type(metadata) == ChunkMetadata:
            self.metadata = metadata
        else:
            raise ValueError("metadata must be a dictionary or ChunkMetadata object.")
        
        self.chunk_key = self._create_chunk_key()
        self.parents = parents
        self.children = children
        self.is_summary = self.metadata.is_summary

    def as_context(self) -> StorySageContext:
        """Convert chunk to a StorySageContext object."""
        return StorySageContext(
            chunk_id=self.chunk_key,
            book_number=self.metadata.book_number,
            chapter_number=self.metadata.chapter_number,
            chunk=self.text
        )

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
        return f"Chunk: {self.chunk_key} * Text: {len(self.text)} * Parents: {len(self.parents)} * Children: {len(self.children)}" 
    
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

        if 'ids' in results:
            if len(results['ids']) == 0:
                logger.error("No results found.")
                return []
            chunk_data = []
            if isinstance(results['ids'][0], list): # QueryResult
                logger.info(f"Converting {len(results['ids'][0])} QueryResult to Chunk objects.")
                for i in range(len(results['ids'][0])):
                    text = results['metadatas'][0][i].pop('full_chunk')
                    metadata = results['metadatas'][0][i]
                    embeddings = results['embeddings'][0][i] if results.get('embeddings', False) else None
                    chunk_data.append((text, metadata, embeddings))
            else:
                logger.info(f"Converting {len(results['ids'])} GetResult to Chunk objects.")
                for i in range(len(results['ids'])): # GetResult
                    text = results['metadatas'][i].pop('full_chunk')
                    metadata = results['metadatas'][i]
                    embeddings = results['embeddings'][i] if results.get('embeddings', False) else None
                    chunk_data.append((text, metadata, embeddings))
            chunks = []
            for chunk_components in chunk_data:
                chunk = Chunk(text=chunk_components[0], 
                              metadata=chunk_components[1], 
                              embedding=chunk_components[2])
                chunks.append(chunk)
            return chunks
        else:
            logger.error("No results found.")
            return []
