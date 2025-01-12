"""Text analysis and hierarchical clustering module for story processing.

This module implements a hierarchical text analysis system that processes books 
through multiple levels using the StorySage infrastructure. It combines:
- StorySageConfig for configuration management
- StorySageChunker for text segmentation
- UMAP for dimensionality reduction
- Gaussian Mixture Models for clustering
- OpenAI GPT models for summarization

Key Features:
    - Multi-level hierarchical analysis
    - Intelligent text chunking with configurable overlap
    - Two-phase clustering (global and local)
    - Automatic cluster optimization
    - GPT-powered summaries
    - Progress tracking

Example:
    ```python
    # Initialize with StorySageConfig
    config_path = "config.yaml"  # Contains OpenAI keys, Redis settings, etc.
    processor = RaptorProcessor(
        config_path=config_path,
        model_name="gpt-4o-mini",
        chunk_size=1000,
        chunk_overlap=50,
        clustering_threshold=0.5,
        metric='cosine'
    )
    
    # Process a book or series
    results = processor.process_texts(
        "path/to/books/*.txt",  # Glob pattern supported
        number_of_levels=3
    )
    
    # Access the hierarchical results
    for book_path, book_data in results.items():
        # Each chapter contains multiple analysis levels
        for chapter_key, chapter_data in book_data.items():
            # Level 1: Original chunks
            original_chunks = chapter_data['level_1']
            
            # Level 2: First-level summaries
            summaries = chapter_data['level_2']
            
            # Example: Print a summary's metadata and children
            for summary in summaries:
                print(f"Summary: {summary.text[:100]}...")
                print(f"Metadata: {summary.metadata.__dict__}")
                print(f"Summarizes chunks: {summary.children}")
    ```

Dependencies:
    - OpenAI: API access for GPT models
    - UMAP: Dimensionality reduction
    - sklearn: Gaussian Mixture Models
    - tqdm: Progress tracking
    - StorySageConfig: Configuration (.yaml format)
    - StorySageChunker: Text processing utilities

Notes:
    - Requires valid OpenAI API key in config.yaml
    - Large texts are processed in chunks to manage memory
    - Uses cosine similarity by default for text embeddings
    - Supports various UMAP metrics for different use cases
"""

from .chunker import StorySageChunker
from ..models import StorySageConfig
from openai import OpenAI
import httpx
import os
import numpy as np
import umap.umap_ as umap
from typing import List, Tuple, Dict, Literal, Union, TypeAlias
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
from tqdm.notebook import tqdm

# Valid UMAP metric values
UMAPMetric = Literal['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'canberra', 'braycurtis',
                    'haversine', 'mahalanobis', 'wminkowski', 'seuclidean', 'cosine', 'correlation',
                    'hamming', 'jaccard', 'dice', 'russelrao', 'kulsinski', 'rogerstanimoto',
                    'sokalmichener', 'sokalsneath', 'yule']

ModelName = Literal['gpt-4o-mini', 'gpt-4o', 'gpt-o1-mini', 'gpt-o1', 'gpt-3.5-turbo', 'gpt-3.5']

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
                 level: int = None):
        self.book_number = book_number
        self.chapter_number = chapter_number
        self.level = level
        self.chunk_index = chunk_index

    def __to_dict__(self) -> dict:
        return {
            "book_number": self.book_number,
            "chapter_number": self.chapter_number,
            "level": self.level,
            "chunk_index": self.chunk_index
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
                 embedding: List[np.ndarray]=None):
        
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
        self.parents: List[str] = []
        self.children: List[str] = []

    def _create_chunk_key(self) -> str:
        """Generates a unique chunk key from the metadata in a consistent format."""
        parts = []
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

class ChunkHierarchy:
    """Utility class to store hierarchical chunked text data as it's processed."""

    def __init__(self, level: int, summary_chunk_key: str, metadata: Union[ChunkMetadata|dict]):
        self.level = level
        self.summary_chunk_key = summary_chunk_key

        if type(metadata) == dict:
            self.metadata = ChunkMetadata(**metadata)
        elif type(metadata) == ChunkMetadata:
            self.metadata = metadata
        else:
            raise ValueError("metadata must be a dictionary or ChunkMetadata object.")

_LevelsDict: TypeAlias = Dict[str, List[Chunk]]
"""Type alias for a dictionary of levels containing chunked text data.
   
    Example:
    {
        'level_1': [Chunk, Chunk, ...],
        'level_2': [Chunk, Chunk, ...],
        ...
    }
"""

class RaptorProcessor:
    """Hierarchical text analysis system using clustering and summarization.
    
    This class implements a multi-level analysis pipeline that integrates with
    StorySage's configuration and chunking infrastructure. The pipeline:
    1. Uses StorySageChunker to split text into semantic chunks
    2. Generates embeddings for each chunk
    3. Performs two-phase clustering (global then local)
    4. Generates summaries using OpenAI's GPT models
    5. Builds a hierarchical tree of text chunks and summaries
    
    Args:
        config_path: Path to StorySage configuration YAML
        skip_summarization: If True, skips GPT summary generation
        seed: Random seed for reproducibility
        model_name: OpenAI model identifier
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        clustering_threshold: Minimum probability for cluster membership
        target_dim: Target dimensions after UMAP reduction
        max_tokens: Maximum tokens for GPT summaries
        n_completions: Number of summary attempts
        stop_sequence: Optional GPT stop token
        temperature: GPT temperature setting
        max_clusters: Maximum clusters to consider
        metric: UMAP distance metric
        n_init: GMM initialization attempts
    
    Attributes:
        chunker: StorySageChunker instance
        client: OpenAI API client
        config: StorySageConfig instance
        chunk_tree: Hierarchical results structure
    
    Example Config YAML:
        ```yaml
        OPENAI_API_KEY: "sk-..."
        CHROMA_PATH: "./chromadb"
        CHROMA_COLLECTION: "book_embeddings"
        N_CHUNKS: 5
        COMPLETION_MODEL: "gpt-3.5-turbo"
        COMPLETION_TEMPERATURE: 0.7
        COMPLETION_MAX_TOKENS: 2000
        ```
    """
    def __init__(self, 
                 config_path: str,
                 skip_summarization: bool = False,
                 seed: int = 8675309, 
                 model_name: ModelName = "gpt-4o-mini", 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 50,
                 clustering_threshold: float = 0.5,
                 target_dim: int = 10,
                 max_tokens: int = 200,
                 n_completions: int = 1,
                 stop_sequence: str = None,
                 temperature: float = 0.7,
                 max_clusters: int = 50,
                 metric: UMAPMetric = 'cosine',
                 n_init: int = 2):
        os.environ['TOKENIZERS_PARALLELISM'] = "false"
        self.seed = seed
        self.config = StorySageConfig.from_file(config_path)
        self.chunker = StorySageChunker()
        self.client = OpenAI(api_key=self.config.openai_api_key,
                             http_client=httpx.Client(verify=False))
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap 
        self.clustering_threshold = clustering_threshold
        self.target_dim = target_dim
        self.max_tokens = max_tokens
        self.n_completions = n_completions
        self.stop_sequence = stop_sequence
        self.temperature = temperature
        self.max_clusters = max_clusters
        self.metric = metric
        self.n_init = n_init
        self.skip_summarization = skip_summarization

        self.chunk_tree = None

    def _dimensionality_reduction(self,
                                  embeddings: np.ndarray,
                                  target_dim: int,
                                  clustering_type: str) -> np.ndarray:
        """
        Reduces the dimensionality of embeddings using UMAP.

        Args:
            embeddings (np.ndarray): The input embeddings to reduce.
            target_dim (int): The target number of dimensions.
            clustering_type (str): Type of clustering ('local' or 'global').

        Returns:
            np.ndarray: The reduced embeddings.
        """
        if clustering_type == "local":
            n_neighbors = max(2, min(10, len(embeddings) - 1))
            min_dist = 0.01
        elif clustering_type == "global":
            n_neighbors = max(2, min(int((len(embeddings) - 1) ** 0.5), len(embeddings) // 10, len(embeddings) - 1))
            min_dist = 0.1
        else:
            raise ValueError("clustering_type must be either 'local' or 'global'")

        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=target_dim,
            metric=self.metric,
        )
        return umap_model.fit_transform(embeddings)

    def _compute_inertia(self, embeddings: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Computes the inertia (sum of squared distances) for clustering.

        Args:
            embeddings (np.ndarray): The input embeddings.
            labels (np.ndarray): Cluster labels for each embedding.
            centroids (np.ndarray): Centroid positions for each cluster.

        Returns:
            float: The computed inertia.
        """
        return np.sum(np.min(np.sum((embeddings[:, np.newaxis] - centroids) ** 2, axis=2), axis=1))

    def _optimal_cluster_number(
        self,
        embeddings: np.ndarray,
        random_state: int = None
    ) -> int:
        """
        Determines the optimal number of clusters using inertia and BIC scores.

        Args:
            embeddings (np.ndarray): The input embeddings.
            random_state (int, optional): Random state for reproducibility. Defaults to SEED.

        Returns:
            int: The optimal number of clusters.
        """
        random_state = self.seed if random_state is None else random_state
        max_clusters = min(self.max_clusters, len(embeddings))
        number_of_clusters = np.arange(1, max_clusters + 1)
        inertias = []
        bic_scores = []
        
        for n in number_of_clusters:
            gmm = GaussianMixture(n_components=n, random_state=random_state)
            labels = gmm.fit_predict(embeddings)
            centroids = gmm.means_
            inertia = self._compute_inertia(embeddings, labels, centroids)
            inertias.append(inertia)
            bic_scores.append(gmm.bic(embeddings))
        
        inertia_changes = np.diff(inertias)
        elbow_optimal = number_of_clusters[np.argmin(inertia_changes) + 1]
        bic_optimal = number_of_clusters[np.argmin(bic_scores)]
        
        return max(elbow_optimal, bic_optimal)

    def _gmm_clustering(
        self,
        embeddings: np.ndarray, 
        random_state: int = None
    ) -> Tuple[List[np.ndarray], int]:
        """
        Performs Gaussian Mixture Model clustering on embeddings.

        Args:
            embeddings (np.ndarray): The input embeddings.
            random_state (int, optional): Random state for reproducibility. Defaults to SEED.

        Returns:
            Tuple[List[np.ndarray], int]: A list of cluster labels and the number of clusters.
        """
        random_state = self.seed if random_state is None else random_state
        n_clusters = self._optimal_cluster_number(embeddings, random_state=random_state)
        gm = GaussianMixture(n_components=n_clusters, random_state=random_state, n_init=self.n_init)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > self.clustering_threshold)[0] for prob in probs] 
        return labels, n_clusters  

    def _clustering_algorithm(
        self,
        embeddings: np.ndarray,
        random_state: int = None
    ) -> Tuple[List[np.ndarray], int]:
        """
        Clustering algorithm that performs global and local clustering.

        Args:
            embeddings (np.ndarray): The input embeddings.
            random_state (int, optional): Random state for reproducibility. Defaults to SEED.

        Returns:
            Tuple[List[np.ndarray], int]: A list of local cluster labels and the total number of clusters.
        """
        random_state = self.seed if random_state is None else random_state
        if len(embeddings) <= self.target_dim + 1:
            return [np.array([0]) for _ in range(len(embeddings))], 1
        
        # Global clustering: uses a 'global' dimension reduction and lumps embeddings into broad groups.
        reduced_global_embeddings = self._dimensionality_reduction(embeddings, self.target_dim, "global")
        global_clusters, n_global_clusters = self._gmm_clustering(reduced_global_embeddings, random_state=random_state)

        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        # Local clustering within each global cluster: uses 'local' dimension reduction.
        for i in range(n_global_clusters):
            global_cluster_mask = np.array([i in gc for gc in global_clusters])
            global_cluster_embeddings = embeddings[global_cluster_mask]

            if len(global_cluster_embeddings) <= self.target_dim + 1:
                # Assign all points in this global cluster to a single local cluster
                for idx in np.where(global_cluster_mask)[0]:
                    all_local_clusters[idx] = np.append(all_local_clusters[idx], total_clusters)
                total_clusters += 1
                continue

            try:
                reduced_local_embeddings = self._dimensionality_reduction(global_cluster_embeddings, self.target_dim, "local")
                local_clusters, n_local_clusters = self._gmm_clustering(reduced_local_embeddings, random_state=random_state)

                # Assign local cluster IDs
                for j in range(n_local_clusters):
                    local_cluster_mask = np.array([j in lc for lc in local_clusters])
                    global_indices = np.where(global_cluster_mask)[0]
                    local_indices = global_indices[local_cluster_mask]
                    for idx in local_indices:
                        all_local_clusters[idx] = np.append(all_local_clusters[idx], j + total_clusters)

                total_clusters += n_local_clusters
            except Exception as e:
                print(f"Error in local clustering for global cluster {i}: {str(e)}")
                # Assign all points in this global cluster to a single local cluster
                for idx in np.where(global_cluster_mask)[0]:
                    all_local_clusters[idx] = np.append(all_local_clusters[idx], total_clusters)
                total_clusters += 1

        return all_local_clusters, total_clusters
    
    def _generate_summary(self, context: str) -> str:
        """
        Generates a summary for the given context using a language model.

        Args:
            context (str): The text to summarize.

        Returns:
            str: The generated summary.
        """
        if self.skip_summarization:
            return context

        prompt = f"""
        Provide the Summary for the given context. Here are some additional instructions for you:

        Instructions:
        1. Don't make things up, Just use the contexts and generate the relevant summary.
        2. Don't mix the numbers, Just use the numbers in the context.
        3. Don't try to use fancy words, stick to the basics of the language that is being used in the context.

        Context: {context}
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            n=self.n_completions,
            stop=self.stop_sequence,
            temperature=self.temperature
        )
        summary = response.choices[0].message.content.strip()
        return summary

    def _recursive_embedding_with_cluster_summarization(
        self,
        chunks: _LevelsDict,
        chapter_num: int,
        chunk_tree_key: str,
        number_of_levels: int = 3,
        level: int = 1
    ) -> _LevelsDict:
        """
        Recursively embeds texts and generates cluster summaries up to a specified number of levels.

        Args:
            texts (List[str]): The list of texts to process.
            number_of_levels (int, optional): The maximum number of recursion levels. Defaults to 3.
            level (int, optional): The current recursion level. Defaults to 1.

        Returns:
            Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]: A dictionary mapping levels to their cluster and summary DataFrames.
        """
        chapter_key = f'chapter_{chapter_num}'

        if level == 1:
            self.chunk_tree[chunk_tree_key][chapter_key] = {}

        while level <= number_of_levels:
            for level_key, chunk_list in chunks.items():
                # Skip if not the current level
                if not level_key == f'level_{level}':
                    continue

                # Create a list for the next level chunks
                if level + 1 <= number_of_levels:
                    next_level: List[Chunk] = []

                # Embed the chunks
                for chunk in chunk_list:
                    chunk.embedding = self.chunker.model.encode(chunk.text)

                # Collect the embeddings   
                chapter_embeddings: np.array[np.ndarray] = np.array([chunk.embedding for chunk in chunk_list])

                # Cluster the embeddings
                clusters_list, num_clusters = self._clustering_algorithm(chapter_embeddings)
                # Collect the chunk keys for each cluster
                summary_chunks_to_build = [[] for _ in range(num_clusters)]
                for idx, clusters in enumerate(clusters_list):
                    for cluster in clusters:
                        summary_chunks_to_build[int(cluster)].append(chunk_list[idx].chunk_key)

                # Build the summaries for each cluster
                for cluster_idx, chunk_keys in enumerate(summary_chunks_to_build):
                    # Set the metadata based on the first chunk in the cluster
                    cluster_metadata = {
                        "book_number": chunk_list[0].metadata.book_number,
                        "chapter_number": chunk_list[0].metadata.chapter_number,
                        "level": level + 1,
                        "chunk_index": cluster_idx
                    }
                    # Collect the texts for the cluster
                    cluster_texts = [chunk.text for chunk in chunk_list if chunk.chunk_key in chunk_keys]
                    summary_text = "------\n------".join(cluster_texts)
                    summary = self._generate_summary(summary_text)
                    # Embed the summary
                    summary_embedding = self.chunker.model.encode(summary)
                    # Add the summary to the next level
                    next_level.append(Chunk(summary, 
                                            metadata=cluster_metadata, 
                                            embedding=summary_embedding, 
                                            is_summary=True))
                
                for chunk_idx, parent_idx in enumerate(clusters_list):
                    chunk_list[chunk_idx].parents.append(next_level[int(parent_idx)].chunk_key)
                    next_level[int(parent_idx)].children.append(chunk_list[chunk_idx].chunk_key)
                
                self.chunk_tree[chunk_tree_key][chapter_key][level_key] = chunk_list
                self.chunk_tree[chunk_tree_key][chapter_key][f'level_{level + 1}'] = next_level

                # Run for the next level
                if num_clusters > 1:
                    chunks = self._recursive_embedding_with_cluster_summarization(chunks, 
                                                                                chapter_num,
                                                                                chunk_tree_key,
                                                                                number_of_levels, 
                                                                                level + 1)
            return chunks

    def _get_chunks_from_filepath(self, file_path: str) -> OrderedDict[str, list[list[Chunk]]]:
        """Loads text from a file path and chunks it into manageable segments.

        Args:
            file_path (str): The path to the file containing the text to process. Accepts glob strings.

        Returns:
            OrderedDict[str, list[list[Chunk]]]: A dictionary of {book_filepath: [[chunks]]}}. List index is chapter number.
        """

        chunked_text: OrderedDict[str, list[list[Chunk]]] = {}
        input_text = self.chunker.read_text_files(file_path)
        if len(input_text) < 1:
            raise ValueError("No text found in the provided file path.")
        for book_filepath, book_info in sorted(input_text.items()):
            chunked_text[book_filepath] = []
            for chapter_num, chapter_data in book_info.chapters.items():
                chunks = self.chunker.sentence_splitter(
                    chapter_data.full_text, 
                    chunk_size=self.chunk_size, 
                    chunk_overlap=self.chunk_overlap
                )
                chunk_list = []
                for idx, text in enumerate(chunks):
                    chunk_metadata = {"book_number": book_info.book_number, "chapter_number": chapter_num, "level": 1, "chunk_index": idx}
                    chunk_list.append(Chunk(text, metadata=chunk_metadata))
                chunked_text[book_filepath].append(chunk_list)
        return chunked_text

    def process_texts(self,
                      file_path: str,
                      number_of_levels: int = 3) -> Dict[str, Dict[str, _LevelsDict]]:
        """
        Processes a hierarchy of texts from a file path by embedding, clustering, and summarizing across multiple levels.

        Args:
            file_path (str): The path to the file containing the texts to process.
            number_of_levels (int, optional): The number of hierarchical levels. Defaults to 3.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary of DataFrames containing cluster and summary information for each level.
        """
        
        self.chunk_tree: Dict[str, Dict[str, _LevelsDict]] = {}

        chunked_text = self._get_chunks_from_filepath(file_path)
        if len(chunked_text) < 1:
            raise ValueError("No text found in the provided file path.")
        
        # Get level 1 chunks for each book into the tree
        for book_filepath, book_info in chunked_text.items():
            self.chunk_tree[book_filepath] = {}
            for chapter_num, chapter_chunks in tqdm(enumerate(book_info), desc=f"Processing {book_filepath}", total=len(book_info)):
                chapter_levels: _LevelsDict = { 'level_1': [chunk for chunk in chapter_chunks] }
                self._recursive_embedding_with_cluster_summarization(chapter_levels, 
                                                                     chapter_num=chapter_num, 
                                                                     chunk_tree_key=book_filepath)

        return self.chunk_tree