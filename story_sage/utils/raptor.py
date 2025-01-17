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
from multiprocessing import Manager, Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import pathlib

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
                 level: int = None,
                 series_id: int = None):
        self.series_id = series_id
        self.book_number = book_number
        self.chapter_number = chapter_number
        self.level = level
        self.chunk_index = chunk_index

    def __to_dict__(self) -> dict:
        return {
            "series_id": self.series_id,
            "book_number": self.book_number,
            "chapter_number": self.chapter_number,
            "level": self.level,
            "chunk_index": self.chunk_index
        }

    def to_json(self) -> dict:
        """Convert metadata to JSON-serializable dictionary."""
        return {
            "series_id": self.series_id,
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

_LevelsDict: TypeAlias = Dict[str, List[Chunk]]
"""Type alias for a dictionary of levels containing chunked text data.
   
    Example:
    {
        'level_1': [Chunk, Chunk, ...],
        'level_2': [Chunk, Chunk, ...],
        ...
    }
"""

_ChapterWorkerArgs: TypeAlias = Tuple[List[Chunk], int, str, str, dict, int]

def _process_chapter_worker(args: _ChapterWorkerArgs) -> Dict[str, _LevelsDict]:
    """Process a single chapter's text chunks in parallel.

    This is a worker function designed to run in a separate process, handling
    one chapter's worth of text processing including embedding, clustering,
    and summarization.

    Args:
        args (_ChapterWorkerArgs): Tuple containing:
            - chapter_data (List[Chunk]): Text chunks for this chapter
            - chapter_num (int): Chapter number being processed
            - book_filename (str): Path to the book file
            - config_path (str): Path to StorySage config
            - book_tree (dict): Shared dictionary for storing results
            - number_of_levels (int): Number of hierarchy levels to generate

    Returns:
        Dict[str, _LevelsDict]: Updated book_tree containing this chapter's results
    """
    chapter_data, chapter_num, book_filename, config_path, book_tree, number_of_levels = args
    # Initialize processor for this process
    processor = RaptorProcessor(config_path)
    chapter_levels: _LevelsDict = {'level_1': [chunk for chunk in chapter_data]}
    print(f'Processing chapter {chapter_num} in {book_filename}...')
    chapter_results = processor._recursive_embedding_with_cluster_summarization(
        chunks=chapter_levels,
        chapter_num=chapter_num,
        book_tree=book_tree,
        book_filename=book_filename,
        number_of_levels=number_of_levels
    )
    book_tree[f'chapter_{chapter_num}'] = chapter_results
    return book_tree

_ClusterChunkData: TypeAlias = Tuple[List[Chunk], List[int], int, int]

_RaptorResults: TypeAlias = Dict[str, Dict[str, _LevelsDict]]
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
        CHROMA_FULL_TEXT_COLLECTION: "book_texts"
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
                 n_init: int = 2,
                 max_summary_threads: int = 1,
                 max_processes: int = None,
                 max_levels: int = 3,
                 series_id: int = None):
        os.environ['TOKENIZERS_PARALLELISM'] = "false"
        self.seed = seed
        self.config_path = config_path
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
        self.max_summary_threads = max_summary_threads
        self.max_levels = max_levels
        self.series_id = series_id
        
        self.max_processes = max_processes or cpu_count()
        

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
    
    def _process_cluster(self, cluster_data: _ClusterChunkData) -> Tuple[str, dict]:
        """Process a single cluster to generate its summary.

        Takes a cluster's worth of chunks and generates a summary, handling
        metadata creation and text processing.

        Args:
            cluster_data (_ClusterChunkData): Tuple containing:
                - chunk_list (List[Chunk]): All chunks in the current level
                - chunk_keys (List[int]): Indices of chunks in this cluster
                - cluster_idx (int): Index of this cluster
                - level (int): Current hierarchy level

        Returns:
            Tuple[str, dict]: Generated summary text and associated metadata

        Example:
            >>> cluster_data = (chunks, [0, 1, 2], 0, 1)
            >>> summary, metadata = processor._process_cluster(cluster_data)
            >>> print(f"Summary length: {len(summary)}")
            Summary length: 150
        """
        chunk_list, chunk_keys, cluster_idx, level = cluster_data
    
        # Set the metadata
        cluster_metadata = {
            "book_number": chunk_list[0].metadata.book_number,
            "chapter_number": chunk_list[0].metadata.chapter_number,
            "level": level + 1,
            "chunk_index": cluster_idx
        }
        
        # Collect and summarize texts
        cluster_texts = [chunk.text for chunk in chunk_list if chunk.chunk_key in chunk_keys]
        summary_text = "------\n------".join(cluster_texts)
        #print(f"Summarizing cluster {cluster_metadata}...")
        summary = self._generate_summary(summary_text)
        
        return summary, cluster_metadata

    def _recursive_embedding_with_cluster_summarization(
        self,
        chunks: _LevelsDict,
        chapter_num: int,
        book_tree: dict,
        book_filename: str,
        number_of_levels: int = 3,
        level: int = 1
    ) -> _LevelsDict:
        """Recursively processes text chunks through multiple levels of embedding and summarization.

        This method implements the core hierarchical analysis pipeline:
        1. Embeds chunks using the sentence transformer
        2. Clusters embeddings using two-phase global/local approach
        3. Generates summaries for each cluster
        4. Builds parent-child relationships between levels
        5. Updates the book tree with results

        Args:
            chunks (_LevelsDict): Dictionary of chunks organized by level
            chapter_num (int): Current chapter being processed
            book_tree (dict): Shared dictionary storing all book processing results
            book_filename (str): Name of the book file being processed
            number_of_levels (int, optional): Maximum levels to generate. Defaults to 3.
            level (int, optional): Current processing level. Defaults to 1.

        Returns:
            _LevelsDict: Updated dictionary containing chunks for all processed levels
                where each level contains a list of Chunk objects with:
                - Original or summary text
                - Embeddings
                - Parent-child relationships
                - Metadata

        Example:
            >>> chunks = {'level_1': [chunk1, chunk2, chunk3]}
            >>> results = processor._recursive_embedding_with_cluster_summarization(
            ...     chunks=chunks,
            ...     chapter_num=1,
            ...     book_tree=book_tree,
            ...     book_filename='book1.txt'
            ... )
            >>> print(results.keys())
            dict_keys(['level_1', 'level_2', 'level_3'])
            >>> print(len(results['level_2']))  # Number of summaries
            2
        """
        chapter_key = f'chapter_{chapter_num}'
        thread_prefix = f"{book_filename}_{chapter_key}"
        book_number = next(iter(chunks.values()))[0].metadata.book_number

        while level <= number_of_levels:
            level_keys = list(chunks.keys())
            target_level_key = f'level_{level}'
            if target_level_key in level_keys:
                chunk_list = chunks[target_level_key]
            else:
                raise ValueError(f"Level {level} not found in chunks for chapter {chapter_num} in book {'book_number'}")

            # Initialize next level list if needed
            if level + 1 <= number_of_levels:
                next_level: List[Chunk] = []
                # Initialize the next level in book_tree if it doesn't exist
                if f'level_{level + 1}' not in book_tree[chapter_key]:
                    book_tree[chapter_key][f'level_{level + 1}'] = []

            # Embed the chunks
            for chunk in chunk_list:
                if chunk.embedding is None:
                    chunk.embedding = self.chunker.model.encode(chunk.text)

            # Collect the embeddings   
            chapter_embeddings: np.array[np.ndarray] = np.array([chunk.embedding for chunk in chunk_list])

            # Cluster the embeddings
            chunk_cluster_map, num_clusters = self._clustering_algorithm(chapter_embeddings)
            #print(f"Finished clustering level {level} for book {book_number}, chapter {chapter_num}. Found {num_clusters} clusters.")
            
            # Collect the chunks that make up each cluster
            summary_chunks_to_build: List[List[str]] = [[] for _ in range(num_clusters)]
            """This is a list of lists, where the outer index is the cluster each list contains the chunk keys for that cluster."""
            for chunk_idx, cluster_labels in enumerate(chunk_cluster_map):
                for cluster_label in cluster_labels:
                    summary_chunks_to_build[int(cluster_label)].append(chunk_list[chunk_idx].chunk_key)

            if self.max_summary_threads > 1: # Use threads to process in parallel
                # Prepare cluster data
                cluster_tasks = [
                    (chunk_list, chunk_keys, cluster_idx, level) 
                    for cluster_idx, chunk_keys in enumerate(summary_chunks_to_build)
                ]
                results = [None] * len(cluster_tasks)
    
                # Process clusters in parallel
                with ThreadPoolExecutor(max_workers=min(len(cluster_tasks), self.max_summary_threads), thread_name_prefix=thread_prefix) as executor:
                    future_to_idx = {
                        executor.submit(self._process_cluster, task_data): i 
                        for i, task_data in enumerate(cluster_tasks)
                    }
                    
                    # Collect results in order
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        summary, cluster_metadata = future.result()
                        # Embed in main thread (CPU-bound)
                        summary_embedding = self.chunker.model.encode(summary)
                        results[idx] = (summary, cluster_metadata, summary_embedding)
                        
                    # Add summaries to next level
                    for summary, metadata, embedding in results:
                        next_level.append(Chunk(summary,
                                            metadata=metadata,
                                            embedding=embedding,
                                            is_summary=True))
            else: # Process sequentially

                for cluster_idx, chunk_keys in enumerate(summary_chunks_to_build):
                    # Set the metadata
                    cluster_metadata = {
                        "book_number": book_number,
                        "chapter_number": chapter_num,
                        "level": level + 1,
                        "chunk_index": cluster_idx
                    }
                    
                    # Collect and summarize texts
                    cluster_texts = [chunk.text for chunk in chunk_list if chunk.chunk_key in chunk_keys]
                    summary_text = "------\n------".join(cluster_texts)
                    #print(f"Summarizing cluster {cluster_metadata}...")
                    summary = self._generate_summary(summary_text)
                    next_level.append(Chunk(summary,
                                            metadata=cluster_metadata,
                                            embedding=self.chunker.model.encode(summary),
                                            is_summary=True))
                #print(f"Finished creating level {level + 1} clusters for book {book_number}, chapter {chapter_num}")
            
            # Update relationships and book_tree
            for curr_lvl_chunk in chunk_list:
                child_idx = curr_lvl_chunk.metadata.chunk_index
                parent_idx_list = chunk_cluster_map[child_idx]
                for parent_idx in parent_idx_list:
                    curr_lvl_chunk.parents.append(next_level[int(parent_idx)].chunk_key)
                    next_level[int(parent_idx)].children.append(curr_lvl_chunk.chunk_key)

            # Store both current and next level chunks
            book_tree[chapter_key][target_level_key] = chunk_list
            if level + 1 <= number_of_levels:
                book_tree[chapter_key][f'level_{level + 1}'] = next_level
                # Add next level to chunks dict for next iteration
                chunks[f'level_{level + 1}'] = next_level

            book_tree[chapter_key][target_level_key] = chunk_list
            book_tree[chapter_key][f'level_{level + 1}'] = next_level

            # Run for the next level
            if num_clusters > 1:
                chunks = self._recursive_embedding_with_cluster_summarization(chunks=chunks, 
                                                                            chapter_num=chapter_num,
                                                                            number_of_levels=number_of_levels, 
                                                                            level=level + 1,
                                                                            book_filename=book_filename,
                                                                            book_tree=book_tree)
            if level == number_of_levels:
                print(f"Finished processing {chapter_key} for book {book_tree['shared_tree_key']}")
            return chunks

    def _get_chunks_from_filepath(self, file_path: Union[str, List[str]], series_id: int = None) -> OrderedDict[str, list[list[Chunk]]]:
        """Loads and chunks text files into a hierarchical structure.

        Processes one or more text files matching the given path pattern,
        splitting each into chapters and then into semantic chunks.

        Args:
            file_path (str): Glob pattern matching text files to process
                (e.g., './books/*.txt')
            series_id (int, optional): ID of the series being processed. Defaults to self.series_id.

        Returns:
            OrderedDict[str, list[list[Chunk]]] : Hierarchical structure where:
                - Keys are book file paths
                - First list level contains chapters
                - Second list level contains chunks within each chapter
                - Each chunk is a Chunk object with text and metadata

        Raises:
            ValueError: If no text files are found at the given path

        Example:
            >>> processor = RaptorProcessor(config_path='config.yaml')
            >>> chunks = processor._get_chunks_from_filepath('./books/*.txt')
            >>> print(f"Found {len(chunks)} books")
            Found 2 books
            >>> first_book = next(iter(chunks.values()))
            >>> print(f"First book has {len(first_book)} chapters")
            First book has 12 chapters
        """
        series_id = series_id or self.series_id
        chunked_text: OrderedDict[str, list[list[Chunk]]] = {}
        input_text = self.chunker.read_text_files(file_path)
        if len(input_text) < 1:
            raise ValueError(f"No text found in file path {file_path}")
        for book_filename, book_info in sorted(input_text.items()):
            chunked_text[book_filename] = []
            for chapter_num, chapter_data in book_info.chapters.items():
                chunks = self.chunker.sentence_splitter(
                    chapter_data.full_text, 
                    chunk_size=self.chunk_size, 
                    chunk_overlap=self.chunk_overlap
                )
                chunk_list = []
                for idx, text in enumerate(chunks):
                    chunk_metadata = {
                        "series_id": series_id,
                        "book_number": book_info.book_number,
                        "chapter_number": chapter_num,
                        "level": 1,
                        "chunk_index": idx
                    }
                    chunk_list.append(Chunk(text, metadata=chunk_metadata))
                chunked_text[book_filename].append(chunk_list)
        return chunked_text

    def process_texts(self, file_path: Union[str, List[str]], 
                      series_id: int = None,
                      max_levels: int = None, 
                      max_processes: int = None, 
                      max_summary_threads: int = None
                     ) -> _RaptorResults:
        """Process text files into a multi-level hierarchical summary structure.

        This is the main entry point for text processing. It handles:
        1. Loading and chunking text files
        2. Parallel processing across chapters
        3. Multi-level summarization
        4. Building parent-child relationships

        Args:
            file_path (str): Glob pattern matching text files to process
            series_id (int, optional): ID of the series being processed. Defaults to self.series_id.
            max_levels (int, optional): Override default max hierarchy levels
            max_processes (int, optional): Override default process pool size
            max_summary_threads (int, optional): Override default summary thread count

        Returns:
            Dict[str, Dict[str, _LevelsDict]]: Hierarchical results structure:
                {
                    'book_file.txt': {
                        'chapter_1': {
                            'level_1': [original chunks],
                            'level_2': [summaries],
                            ...
                        },
                        'chapter_2': {...}
                    },
                    'book2_file.txt': {...}
                }

        Raises:
            ValueError: If no text files are found at the given path

        Example:
            >>> processor = RaptorProcessor('config.yaml')
            >>> results = processor.process_texts(
            ...     './books/*.txt',
            ...     max_levels=3,
            ...     max_processes=4,
            ...     max_summary_threads=2
            ... )
            >>> print(f"Processed {len(results)} books")
            Processed 2 books
        """
        
        series_id = series_id or self.series_id
        self.chunk_tree: _RaptorResults = {}

        # Overwrite default values if provided in this method call
        n_processes = max_processes or self.max_processes
        self.max_summary_threads = max_summary_threads or self.max_summary_threads
        self.max_levels = max_levels or self.max_levels
        
        if max_summary_threads is not None:
            self.max_summary_threads = max_summary_threads

        with Manager() as manager:
            chunked_text = self._get_chunks_from_filepath(file_path, series_id)
            if len(chunked_text) < 1:
                raise ValueError("No text found in the provided file path.")
            
            # Get level 1 chunks for each book into the tree
            for book_filename, book_info in chunked_text.items():
                book_tree = manager.dict()
                for chapter_num in range(len(book_info)):
                    book_tree[f'chapter_{chapter_num}'] = {}

                process_args = [
                    (chapter_data, chapter_num, book_filename, self.config_path, book_tree, self.max_levels)
                    for chapter_num, chapter_data in enumerate(book_info)
                ]
                
                with Pool(processes=n_processes, maxtasksperchild=5) as pool:
                    for book_tree in pool.imap(_process_chapter_worker, process_args):
                        self.chunk_tree[book_filename] = dict(book_tree)

        return self.chunk_tree

    def save_chunk_tree(self, output_path: str, compress: bool = True) -> None:
        """Save each book in the chunk tree to a separate JSON file, optionally compressed.

        The output filename will include the book number from the first chunk's metadata.
        Example: For output_path 'data.json' and book_number 1, creates 'data_1.json'
        Each file maintains the original book_filename as the top-level key for compatibility.

        Args:
            output_path (str): Base path where to save the files
            compress (bool, optional): Whether to use gzip compression. Defaults to True.
                                    If True, '.gz' extension will be added if not present.
        """
        import json
        
        def chunk_tree_to_json(book_filename: str, book_data: Dict[str, _LevelsDict]) -> dict:
            """Convert a single book's data to a JSON-serializable dictionary."""
            return {
                book_filename: {
                    chapter_key: {
                        level_key: [chunk.to_json() for chunk in level_data]
                        for level_key, level_data in chapter_data.items()
                    }
                    for chapter_key, chapter_data in book_data.items()
                }
            }

        if self.chunk_tree is None:
            raise ValueError("No chunk tree to save. Run process_texts() first.")

        # Handle base path
        path = pathlib.Path(output_path)
        base_stem = path.stem
        base_suffix = '.gz' if compress else ''
        
        # Process each book separately
        for book_filename, book_data in self.chunk_tree.items():
            # Get book number from first chunk of first chapter
            first_chapter = next(iter(book_data.values()))
            first_level = first_chapter.get('level_1', [])
            if not first_level:
                print(f"Warning: No chunks found for {book_filename}, skipping...")
                continue
                
            book_number = first_level[0].metadata.book_number
            
            # Construct output filename with book number
            output_filename = f"{base_stem}_{book_number}{path.suffix}{base_suffix}"
            output_filepath = str(path.parent / output_filename)
            
            # Convert book data to JSON, maintaining book_filename as top key
            json_data = json.dumps(
                chunk_tree_to_json(book_filename, book_data),
                indent=None if compress else 2,
                separators=(',', ':') if compress else (', ', ': '),
                ensure_ascii=False
            )
            
            # Save the file
            if compress:
                with gzip.open(output_filepath, 'wt', encoding='utf-8', compresslevel=9) as f:
                    f.write(json_data)
            else:
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    f.write(json_data)

    @staticmethod
    def load_chunk_tree(input_path: str) -> _RaptorResults:
        """Load a chunk tree from a JSON file, automatically handling compression.

        Args:
            input_path (str): Path to the JSON file (can be .gz or uncompressed)
        """
        import json

        def json_to_chunk_tree(json_tree: dict) -> _RaptorResults:
            """Convert a JSON dictionary back to a chunk tree."""
            chunk_tree = {}
            for book_filename, book_data in json_tree.items():
                chunk_tree[book_filename] = {}
                for chapter_key, chapter_data in book_data.items():
                    chunk_tree[book_filename][chapter_key] = {}
                    for level_key, level_data in chapter_data.items():
                        chunk_tree[book_filename][chapter_key][level_key] = [
                            Chunk(
                                text=chunk_data["text"],
                                metadata=chunk_data["metadata"],
                                is_summary=chunk_data["is_summary"],
                                embedding=np.array(chunk_data["embedding"]) if chunk_data["embedding"] is not None else None
                            ) for chunk_data in level_data
                        ]
                        # Restore relationships
                        for chunk_idx, chunk_data in enumerate(level_data):
                            curr_chunk = chunk_tree[book_filename][chapter_key][level_key][chunk_idx]
                            curr_chunk.parents = chunk_data["parents"]
                            curr_chunk.children = chunk_data["children"]
            return chunk_tree

        # Auto-detect compression
        path = pathlib.Path(input_path)
        is_compressed = path.suffix == '.gz'

        if is_compressed:
            with gzip.open(input_path, 'rt', encoding='utf-8') as f:
                return json_to_chunk_tree(json.load(f))
        else:
            with open(input_path, 'r', encoding='utf-8') as f:
                return json_to_chunk_tree(json.load(f))