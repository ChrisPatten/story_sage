"""Text analysis and hierarchical clustering module for story processing.

This module implements a hierarchical text analysis system that processes books
at multiple levels (chapter, book, and cross-book) using clustering and summarization.
It uses UMAP for dimensionality reduction and Gaussian Mixture Models for clustering.

Key Features:
    - Multi-level text analysis (chapter, book, series)
    - Intelligent chunking with configurable overlap
    - Hierarchical clustering with both global and local analysis
    - Automatic cluster number optimization
    - GPT-4 powered summaries for each cluster
    - Progress tracking with tqdm

Example usage:
    # Initialize with configuration
    config_path = "config.yaml"
    raptor = Raptor(config_path)
    
    # Process a book or series of books
    results = raptor.process_texts(
        "path/to/books/",
        number_of_levels=3,
        target_dim=10
    )
    
    # Example accessing results
    for book_path, book_results in results.items():
        # Access chapter-level analysis
        chapter_clusters = book_results["chapter_1"]["level_1_clusters"]
        chapter_summaries = book_results["chapter_1"]["level_1_summaries"]
        
        # Access book-level aggregation
        book_clusters = book_results["book_aggregate"]["level_1_clusters"]
        
        # Access cross-book analysis
        series_clusters = book_results["books_up_to_current"]["level_1_clusters"]

Returns:
    The process_texts method returns a nested dictionary structure:
    {
        'book_filepath.txt': {
            'chapter_1': {
                'level_1_clusters': DataFrame(
                    columns=['texts', 'embedding', 'clusters', 'book_number', 'chapter_number']
                    # Example row: {'texts': 'Chapter text...', 'embedding': [...], 'clusters': [0, 1], 
                    #              'book_number': 1, 'chapter_number': 1}
                ),
                'level_1_summaries': DataFrame(
                    columns=['summaries', 'level', 'clusters', 'book_number', 'chapter_number']
                    # Example row: {'summaries': 'Summary text...', 'level': 1, 'clusters': 0,
                    #              'book_number': 1, 'chapter_number': 1}
                ),
                'level_2_clusters': DataFrame(...),
                'level_2_summaries': DataFrame(...),
                ...
            },
            'book_aggregate': {
                'level_1_clusters': DataFrame(
                    columns=['texts', 'embedding', 'clusters', 'book_number']
                    # Example row: {'texts': 'Book text...', 'embedding': [...], 'clusters': [0, 2],
                    #              'book_number': 1}
                ),
                'level_1_summaries': DataFrame(...),
                ...
            },
            'books_up_to_current': {
                'level_1_clusters': DataFrame(
                    columns=['texts', 'embedding', 'clusters', 'max_book_number']
                    # Example row: {'texts': 'Series text...', 'embedding': [...], 'clusters': [1, 3],
                    #              'max_book_number': 2}
                ),
                'level_1_summaries': DataFrame(...),
                ...
            }
        },
        ...
    }

Dependencies:
    - StorySageConfig: Configuration management
    - StorySageChunker: Text chunking utilities
    - OpenAI: Completion API for summarization
    - UMAP: Dimensionality reduction
    - sklearn: Gaussian Mixture Models
    - pandas: Data management
    - numpy: Numerical operations
"""

from .chunker import StorySageChunker
from ..models import StorySageConfig
from openai import OpenAI
import httpx
import os
import numpy as np
import umap.umap_ as umap
from typing import List, Tuple, Dict
from sklearn.mixture import GaussianMixture
import pandas as pd
from tqdm.notebook import tqdm

class ChunkedText:
    """Container for chunked text data.
    
    Attributes:
        text (str): The original complete text.
        chunks (list[str]): List of text chunks after splitting.
    """
    def __init__(self, chunks: list[str]):
        self.text = ""
        self.chunks = chunks

class Raptor:
    """Main class for text processing and analysis using hierarchical clustering.
    
    This class implements a multi-level text analysis pipeline that:
    1. Chunks input text into manageable segments
    2. Generates embeddings for text chunks
    3. Performs dimensionality reduction using UMAP
    4. Clusters the reduced embeddings using Gaussian Mixture Models
    5. Generates summaries for each cluster using GPT-4
    
    Attributes:
        seed (int): Random seed for reproducibility
        config (StorySageConfig): Configuration object containing API keys and settings
        chunker (StorySageChunker): Text chunking utility
        client (OpenAI): OpenAI API client instance
        chunk_size (int): Size of text chunks in characters
        chunk_overlap (int): Overlap between consecutive chunks in characters
        clustering_threshold (float): Similarity threshold for cluster assignment
    """

    def __init__(self, 
                 config_path: str, 
                 seed: int = 8675309, 
                 model_name: str = "gpt-4o-mini", 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 50,
                 clustering_threshold: float = 0.5):
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

    def _dimensionality_reduction(self,
                                  embeddings: np.ndarray,
                                  target_dim: int,
                                  clustering_type: str,
                                  metric: str = 'cosine') -> np.ndarray:
        """
        Reduces the dimensionality of embeddings using UMAP.

        Args:
            embeddings (np.ndarray): The input embeddings to reduce.
            target_dim (int): The target number of dimensions.
            clustering_type (str): Type of clustering ('local' or 'global').
            metric (str, optional): The metric to use for UMAP. Defaults to "cosine".

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
            metric=metric,
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
        max_clusters: int = 50,
        random_state: int = None
    ) -> int:
        """
        Determines the optimal number of clusters using inertia and BIC scores.

        Args:
            embeddings (np.ndarray): The input embeddings.
            max_clusters (int, optional): Maximum number of clusters to consider. Defaults to 50.
            random_state (int, optional): Random state for reproducibility. Defaults to SEED.

        Returns:
            int: The optimal number of clusters.
        """
        random_state = self.seed if random_state is None else random_state
        max_clusters = min(max_clusters, len(embeddings))
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
        gm = GaussianMixture(n_components=n_clusters, random_state=random_state, n_init=2)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > self.clustering_threshold)[0] for prob in probs] 
        return labels, n_clusters  

    def _clustering_algorithm(
        self,
        embeddings: np.ndarray,
        target_dim: int,
        random_state: int = None
    ) -> Tuple[List[np.ndarray], int]:
        """
        Clustering algorithm that performs global and local clustering.

        Args:
            embeddings (np.ndarray): The input embeddings.
            target_dim (int): Target number of dimensions for reduction.
            random_state (int, optional): Random state for reproducibility. Defaults to SEED.

        Returns:
            Tuple[List[np.ndarray], int]: A list of local cluster labels and the total number of clusters.
        """
        random_state = self.seed if random_state is None else random_state
        if len(embeddings) <= target_dim + 1:
            return [np.array([0]) for _ in range(len(embeddings))], 1
        
        # Global clustering: uses a 'global' dimension reduction and lumps embeddings into broad groups.
        reduced_global_embeddings = self._dimensionality_reduction(embeddings, target_dim, "global")
        global_clusters, n_global_clusters = self._gmm_clustering(reduced_global_embeddings, random_state=random_state)

        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        # Local clustering within each global cluster: uses 'local' dimension reduction.
        for i in range(n_global_clusters):
            global_cluster_mask = np.array([i in gc for gc in global_clusters])
            global_cluster_embeddings = embeddings[global_cluster_mask]

            if len(global_cluster_embeddings) <= target_dim + 1:
                # Assign all points in this global cluster to a single local cluster
                for idx in np.where(global_cluster_mask)[0]:
                    all_local_clusters[idx] = np.append(all_local_clusters[idx], total_clusters)
                total_clusters += 1
                continue

            try:
                reduced_local_embeddings = self._dimensionality_reduction(global_cluster_embeddings, target_dim, "local")
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
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.7
        )
        summary = response.choices[0].message.content.strip()
        return summary
    
    def _embed_clusters(
        self,
        chunk_key_base: str,
        texts: List[str],
        target_dim: int = 10
    ) -> pd.DataFrame:
        """
        Embeds texts into clusters and returns a DataFrame with the results.

        Args:
            texts (List[str]): The list of texts to embed and cluster.
            target_dim (int, optional): Target number of dimensions for reduction. Defaults to 10.

        Returns:
            pd.DataFrame: DataFrame containing texts, their embeddings, and cluster assignments.
        """
        # chunk_key_base book_<book_number>|level_<level> for book summary, book_<book_number>|chapter_<chapter_number>|level_<level> for chapter summary
        textual_embeddings = np.array(self.chunker.model.encode(texts))
        clusters, number_of_clusters = self._clustering_algorithm(textual_embeddings, target_dim)
        cluster_ids = []
        for cluster in clusters:
            cluster_ids.append([f'cluster_key|{chunk_key_base}|{int(cluster_idx)}' for cluster_idx in cluster.tolist()])
        return pd.DataFrame({
            "texts": texts,
            "embedding": list(textual_embeddings),
            "clusters": clusters,
            "cluster_ids": cluster_ids
        })


    def _embed_cluster_summaries(
        self,
        texts: List[str],
        level: int,
        chunk_key_base: str,
        target_dim: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Embeds texts, clusters them, and generates summaries for each cluster.

        Args:
            texts (List[str]): The list of texts to process.
            level (int): The current level of recursion.
            target_dim (int, optional): Target number of dimensions for reduction. Defaults to 10.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing cluster assignments and their summaries.
        """
        # chunk_key_base book_<book_number> for book summary, book_<book_number>|chapter_<chapter_number> for chapter summary

        chunk_key_base += f"|level_{level}"
        df_clusters = self._embed_clusters(chunk_key_base, texts, target_dim)
        # cluster keys are now assigned
        main_list = []
        
        for idx, row in df_clusters.iterrows():
            for c_idx, cluster in enumerate(row["clusters"]):
                main_list.append({
                    "text": row["texts"],
                    "embedding": row["embedding"],
                    "clusters": cluster,
                    "cluster_ids": row["cluster_ids"][c_idx],
                    "chunk_ids": f'chunk_key|{chunk_key_base}|{idx}'
                })
        
        main_df = pd.DataFrame(main_list)
        unique_clusters = main_df["clusters"].unique()
        unique_cluster_ids = main_df["cluster_ids"].unique()
        if len(unique_clusters) == 0:
            return df_clusters, pd.DataFrame(columns=["summaries", "level", "clusters", "cluster_ids", "chunk_ids"])

        summaries = []
        for cluster in unique_clusters:
            text_in_df = main_df[main_df["clusters"] == cluster]
            unique_texts = text_in_df["text"].tolist()
            text = "------\n------".join(unique_texts)
            summary = text #self._generate_summary(text)
            summaries.append(summary)
        
        summary_chunk_ids = [f'chunk_key|{chunk_key_base}|{int(cluster_idx)}' for cluster_idx in unique_clusters]

        df_summaries = pd.DataFrame({
            "summaries": summaries,
            "level": [level] * len(summaries),
            "clusters": unique_clusters,
            "cluster_ids": unique_cluster_ids,
            "chunk_ids": summary_chunk_ids
        })

        return df_clusters, df_summaries

    def _recursive_embedding_with_cluster_summarization(
        self,
        texts: List[str],
        chunk_key_base: str,
        number_of_levels: int = 3,
        level: int = 1,
        target_dim: int = 10
    ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Recursively embeds texts and generates cluster summaries up to a specified number of levels.

        Args:
            texts (List[str]): The list of texts to process.
            number_of_levels (int, optional): The maximum number of recursion levels. Defaults to 3.
            level (int, optional): The current recursion level. Defaults to 1.
            target_dim (int, optional): Target number of dimensions for reduction. Defaults to 10.

        Returns:
            Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]: A dictionary mapping levels to their cluster and summary DataFrames.
        """
        # chunk_key_base book_<book_number> for book summary, book_<book_number>|chapter_<chapter_number> for chapter summary
        if level > number_of_levels:
            return {}
        results = {}
        df_clusters, df_summaries = self._embed_cluster_summaries(texts, level, chunk_key_base, target_dim)
        results[level] = (df_clusters, df_summaries)
        
        if df_summaries.empty or len(df_summaries['clusters'].unique()) == 1:
            #print(f"No more unique clusters found at level {level}. Stopping recursion.")
            return results
        
        if level < number_of_levels:
            next_level_texts = df_summaries['summaries'].tolist()
            next_level_results = self._recursive_embedding_with_cluster_summarization(
                next_level_texts, 
                chunk_key_base,
                number_of_levels, 
                level + 1,
                target_dim
            )
            results.update(next_level_results)
        
        return results
    
    def _process_text_hierarchy(
        self,
        texts: List[str], 
        number_of_levels: int = 3,
        target_dim: int = 10,
        metadata: Dict[str, int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Processes a hierarchy of texts by embedding, clustering, and summarizing across multiple levels.

        Args:
            texts (List[str]): The list of texts to process.
            number_of_levels (int, optional): The number of hierarchical levels. Defaults to 3.
            target_dim (int, optional): Target number of dimensions for reduction. Defaults to 10.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary of DataFrames containing cluster and summary information for each level.
        """
        if metadata and 'chunk_key' in metadata:
            chunk_key_base = metadata.pop('chunk_key')
            # chunk_key_base book_<book_number> for book summary, book_<book_number>|chapter_<chapter_number> for chapter summary

        hierarchy_results = self._recursive_embedding_with_cluster_summarization(
            texts, chunk_key_base, number_of_levels, target_dim=target_dim
        )
        
        processed_results = {}
        for level, (df_clusters, df_summaries) in hierarchy_results.items():
            if df_clusters.empty or df_summaries.empty:
                continue
            # Store metadata in dataframes
            if metadata:
                for meta_key, meta_val in metadata.items():
                    df_clusters[meta_key] = meta_val
                    df_summaries[meta_key] = meta_val

            processed_results[f"level_{level}_clusters"] = df_clusters
            processed_results[f"level_{level}_summaries"] = df_summaries
        
        return processed_results

    def _get_chunks_from_filepath(self, file_path: str) -> dict[str, dict[int, ChunkedText]]:
        chunked_text: dict[str, dict[int, ChunkedText]] = {}
        input_text = self.chunker.read_text_files(file_path)
        if len(input_text) < 1:
            raise ValueError("No text found in the provided file path.")
        for book_filepath, book_info in input_text.items():
            chunked_text[book_filepath] = {}
            for chapter_num, chapter_data in book_info.chapters.items():
                chunks = self.chunker.sentence_splitter(
                    chapter_data.full_text, 
                    chunk_size=self.chunk_size, 
                    chunk_overlap=self.chunk_overlap
                )
                chunked_text[book_filepath][chapter_num] = ChunkedText(chunks=chunks)
        return chunked_text

    def process_texts(self,
                      file_path: str,
                      number_of_levels: int = 3,
                      target_dim: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Processes a hierarchy of texts from a file path by embedding, clustering, and summarizing across multiple levels.

        Args:
            file_path (str): The path to the file containing the texts to process.
            number_of_levels (int, optional): The number of hierarchical levels. Defaults to 3.
            target_dim (int, optional): Target number of dimensions for reduction. Defaults to 10.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary of DataFrames containing cluster and summary information for each level.
        """
        chunked_text = self._get_chunks_from_filepath(file_path)
        if not chunked_text:
            raise ValueError("No text found in the provided file path.")
        processed_results = {}
        book_filepaths = sorted(chunked_text.keys())
        all_books_chunks_upto_current = []

        for i, book_filepath in enumerate(book_filepaths, start=1):
            book_info = chunked_text[book_filepath]
            book_chapter_results = {}

            # Process text hierarchy for each chapter and add metadata
            for j, (chapter_key, chapter) in enumerate(book_info.items(), start=1):
                chapter_result = self._process_text_hierarchy(
                    chapter.chunks,
                    number_of_levels,
                    target_dim,
                    metadata={"book_number": i, "chapter_number": j, "chunk_key": f'book_{i}|chapter_{j}'}
                )
                book_chapter_results[f"chapter_{chapter_key}"] = chapter_result

            # Process all chunks for the current book
            all_book_chunks = [chunk for chapter in book_info.values() for chunk in chapter.chunks]
            print(f'Processing all chunks for book {book_filepath}')
            book_result = self._process_text_hierarchy(
                all_book_chunks,
                number_of_levels,
                target_dim,
                metadata={"book_number": i, "chunk_key": f'book_{i}'}
            )
            book_chapter_results["book_aggregate"] = book_result

            # If there's more than one book, also process all chunks up to the current book
            
            if i > 1:
                all_books_chunks_upto_current.extend(all_book_chunks)
                print(f'Processing all chunks up to book {book_filepath}')
                aggregated_books_result = self._process_text_hierarchy(
                    all_books_chunks_upto_current,
                    number_of_levels,
                    target_dim,
                    metadata={"max_book_number": i, "chunk_key": f'max_{i}'}
                )
                book_chapter_results["books_up_to_current"] = aggregated_books_result

            processed_results[book_filepath] = book_chapter_results

        return processed_results