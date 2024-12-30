"""
Entity extraction and grouping module for StorySage.

This module provides functionality for extracting named entities from text documents
and grouping them based on similarity. It uses the GLiNER model for entity extraction
and TF-IDF vectorization with cosine similarity for entity grouping.
"""

from tqdm import tqdm
from gliner import GLiNER
from gliner.model import GLiNER
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from scipy.sparse import spmatrix, vstack, csr_matrix
import numpy as np
import re
from langchain_core.documents.base import Document
from typing import List, Dict, Set, Tuple
from story_sage.data_classes.story_sage_series import StorySageSeries
from story_sage.story_sage_entity import StorySageEntityCollection, StorySageEntityGroup, StorySageEntity, GroupType
import json




class StorySageEntityExtractor():
    """A class for extracting and grouping named entities from text documents.

    This class uses the GLiNER model to extract named entities and groups them
    based on TF-IDF vector similarity. It provides methods for cleaning entity
    strings, extracting entities from text, and grouping similar entities together.

    Attributes:
        model (GLiNER): The loaded GLiNER model instance
        target_series_info (StorySageSeries): Information about the target series
        entities (List[GroupType]): List of entity groups with their vectors
        entity_collection (StorySageEntityCollection): Collection of grouped entities
        similarity_threshold (float): Threshold for grouping similar entities
        chunk_max_length (int): Maximum text length to process at once
    """

    def __init__(self, series: Dict,
                 model_name: str = 'urchade/gliner_base', 
                 device: str = 'cpu',
                 similarity_threshold: float = 0.7,
                 chunk_max_length: int = 350,
                 existing_collection: StorySageEntityCollection = None):
        """
        Initializes the StorySageEntityExtractor with the given parameters.

        This constructor initializes the entity extractor by loading a pre-trained GLiNER model,
        setting the device for model computation, and configuring the series information and
        extraction parameters.

        Args:
            series (Dict): A dictionary containing series data to be used for entity extraction.
            model_name (str, optional): The name of the pre-trained GLiNER model to use. Defaults to 'urchade/gliner_base'.
            device (str, optional): The device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
            similarity_threshold (float, optional): The cosine similarity threshold for grouping entities. Defaults to 0.7.
            chunk_max_length (int, optional): The maximum length of text chunks to process at once. Defaults to 350.
            existing_collection (StorySageEntityCollection, optional): Existing entity collection to extend. Defaults to None.

        Example:
            >>> series_data = {
            ...     "series_id": 1,
            ...     "series_name": "Example Series",
            ...     "series_metadata_name": "example_series",
            ...     "entity_settings": {
            ...         "names_to_skip": ["example"],
            ...         "person_titles": ["Dr", "Mr", "Mrs"]
            ...     },
            ...     "books": [
            ...         {
            ...             "number_in_series": 1,
            ...             "title": "First Book",
            ...             "book_metadata_name": "first_book",
            ...             "number_of_chapters": 10
            ...         }
            ...     ]
            ... }
            >>> extractor = StorySageEntityExtractor(series_data, model_name='urchade/gliner_base', device='cpu')
        """

        self.model = GLiNER.from_pretrained(model_name, model_kwargs={'model_max_length': 512, 'use_fast': False})
        self.model.to(torch.device(device))
        self.target_series_info = StorySageSeries.from_dict(series)
        self.entities: List[GroupType] = []
        self.entity_collection = existing_collection
        self.similarity_threshold = similarity_threshold
        self.chunk_max_length = chunk_max_length
        self.vectorizer = None


    def _clean_string(self, text: str, 
                      names_to_skip: list[str] = [], 
                      person_titles: list[str] = [],
                      min_len: int = 3,
                      max_len: int = 30) -> str:
        """
        Cleans a given string by removing specified titles, unwanted characters, and skipping certain names.

        This function processes a given text string by removing titles, unwanted characters, and skipping
        names that are either too short, too long, or included in a list of names to skip. It also ensures
        that the cleaned text does not start with 'of ' and is not in the list of person titles.

        Args:
            text (str): The text string to be cleaned.
            names_to_skip (list[str], optional): A list of names to skip during cleaning. Defaults to an empty list.
            person_titles (list[str], optional): A list of person titles to remove from the text. Defaults to an empty list.
            min_len (int, optional): The minimum length of the text to be considered valid. Defaults to 3.
            max_len (int, optional): The maximum length of the text to be considered valid. Defaults to 30.

        Returns:
            str: The cleaned text string, or None if the text is invalid or should be skipped.

        Example:
            >>> text = "Dr. John Doe"
            >>> names_to_skip = ["john"]
            >>> person_titles = ["Dr", "Mr", "Mrs"]
            >>> cleaned_text = extractor._clean_string(text, names_to_skip, person_titles)
            >>> print(cleaned_text)
            'doe'
        """

        if len(names_to_skip) == 0:
            names_to_skip = self.target_series_info.entity_settings.names_to_skip
        
        if len(person_titles) == 0:
            person_titles = self.target_series_info.entity_settings.person_titles
        
        # Add common pronouns to names_to_skip
        default_skip = ['he', 'she', 'they', 'we', 'you', 'i', 'it', 'us', 'her', 'him',
                        'child', 'sister', 'woman', 'women', 'man', 'men', 'people', 
                        'boy', 'boys', 'girl', 'girls', 'someone', 'one', 'children', 
                        'friend', 'husband', 'husbands', 'father', 'mother']
        words_to_strip = ['a', 'an', 'the', 'of']

        names_to_skip = names_to_skip + default_skip if names_to_skip else default_skip
        
        titles_pattern = r'^(?:-?)(?:' + '|'.join(person_titles) + r'\s+)|(?:\s+' + \
                        '|'.join(person_titles) + r')$'
        
        words_to_strip_pattern = r'^\s?(?:' + '|'.join(words_to_strip) + r')\s+'

        starts_to_remove_pattern = r'^[a-z]\s|-\w\s?'
        
        allowed_characters_pattern = r'[^a-z\s-]'
        
        # Remove titles using regex
        cleaned_text = re.sub(allowed_characters_pattern, '', text.strip().lower(), flags=re.IGNORECASE)
        cleaned_text = re.sub(titles_pattern, '', cleaned_text.strip(), flags=re.IGNORECASE|re.MULTILINE)
        cleaned_text = re.sub(titles_pattern, '', cleaned_text.strip(), flags=re.IGNORECASE|re.MULTILINE) # Do this twice to catch multiple titles for a person
        cleaned_text = re.sub(words_to_strip_pattern, '', cleaned_text.strip(), flags=re.IGNORECASE|re.MULTILINE)
        cleaned_text = re.sub(starts_to_remove_pattern, '', cleaned_text.strip(), flags=re.IGNORECASE|re.MULTILINE)
        cleaned_text = cleaned_text.strip()
        
        if len(cleaned_text) > min_len and len(cleaned_text) < max_len:
            if cleaned_text.startswith('of '):
                return None
            if cleaned_text in (names_to_skip + person_titles):
                return None
            if cleaned_text not in names_to_skip:
                return cleaned_text
        else:
            return None
        
    
    def _add_new_string_to_groups(self, new_string: str):
        """Adds a new string to existing entity groups or creates a new group.

        This method vectorizes the input string and either adds it to an existing
        group if the similarity threshold is met, or creates a new group. It uses
        TF-IDF vectorization and cosine similarity for comparison.

        Args:
            new_string (str): The string to be added to entity groups.

        Example:
            >>> extractor = StorySageEntityExtractor(series_data)
            >>> extractor._add_new_string_to_groups("John Smith")
        """

        new_vector = self.vectorizer.transform([new_string])

        if len(self.entities) == 0:
            self.entities.append((new_vector, new_vector, {new_string}))
            return

        if any(new_string in strings for _, _, strings in self.entities):
            return
        
        similarities = [
            cosine_similarity(new_vector, centroid).item()
            for centroid, _, _ in self.entities
        ]
        max_similarity_idx = np.argmax(similarities)

        if similarities[max_similarity_idx] >= self.similarity_threshold:
            _, vectors, strings = self.entities[max_similarity_idx]
            strings.add(new_string)
            new_vectors = vstack([vectors, new_vector])
            new_centroid = csr_matrix(new_vectors.mean(axis=0))
            self.entities[max_similarity_idx] = (new_centroid, new_vectors, strings)
        else:
            self.entities.append((new_vector, new_vector, {new_string}))

        return

    def _extract_entities(self, documents: list[Document], labels: list[str]) -> dict[str, list[str]]:
        """Extracts named entities from documents using the GLiNER model.

        Processes text documents to identify and extract named entities matching the
        specified labels. Handles long documents by splitting them into manageable chunks.

        Args:
            documents (list[Document]): List of documents to process
            labels (list[str]): Entity types to extract (e.g., ['CHARACTER', 'LOCATION'])

        Returns:
            dict[str, list[str]]: Dictionary mapping entity labels to lists of extracted entities

        Example:
            >>> docs = [Document(page_content="John visited Paris with Mary")]
            >>> labels = ['CHARACTER']
            >>> entities = extractor._extract_entities(docs, labels)
            >>> print(entities)
            {'CHARACTER': ['John', 'Mary']}
        """
        
        entities_dict = {label: [] for label in labels}
        
        for chunk in tqdm(documents, desc='Extracting entities from chunks'):
            text = chunk.page_content
            if len(text) <= self.chunk_max_length:
                entities = self.model.predict_entities(text, labels)
                for entity in entities:
                    entities_dict[entity["label"]].append(entity["text"])
            else:
                text_parts = text.split()
                current_part = []
                current_length = 0

                for word in text_parts:
                    if current_length + len(word) + 1 <= self.chunk_max_length:
                        current_part.append(word)
                        current_length += len(word) + 1
                    else:
                        entities = self.model.predict_entities(' '.join(current_part), labels)
                        for entity in entities:
                            entities_dict[entity["label"]].append(entity["text"])
                        current_part = [word]
                        current_length = len(word)

                if current_part:
                    entities = self.model.predict_entities(' '.join(current_part), labels)
                    for entity in entities:
                        entities_dict[entity["label"]].append(entity["text"])
        
        return entities_dict
    
    def _get_dbscan_clusters(self, vectors: spmatrix, strings: List[str], eps: float = 0.5, min_samples: int = 2) -> StorySageEntityCollection:
        """Groups strings using DBSCAN clustering based on cosine similarity.

        This method uses DBSCAN clustering to group strings based on cosine similarity
        and returns the resulting clusters as a StorySageEntityCollection.

        Args:
            strings (List[str]): List of strings to cluster
            eps (float, optional): Maximum distance between samples for clustering. Defaults to 0.5.
            min_samples (int, optional): Minimum number of samples for a cluster. Defaults to 2.

        Returns:
            StorySageEntityCollection: Collection of grouped entities

        Example:
            >>> strings = ["John", "Johnny", "Paris", "London"]
            >>> clusters = extractor._get_dbscan_clusters(strings)
            >>> print(len(clusters.entity_groups))
            2
        """

        # Vectorize the input strings

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)

        vectors = normalize(vectors, norm='l2', axis=1)

        clustering = dbscan.fit(vectors)
        labels = clustering.labels_


        # Create a dictionary to store the clusters
        clusters = {}
        for entity, label in zip(strings, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(entity)

        # Remove the noise cluster
        noise_cluster: StorySageEntityGroup = StorySageEntityGroup(entities=[])
        if -1 in clusters:
            for entity in clusters[-1]:
                noise_cluster.add_entity(StorySageEntity(entity))
            del clusters[-1]

        clusters_list = [clusters[key] for key in clusters.keys()]

        return StorySageEntityCollection.from_sets(clusters_list), noise_cluster

    def _merge_single_entities(self, entities_collection: StorySageEntityCollection, verbose: bool = False) -> StorySageEntityCollection:
        if verbose:
            print('---------------------------------')
            print('Merging single-entity groups with multi-entity groups')
        single_member_groups: list[StorySageEntityGroup] = [group for group in entities_collection if len(group) == 1]
        multi_member_groups: list[StorySageEntityGroup] = [group for group in entities_collection if len(group) > 1]
        multi_member_group_words = [word for group in multi_member_groups for entity in group for word in entity.entity_name.split()]

        merged_single_entities = 0
        single_entities_to_merge = 0
        unmerged_single_entities = 0
        for single_group in tqdm(single_member_groups, desc='Merging single-entity groups'):
            single_entity = single_group.entities[0]  # Get the only entity in the group
            if single_entity.entity_name not in multi_member_group_words:
                unmerged_single_entities += 1
                entities_collection.remove_entity_group(single_group.entity_group_id)
                continue
            else:
                single_entities_to_merge += 1
                
            # Look for matches in other groups
            merge_found = False
            for other_group in multi_member_groups:
                # Check if single entity matches any word in any entity in other group
                for other_entity in other_group:
                    other_entity_words = other_entity.entity_name.split()
                    if single_entity.entity_name in other_entity_words:
                        # Merge the two groups
                        other_group: StorySageEntityGroup = StorySageEntityCollection.merge_groups(other_group, single_group)
                        merge_found = True
                        merged_single_entities += 1
                        break
                if merge_found:
                    break
                # Remove the old single group
                if single_group in entities_collection:
                    entities_collection.remove_entity_group(single_group.entity_group_id)
        if verbose:
            print(f'Found {single_entities_to_merge} entities to merge')
            print(f'Merged {merged_single_entities} single-entity groups')
            print(f'Removed {unmerged_single_entities} unmergable single-entity groups')

        return entities_collection
    
    def _get_multipass_cluster(self, vectors: spmatrix, strings: list[str], eps_first: float=0.5,
                               eps_second: float=0.1, min_samples: int = 2, 
                               merge_single_entities: bool = True, verbose: bool = False) -> StorySageEntityCollection:
        
        collected_entities, noise_cluster = self._get_dbscan_clusters(vectors=vectors, strings=strings, eps=eps_first, min_samples=min_samples)
        if len(collected_entities) < 1:
            raise AttributeError(f'No entities were grouped for eps_first {eps_first}, min_samples {min_samples}')
        max_group_size = max(len(group) for group in collected_entities)

        if verbose:
            print(f"Total entities: {len(strings)}")
            print(f"Number of groups (first clustering): {len(collected_entities)}")
            print(f"Largest group size (first clustering): {max_group_size} ({max_group_size / len(strings):.2%})")
            print(f"Noise cluster size: {len(noise_cluster)}")

        # Recluster noise_cluster
        if noise_cluster and len(noise_cluster) > 1:
            for entity in noise_cluster:
                if type(entity) == str:
                    print(f'String entity: {entity}')
            entity_strings = [entity.entity_name for entity in noise_cluster]
            vectors_sub = self.vectorizer.fit_transform(entity_strings)
            sub_clusters, _ = self._get_dbscan_clusters(vectors=vectors_sub, strings=entity_strings, eps=eps_second, min_samples=1)
            for sub_cluster in sub_clusters:
                collected_entities.add_entity_group(sub_cluster)
            if verbose:
                print('---------------------------------')
                print(f"Reclustered noise cluster with eps {eps_second}: {len(sub_clusters)}")
                print(f'Multi-entity groups created: {sum([1 for group in sub_clusters if len(group) > 1])}')
                print(f'Single-entity groups created: {sum([1 for group in sub_clusters if len(group) == 1])}')

        if merge_single_entities:
            collected_entities = self._merge_single_entities(collected_entities)
            
            # Compress multi-member groups by removing multi-word entities that contain single-word entities
            multi_member_groups: list[StorySageEntityGroup] = []
            for group in collected_entities:
                if len(group) > 1:
                    multi_member_groups.append(group)
            #multi_member_groups: list[StorySageEntityGroup] = [group for group in collected_entities if len(group) > 1]
            for group in multi_member_groups:
                # Get single word entities in this group
                single_word_entities: list[StorySageEntity] = [entity for entity in group if len(entity.entity_name.split()) == 1]
                
                # For each single word entity
                for single_entity in single_word_entities:
                    # Get all multi-word entities in same group that contain this word
                    to_remove = [entity for entity in group 
                                if len(entity.entity_name.split()) > 1 
                                and single_entity.entity_name in entity.entity_name.split()]
                    
                    # Remove matching entities from group
                    for entity in to_remove:
                        group.remove_entity_by_id(entity.entity_id)

        else:
            print('---------------------------------')

        # Analyze statistics about collected entities
        total_groups = len(collected_entities)
        group_sizes = [len(group) for group in collected_entities]
        avg_group_size = sum(group_sizes) / len(group_sizes)
        median_group_size = np.median(group_sizes)
        max_group_size = max(group_sizes)
        min_group_size = min(group_sizes)
        single_entity_groups = sum(1 for size in group_sizes if size == 1)
        max_group = max(collected_entities, key=lambda x: len(x))
        if verbose:
            print(f"Total remaining groups: {total_groups}")
            print(f"Average group size: {avg_group_size:.2f}")
            print(f"Median group size: {median_group_size}")
            print(f"Largest group size: {max_group_size}")
            print(f"Smallest group size: {min_group_size}")
            print(f"Number of single-entity groups: {single_entity_groups} ({single_entity_groups / total_groups:.2%})")
            print(f"Number of multi-entity groups: {total_groups - single_entity_groups} ({(total_groups - single_entity_groups) / total_groups:.2%})")
            print('---------------------------------')
            # print the largest group
            print(f"Largest group size: {max_group_size}")
            for entity in sorted(max_group, key=lambda x: x.entity_name):
                print('    ', entity.entity_name)

            print('---------------------------------')
            print('Removing single-entity groups')
        
        # Remove single-entity groups
        for group in [group for group in collected_entities if len(group) == 1]:
            collected_entities.remove_entity_group(group.entity_group_id)

        return collected_entities

    def _group_entity_strings(self, entity_strings_list: list[str]) -> StorySageEntityCollection:
        
        self.vectorizer = TfidfVectorizer()
        vectors = self.vectorizer.fit_transform(entity_strings_list)

        return self._get_multipass_cluster(vectors=vectors, strings=entity_strings_list, 
                                           eps_first=0.8, eps_second=0.5,
                                           min_samples=4)

    def get_grouped_entities(self, documents: list[Document], labels: list[str] = ['PERSON']) -> StorySageEntityCollection:
        """Extracts and groups named entities from documents.

        This method combines entity extraction and grouping:
        1. Extracts entities using GLiNER model
        2. Cleans extracted entity strings
        3. Vectorizes entities using TF-IDF
        4. Groups similar entities based on cosine similarity
        5. Returns grouped entities as a StorySageEntityCollection

        Args:
            documents (list[Document]): Documents to process for entity extraction
            labels (list[str], optional): Entity types to extract. Defaults to ['CHARACTER']

        Returns:
            List[GroupType]: List of entity groups, each containing:
                - Centroid vector (spmatrix)
                - Matrix of member vectors (spmatrix)
                - Set of entity strings (Set[str])

        Example:
            >>> docs = [Document(page_content="John and Johnny discussed their plans")]
            >>> grouped = extractor.get_grouped_entities(docs)
            >>> print(len(grouped))  # Number of entity groups
            1
        """

        entities_dict = self._extract_entities(documents, labels)
        entity_strings = set()
        for entity_type, entities in entities_dict.items():
            if entity_type != 'PERSON':
                continue
            for entity in entities:
                cleaned = self._clean_string(entity)
                if cleaned:
                    entity_strings.add(cleaned)

        entity_strings_list = list(entity_strings)

        with open(f'all_strings_{self.target_series_info.series_metadata_name}.json', 'w') as f:
            json.dump(entity_strings_list, f)

        self.entity_collection = self._group_entity_strings(entity_strings_list=entity_strings_list)
        
        return self.entity_collection

    def regroup_entities(self, new_threshold: float = 0.7) -> List[GroupType]:
        """Regroups entities based on a new similarity threshold.

        This method regroups entities based on a new similarity threshold
        and returns the updated list of entity groups.

        Args:
            new_threshold (float, optional): New cosine similarity threshold. Defaults to 0.7.

        Returns:
            List[GroupType]: List of entity groups after regrouping

        Example:
            >>> grouped = extractor.regroup_entities(new_threshold=0.8)
            >>> print(len(grouped))
        """
        
        self.similarity_threshold = new_threshold
        self.entities = []
        entity_strings = set()
        for group in self.entity_collection.entity_groups:
            for entity in group.entities:
                entity_strings.add(entity.entity_name)

        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit_transform(entity_strings)

        for entity in tqdm(entity_strings, desc=f'Regrouping entities with threshold {new_threshold}'):
            self._add_new_string_to_groups(entity)

        self.entity_collection = StorySageEntityCollection.from_sets(self.entities)

        return self.entity_collection
