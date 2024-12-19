from typing import List, Set, Dict
from pydantic import BaseModel
from openai import OpenAI
import httpx
import os
import glob
import json
import re
from tqdm import tqdm

class StorySageEntityProcessor:
    """
    A class for processing entities in a story series.

    This class provides methods to aggregate and process entities such as people and
    other items within a story series. It utilizes natural language processing (NLP)
    techniques to group similar names and remove duplicates.

    Attributes:
        api_key (str): The API key for accessing the OpenAI services.
        client (OpenAI): An instance of the OpenAI client for making API requests.
    """

    class GroupedEntities(BaseModel):
        """
        A Pydantic model representing grouped entities.

        Attributes:
            entities (List[List[str]]): A list of lists, where each sublist contains
                names that have been grouped together as representing the same entity.
        """
        entities: list[list[str]]

    def __init__(self, api_key: str):
        """
        Initialize the StorySageEntityProcessor with the given API key.

        Args:
            api_key (str): The API key for accessing the OpenAI services.
        """
        # Initialize the HTTP client with SSL verification disabled.
        req_client = httpx.Client(verify=False)
        self.api_key = api_key
        # Initialize the OpenAI client for making API requests.
        self.client = OpenAI(api_key=api_key, http_client=req_client)

    def get_cleaned_entities_list(self, entities: list) -> list:
        """
        Remove entities that are too short or contain special characters.

        Args:
            entities (list): A list of entities to be cleaned.

        Returns:
            list: A list of cleaned entities.
        """
        result = []
        for entity in entities:
            # Remove non-alphabetic characters and convert to lowercase.
            entity = ''.join(c for c in entity.lower() if c.isalpha() or c.isspace())
            if len(entity) >= 3:
                result.append(entity)
        return result

    def zip_entities(self, series_entities: dict[str, Set[str]], new_entities: dict[str, Set[str]]) -> dict[str, Set[str]]:
        """
        Combine new chapter entities into the overall series entities.

        Args:
            series_entities (dict): The existing series entities.
            new_entities (dict): A dictionary of new entities to be added to the series.

        Returns:
            dict: Updated series entities with new entities incorporated.
        """
        # Iterate over each entity type in the new entities.
        for entity_type, entities in new_entities.items():
            if entity_type not in ['people', 'entities']:
                continue  # Skip types we're not interested in.
            if entity_type not in series_entities:
                series_entities[entity_type] = set()
            # Update the set of entities for each type.
            series_entities[entity_type].update(entities)
        return series_entities

    def get_unique_entities(self, entities: list) -> list:
        """
        Extract unique entities from the entity lists.

        Args:
            entities (list): A list of entity lists.

        Returns:
            list: A list of unique entities.
        """
        entities_set = set()
        for entity in entities:
            entities_set.update(entity)  # Add entities to the set to ensure uniqueness.
        return list(entities_set)

    def group_similar_names(self, names_to_group: list[str]) -> GroupedEntities:
        """
        Group similar names together using natural language processing.

        Args:
            names_to_group (list[str]): A list of names to be grouped.

        Returns:
            GroupedEntities: An object containing grouped names.
        """
        if not names_to_group:
            raise ValueError("The list of names to group cannot be empty.")
        # Join the names into a single string for processing.
        text = ', '.join(names_to_group)
        # Make an API call to group similar names using the OpenAI model.
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """
                    You are a highly advanced natural language processing agent that 
                    is optimized to do named entity recognition (NER). Your goal is to
                    group together names that represent the same thing from the text
                    provided to you.
                 
                    Make sure all names in the input are present in the output.
                    Do not add any names that are not in the input.
                 
                    For example:
                        Input: Bran, Mat, Bran al'Vere, Haral Luhhan, Breyan, Matrim Cauthon, Alsbet Luhhan, Master al'Vere, Mat Cauthon
                        Output: [['Bran', "Bran al'Vere", "Master al'Vere"], ['Mat', 'Matrim Cauthon', 'Mat Cauthon'], ['Breyan'], ['Haral Luhhan'], ['Alsbet Luhhan']]
                 
                    Another example:
                        Input: sword, axe, horse, spear, mare
                        Output: [['sword', 'axe', 'spear'], ['horse', 'mare']]
                    """},
                {"role": "user", "content": text},
            ],
            response_format=self.GroupedEntities
        )
        # Return the parsed grouped entities.
        return completion.choices[0].message.parsed

    def remove_duplicate_elements(self, grouped_entities: GroupedEntities) -> List[List[str]]:
        """
        Remove duplicate names across grouped entities.

        Args:
            grouped_entities (GroupedEntities): The grouped entities from NLP processing.

        Returns:
            List[List[str]]: A list of groups with duplicates removed.
        """
        seen_names = set()  # Set to keep track of names we've already seen.
        filtered_groups = []

        # Iterate through each group and remove duplicates.
        for group in grouped_entities.entities:
            filtered_group = []
            for name in group:
                if name not in seen_names:
                    filtered_group.append(name)
                    seen_names.add(name)  # Mark the name as seen.
            if filtered_group:
                filtered_groups.append(filtered_group)
        return filtered_groups

    def create_result_dict(self, people: List[List[str]], entities: GroupedEntities, base_id: str) -> Dict[str, Dict[str, List[str]]]:
        """
        Create a dictionary containing mappings by ID and name for people and entities.

        Args:
            people (List[List[str]]): Grouped lists of people names.
            entities (List[List[str]]): Grouped lists of other entity names.
            base_id (str): The base identifier for the series.

        Returns:
            dict: A dictionary containing mappings by ID and name for people and entities.
        """
        result = {
            'people_by_id': {},
            'people_by_name': {},
            'entity_by_id': {},
            'entity_by_name': {}
        }
        
        # Assign unique IDs to each group of people.
        for i, person_list in enumerate(people.entities):
            person_id = f"{base_id}_p_{i}"
            result['people_by_id'][person_id] = person_list
            try:
                for name in person_list:
                    result['people_by_name'][name] = person_id
            except TypeError as e:
                print(f"Error processing person list: {person_list}")
                raise e
        
        # Assign unique IDs to each group of entities.
        for j, entity_list in enumerate(entities.entities):
            # Filter out entities that are already classified under people.
            filtered_entities = [entity for entity in entity_list if entity not in result['people_by_name']]
            if filtered_entities:
                entity_id = f"{base_id}_e_{j}"
                result['entity_by_id'][entity_id] = filtered_entities
                for entity in filtered_entities:
                    result['entity_by_name'][entity] = entity_id
        return result

    def update_entities_from_directory(self, entities_json_path: str, entities_dir: str, series_metadata_name: str, series_id: int):
        """
        Update entities.json based on the JSON files in the entities directory.

        Args:
            entities_json_path (str): Path to the entities.json file.
            entities_dir (str): Directory containing entity JSON files.
            series_metadata_name (str): Metadata name for the series.
            series_id (int): ID of the series.
        """
        # Load existing entities from entities.json
        if os.path.exists(entities_json_path):
            print('Loading existing entities.json')
            with open(entities_json_path, 'r') as f:
                existing_entities = json.load(f)
        else:
            print('No existing entities.json')
            existing_entities = {'series': {}}

        # Process each entity file in the directory
        for entity_file in tqdm(glob.glob(os.path.join(entities_dir, '*.json')), desc='Processing entity files'):
            with open(entity_file, 'r') as f:
                new_entities = json.load(f)
                all_people = set()
                all_entities = set()
                for chapter in new_entities:
                    all_people.update(chapter.get('people', []))
                    for entity_type, entities in chapter.items():
                        if entity_type != 'people':
                            all_entities.update(entities)

                all_people = self.get_cleaned_entities_list(list(all_people))
                all_entities = self.get_cleaned_entities_list(list(all_entities))

                grouped_people = self.group_similar_names(list(all_people))
                grouped_entities = self.group_similar_names(list(all_entities))
                entities_with_ids = self.create_result_dict(grouped_people, grouped_entities, str(series_id))
                # Merge new entities into existing entities
                if str(series_id) not in existing_entities['series']:
                    existing_entities['series'][str(series_id)] = {
                        'series_metadata_name': series_metadata_name,
                        'series_id': series_id,
                        'series_entities': entities_with_ids
                    }
                else:
                    existing_entities['series'][str(series_id)]['series_entities'] = self.zip_entities(
                        existing_entities['series'][str(series_id)]['series_entities'],
                        entities_with_ids
                    )

        # Save updated entities back to entities.json
        with open(entities_json_path, 'w') as f:
            json.dump(existing_entities, f, indent=4)

    def clean_entity_list(self, entities: list) -> list:
        """
        Clean entities by converting to lowercase and removing invalid characters.

        Args:
            entities (list): The list of entities.

        Returns:
            list: The cleaned list of entities.
        """
        result = []
        for entity in entities:
            # Remove any characters not in [a-z] or space, convert to lowercase, and strip whitespace.
            cleaned_entity = re.sub(r'[^a-z\s]', '', entity.lower()).strip()
            if len(cleaned_entity) >= 3:
                result.append(cleaned_entity)
        return result

# Example usage:
if __name__ == "__main__":
    # Initialize the processor with your OpenAI API key.
    api_key = os.getenv('OPENAI_API_KEY', 'API_KEY')

    processor = StorySageEntityProcessor(api_key=api_key)

    # Define the series metadata name and ID.
    series_metadata_name = 'the_expanse'
    series_id = 4

    # Update entities based on JSON files in the specified directory.
    processor.update_entities_from_directory(entities_json_path='./entities.json', 
                                             entities_dir=f'./entities/{series_metadata_name}/',
                                             series_metadata_name=series_metadata_name, 
                                             series_id=series_id)


    # Example result output:
    """
    {
        "series": {
            "1": {
                "series_metadata_name": "harry_potter",
                "series_id": 2,
                "series_name": "Harry Potter",
                "series_entities": {
                    "people_list": [...],
                    "other_entities_list": [...]
                }
            }
        }
    }
    """