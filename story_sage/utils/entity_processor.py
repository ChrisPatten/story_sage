from typing import List
from pydantic import BaseModel
from openai import OpenAI
import httpx
import time

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
            entities (list[list[str]]): A list of lists, where each sublist contains
                names that have been grouped together as representing the same entity.
        """
        entities: list[list[str]]

    def __init__(self, api_key: str):
        """
        Initialize the StorySageEntityProcessor with the given API key.

        Args:
            api_key (str): The API key for accessing the OpenAI services.
        """
        req_client = httpx.Client(verify=False)
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, http_client=req_client)

    def zip_entities(self, series_entities, new_entities):
        """
        Combine new chapter entities into the overall series entities.

        Args:
            series_entities (dict): The existing series entities.
            new_entities (list): A list of new entities to be added to the series.

        Returns:
            dict: Updated series entities with new entities incorporated.
        """
        # Iterate over each chapter's entities
        for chapter_entities in new_entities:
            for entity_type, entities in chapter_entities[0].items():
                if entity_type not in series_entities:
                    series_entities[entity_type] = []
                # Extend the list of entities for each type
                series_entities[entity_type].extend(entities)
        return series_entities

    def collect_unique_values(self, series_entities: dict) -> tuple[list, list]:
        """
        Extract unique people and entities from the series entities.

        Args:
            series_entities (dict): The series entities containing various types.

        Returns:
            tuple:
                list: A list of unique people names.
                list: A list of unique other entity names.
        """
        series_people_set = set()
        series_entities_set = set()
        
        # Collect people
        series_people_set.update(series_entities.get('people', []))
            
        # Collect other entities
        for key, values in series_entities.items():
            if key != 'people':
                series_entities_set.update(values)

        series_people_list = []
        series_entities_list = []
        
        # Normalize people names
        for person in series_people_set:
            person = person.lower()
            person = ''.join(c for c in person if c.isalpha() or c.isspace())
            series_people_list.append(person)

        # Normalize entity names
        for entity in series_entities_set:
            entity = entity.lower()
            entity = ''.join(c for c in entity if c.isalpha() or c.isspace())
            series_entities_list.append(entity)
        
        return series_people_list, series_entities_list

    def group_similar_names(self, names_to_group: list[str]) -> GroupedEntities:
        """
        Group similar names together using natural language processing.

        Args:
            names_to_group (list[str]): A list of names to be grouped.

        Returns:
            GroupedEntities: An object containing grouped names.
        """
        text = ', '.join(names_to_group)
        # Make an API call to group similar names
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """
                    You are a highly advanced natural language processing agent that 
                    is optimized to do named entity recognition (NER). Your goal is to
                    group together names that represent the same thing from the text
                    provided to you.
                 
                    Make sure all names in the input are present in the output.   
                 
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
        return completion.choices[0].message.parsed

    def remove_duplicate_elements(self, grouped_entities: GroupedEntities) -> List[List[str]]:
        """
        Remove duplicate names across grouped entities.

        Args:
            grouped_entities (GroupedEntities): The grouped entities from NLP processing.

        Returns:
            List[List[str]]: A list of groups with duplicates removed.
        """
        # Create a set to track seen names
        seen_names = set()
        filtered_groups = []

        # Iterate through each group
        for group in grouped_entities.entities:
            filtered_group = []
            for name in group:
                if name not in seen_names:
                    filtered_group.append(name)
                    seen_names.add(name)
            # Only keep non-empty groups
            if filtered_group:
                filtered_groups.append(filtered_group)
        return filtered_groups

    def create_result_dict(self, people, entities, base_id):
        """
        Generate a result dictionary mapping IDs to names.

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
        
        # Assign unique IDs to people groups
        for i, person_list in enumerate(people):
            person_id = f"{base_id}_p_{i}"
            result['people_by_id'][person_id] = person_list
            for name in person_list:
                result['people_by_name'][name] = person_id
        
        # Assign unique IDs to entity groups
        for j, entity_list in enumerate(entities):
            filtered_entities = [entity for entity in entity_list if entity not in result['people_by_name']]
            if filtered_entities:
                entity_id = f"{base_id}_e_{j}"
                result['entity_by_id'][entity_id] = filtered_entities
                for entity in filtered_entities:
                    result['entity_by_name'][entity] = entity_id
        return result

# Example usage:
if __name__ == "__main__":
    # Initialize the processor with your OpenAI API key
    processor = StorySageEntityProcessor(api_key='your_api_key')

    # Sample data
    series_entities = {}
    new_entities = [
        [{'people': ['Alice', 'Bob'], 'places': ['Wonderland']}],
        [{'people': ['Alice Liddell'], 'places': ['Looking Glass']}]
    ]

    # Combine entities from new chapters into the series
    combined_entities = processor.zip_entities(series_entities, new_entities)
    
    # Extract unique people and entities
    people_list, entities_list = processor.collect_unique_values(combined_entities)
    
    # Group similar names
    grouped_people = processor.group_similar_names(people_list)
    grouped_entities = processor.group_similar_names(entities_list)
    
    # Remove duplicates across groups
    unique_people = processor.remove_duplicate_elements(grouped_people)
    unique_entities = processor.remove_duplicate_elements(grouped_entities)
    
    # Create the final result dictionary
    result_dict = processor.create_result_dict(unique_people, unique_entities, base_id='series1')
    
    # Example result output
    print("People by ID:")
    print(result_dict['people_by_id'])
    print("\nPeople by Name:")
    print(result_dict['people_by_name'])
    print("\nEntities by ID:")
    print(result_dict['entity_by_id'])
    print("\nEntities by Name:")
    print(result_dict['entity_by_name'])