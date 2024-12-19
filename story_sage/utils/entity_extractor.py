"""
entity_extractor.py

This module provides the `StorySageEntityExtractor` class, which is used to extract named entities from text using the GPT-4 language model.

The extracted entities include people, places, groups, animals, and objects.

Example Usage:
    ```python
    extractor = StorySageEntityExtractor(api_key="your_api_key")
    text = "John Doe went to New York with his dog."
    entities, usage = extractor._extract_named_entities(text)
    print(entities)
    # Output:
    # {
    #     "people": ["John Doe"],
    #     "places": ["New York"],
    #     "groups": [],
    #     "animals": ["dog"],
    #     "objects": []
    # }
    print(usage)
    # Output: Usage information dictionary
    ```
"""

from openai import OpenAI
import httpx
import time
from pydantic import BaseModel

class StorySageEntityExtractor():
    """
    A class for extracting named entities from text using the GPT-4 language model.

    Attributes:
        client (OpenAI): An instance of the OpenAI client for making API calls.

    Example Usage:
        ```
        extractor = StorySageEntityExtractor(api_key="your_api_key")
        text = "John Doe went to New York with his dog."
        entities, usage = extractor.extract_named_entities(text)
        print(entities)
        # Output:
        # {
        #     "people": ["John Doe"],
        #     "places": ["New York"],
        #     "groups": [],
        #     "animals": ["dog"],
        #     "objects": []
        # }
        print(usage)
        # Output: Usage information dictionary
        ```
    """

    def __init__(self, api_key: str):
        """
        Initialize the StorySageEntityExtractor with the provided OpenAI API key.

        Args:
            api_key (str): Your OpenAI API key.
        """
        # Create an HTTP client with SSL verification disabled
        req_client = httpx.Client(verify=False)
        # Initialize the OpenAI client with the API key and HTTP client
        self.client = OpenAI(api_key=api_key, http_client=req_client)

    def _extract_named_entities(self, text: str) -> tuple[dict, dict]:
        """
        Extract named entities and usage information from the given text using a GPT-4 model.

        This method sends a request to a GPT-4 model to perform named entity recognition (NER) on the provided text.
        The model identifies and extracts entities such as people, places, groups, animals, and objects.

        Args:
            text (str): The input text from which to extract named entities.

        Returns:
            tuple[dict, dict]: 
                - extracted_entity (dict): Extracted entities categorized by type.
                - usage_information (dict): Information about the API usage.

        Example:
            ```
            text = "Alice visited Wonderland with her cat."
            entities, usage = extractor.extract_named_entities(text)
            print(entities)
            # Output:
            # {
            #     "people": ["Alice"],
            #     "places": ["Wonderland"],
            #     "groups": [],
            #     "animals": ["cat"],
            #     "objects": []
            # }
            print(usage)
            # Output: Usage information dictionary
            ```
        """
        # Initialize the completion request to the OpenAI model
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
                    You are a highly advanced natural language processing agent that 
                    is optimized to perform named entity recognition (NER). Your goal is to
                    extract entities from text provided to you.

                    For example, if the text is:
                        Alice saw the White Rabbit while sitting by the river.

                    Extract:
                        People: Alice, White Rabbit
                        Places: river
                        Groups: []
                        Animals: rabbit
                        Objects: []
                    """},
                {"role": "user", "content": text},
            ],
            response_format=self.StorySageEntities
        )

        # Parse the extracted entities from the completion response
        extracted_entity = completion.choices[0].message.parsed
        usage_information = completion.usage

        # Return the extracted entities and usage information
        return extracted_entity, usage_information

    def extract(self, book_chunks: dict, 
                token_per_min_limit: int = 200000, 
                cooldown_secs: int = 30) -> list:
        """
        Extract named entities from chunks of text in a book.

        This method processes a dictionary of book chunks, extracting entities from each chapter while respecting API rate limits.

        Args:
            book_chunks (dict): A dictionary where keys are chapter indices and values are lists of text chunks.
            token_per_min_limit (int, optional): The limit on the number of tokens processed per minute. Defaults to 200000.
            cooldown_secs (int, optional): The cooldown period in seconds to wait if the character limit is exceeded. Defaults to 30.

        Raises:
            ValueError: If cooldown_secs is greater than 30.

        Returns:
            list: A list of extracted named entities from each chapter.

        Example:
            ```
            book_chunks = {
                1: ["Once upon a time...", "The brave knight..."],
                2: ["In a galaxy far away...", "Space battles..."]
            }
            entities = extractor.extract_entities_from_chunks(book_chunks)
            print(entities)
            # Output:
            # [
            #     ({'people': [...], 'places': [...]}, ...),
            #     ({'people': [...], 'places': [...]}, ...)
            # ]
            ```
        """
        # Raise an error if cooldown_secs exceeds the maximum allowed limit
        if cooldown_secs > 30:
            raise ValueError('Cooldown seconds cannot exceed 30 seconds.')
        
        # Calculate the number of chapters and initialize a list to store results
        num_chapters = len(book_chunks)
        result = []

        # Set a limit on the number of tokens processed per minute based on cooldown period
        len_cap = (token_per_min_limit * 4) / (60 / cooldown_secs)  # ~ 4 characters per token, adjust for cooldown
        
        # Initialize a counter to keep track of processed characters
        counter = 0

        # Iterate over each chapter in the book_chunks
        for i, chapter_chunks in book_chunks.items():
            # Combine all text chunks in the chapter into a single string
            chapter_text = '\n'.join(chapter_chunks)

            # Calculate the length of the current chapter
            chapter_len = len(chapter_text)
            
            # Check if processing this chapter would exceed the token limit
            if counter + chapter_len > len_cap:
                print(f'Waiting for {cooldown_secs} seconds to avoid exceeding the character limit. Current chapter: {i + 1}. Current length: {counter}')
                time.sleep(cooldown_secs)
                counter = 0  # Reset the counter after cooldown
            
            # Extract named entities from the current chapter's text
            entities, usage = self._extract_named_entities(chapter_text)
            result.append(entities)

            # Update the counter with the length of the processed chapter
            counter += chapter_len

        # Process the extracted entities to clean and standardize them
        for chapter_entities in result:
            for entity_type, entities in chapter_entities.items():
                processed_entities = []
                for entity in entities:
                    # Convert to lowercase and remove non-alphabetic characters
                    processed_entity = ''.join(
                        char for char in entity.lower() if char.isalpha() or char.isspace()
                    ).strip()
                    # Add to the list if the length is >= 3
                    if len(processed_entity) >= 3:
                        processed_entities.append(processed_entity)
                # Update the entities with the processed list
                chapter_entities[entity_type] = processed_entities

        print(f'Finished extracting from {num_chapters} chapters')
        
        return result


    class StorySageEntities(BaseModel):
        """
        Pydantic model representing the structure of extracted entities.

        Attributes:
            people (List[str]): List of people entities.
            places (List[str]): List of place entities.
            groups (List[str]): List of group entities.
            animals (List[str]): List of animal entities.
            objects (List[str]): List of object entities.
        """
        people: list[str]
        places: list[str]
        groups: list[str]
        animals: list[str]
        objects: list[str]