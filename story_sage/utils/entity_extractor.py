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
        req_client = httpx.Client(verify=False)
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
                    is optimized to do named entity recognition (NER). Your goal is to
                    extract entities and a summary from text provided to you.
                    
                    For example, if the text is:
                        Standing with the other Whitecloaks, Perrin saw the Lugard Road near the Manetherendrelle and the border of Murandy.
                        If dogs had been able to make footprints on stone, he would have said the tracks were the prints of a pack of large hounds.
                        He hefted his axe and kicked aside the basket on the road.
                
                    Extract:
                        People: Perrin
                        Places: Lugard Road, Manetherendrelle, Murandy
                        Groups: Whitecloaks, pack
                        Animals: dogs
                        Objects: axe, basket
                    """},
                {"role": "user", "content": text},
            ],
            response_format=self.StorySageEntities
        )

        # Parse the extracted entities from the completion response
        extracted_entity = completion.choices[0].message.parsed
        usage_information = completion.usage

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
            result.append(self._extract_named_entities(chapter_text))

            # Update the counter with the length of the processed chapter
            counter += chapter_len

        print(f'Finished extracting from {num_chapters} chapters')
        
        return result


    class StorySageEntities(BaseModel):
        people: list[str]
        places: list[str]
        groups: list[str]
        animals: list[str]
        objects: list[str]