
import logging
import httpx
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


class StorySageStepback:
    """Class to optimize user queries using OpenAI before sending to ChromaDB."""

    def __init__(self, api_key: str, model_name: str = 'gpt-4o-mini', temperature: float = 0.5):
        """
        Initialize the StorySageStepback instance.

        Args:
            api_key (str): The API key for OpenAI.
            model_name (str, optional): The OpenAI model to use. Defaults to 'gpt-4o-mini'.
            temperature (float, optional): Sampling temperature. Defaults to 0.5.
        """
        self.logger = logging.getLogger(__name__)
        self.llm = ChatOpenAI(api_key=api_key, model=model_name, temperature=temperature, http_client=httpx.Client(verify=False))
        self.prompt = PromptTemplate(
            input_variables=['query'],
            template="""
                Optimize the following user query for better retrieval from a vector database.
                Return only the optimized query with no introduction, summary, or other text.

                Original Query: {query}

                Optimized Query:
            """
        )

    def optimize_query(self, query: str) -> Optional[str]:
        """
        Optimize the user's query using OpenAI.

        Args:
            query (str): The original user query.

        Returns:
            Optional[str]: The optimized query or None if optimization fails.
        """
        self.logger.debug(f"Optimizing query: {query}")
        try:
            messages = self.prompt.invoke({'query': query})
            response = self.llm.invoke(messages)
            optimized_query = response.content.strip()
            self.logger.debug(f"Optimized query: {optimized_query}")
            return optimized_query
        except Exception as e:
            self.logger.error(f"Failed to optimize query: {e}")
            return None