from dataclasses import dataclass, field
from typing import List, Dict
from .story_sage_series import StorySageSeries
from .story_sage_entity import StorySageEntityCollection
import json
import yaml
import logging
import redis

logger = logging.getLogger()

@dataclass
class StorySageConfig:
    """Configuration manager for the StorySage application.

    This class manages the configuration for the StorySage application, including:
    - OpenAI API settings and parameters
    - ChromaDB vector database configurations
    - Entity management and series configurations
    - Redis caching settings
    - Prompt template management

    Attributes:
        openai_api_key (str): API key for OpenAI authentication.
        chroma_path (str): Local filesystem path for ChromaDB storage.
        chroma_collection (str): Name of the main ChromaDB collection for embeddings.
        chroma_full_text_collection (str): Name of the ChromaDB collection for full text storage.
        n_chunks (int): Number of chunks to split text into during processing.
        prompts (Dict[str, List[Dict[str, str]]]): Dictionary of prompt templates for various operations.
        entities (Dict[str, StorySageEntityCollection]): Dictionary of entity collections by name.
        series (List[StorySageSeries]): List of book series configurations.
        redis_instance (redis.Redis): Redis connection for caching.
        redis_ex (int): Redis cache expiration time in seconds.
        completion_model (str): OpenAI model name for completions (e.g., "gpt-3.5-turbo").
        completion_temperature (float): Temperature setting for OpenAI completions (0.0-1.0).
        completion_max_tokens (int): Maximum tokens for OpenAI completion responses.

    Example:
        >>> config = {
        ...     'OPENAI_API_KEY': 'sk-...',
        ...     'CHROMA_PATH': './chromadb',
        ...     'CHROMA_COLLECTION': 'book_embeddings',
        ...     'CHROMA_FULL_TEXT_COLLECTION': 'book_texts',
        ...     'N_CHUNKS': 5,
        ...     'COMPLETION_MODEL': 'gpt-3.5-turbo',
        ...     'COMPLETION_TEMPERATURE': 0.7,
        ...     'COMPLETION_MAX_TOKENS': 2000,
        ...     'SERIES_PATH': 'configs/series.yaml',
        ...     'ENTITIES_PATH': 'configs/entities.json',
        ...     'REDIS_URL': 'redis://localhost:6379/0',
        ...     'REDIS_EXPIRE': 3600,
        ...     'PROMPTS_PATH': 'configs/prompts.yaml'
        ... }
        >>> config_instance = StorySageConfig.from_config(config)
        >>> print(config_instance.chroma_collection)
        'book_embeddings'
    """
    openai_api_key: str
    chroma_path: str
    chroma_collection: str
    chroma_full_text_collection: str
    n_chunks: int
    prompts: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    entities: Dict[str, StorySageEntityCollection] = field(default_factory=dict)
    series: List[StorySageSeries] = field(default_factory=list)
    redis_instance: redis.Redis = None
    redis_ex: int = None
    completion_model: str = None
    completion_temperature: float = None
    completion_max_tokens: int = None

    def get_series_json(self) -> List[dict]:
        """Converts all series configurations to a JSON-serializable format.

        This method is useful for API responses or data serialization.

        Returns:
            List[dict]: List of series configurations where each series contains:
                - series_id (str): Unique identifier for the series
                - series_name (str): Display name of the series
                - series_metadata_name (str): Machine-friendly name for the series
                - books (List[dict]): List of books in the series

        Example:
            >>> config = StorySageConfig.from_config(config_dict)
            >>> series_data = config.get_series_json()
            >>> print(series_data)
            [
                {
                    'series_id': 'HP001',
                    'series_name': 'Harry Potter',
                    'series_metadata_name': 'harry_potter',
                    'books': [
                        {
                            'book_id': 'HP001-1',
                            'book_name': "Harry Potter and the Philosopher's Stone",
                            'file_path': './books/hp1.txt'
                        }
                    ]
                }
            ]
        """
        return [series.to_metadata_json() for series in self.series]
    
    def get_series_by_meta_name(self, series_metadata_name: str) -> StorySageSeries:
        """Retrieves a series configuration by its metadata name.

        Args:
            series_metadata_name (str): The machine-friendly name of the series
                (e.g., 'harry_potter', 'lord_of_rings')

        Returns:
            StorySageSeries: The matching series configuration if found
            None: If no series matches the provided metadata name

        Example:
            >>> config = StorySageConfig.from_config(config_dict)
            >>> hp_series = config.get_series_by_meta_name('harry_potter')
            >>> print(hp_series.series_name)
            'Harry Potter'
            >>> print(hp_series.series_id)
            'HP001'
        """
        return next((series for series in self.series if series.series_metadata_name == series_metadata_name), None)

    @classmethod
    def from_config(cls, config: dict) -> 'StorySageConfig':
        """Creates a new StorySageConfig instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing all required settings:
                - OPENAI_API_KEY (str): OpenAI API authentication key
                - CHROMA_PATH (str): Path to ChromaDB storage
                - CHROMA_COLLECTION (str): Name of embeddings collection
                - CHROMA_FULL_TEXT_COLLECTION (str): Name of full text collection
                - N_CHUNKS (int): Number of text chunks for processing
                - COMPLETION_MODEL (str): OpenAI model name
                - COMPLETION_TEMPERATURE (float): Model temperature setting
                - COMPLETION_MAX_TOKENS (int): Maximum completion tokens
                - SERIES_PATH (str): Path to series YAML config
                - ENTITIES_PATH (str): Path to entities JSON config
                - REDIS_URL (str): Redis connection URL
                - REDIS_EXPIRE (int): Redis cache TTL in seconds
                - PROMPTS_PATH (str): Path to prompts YAML config

        Returns:
            StorySageConfig: Fully initialized configuration instance

        Raises:
            ValueError: If any required configuration key is missing
            Exception: If Redis connection fails or configuration files are invalid

        Example:
            >>> import yaml
            >>> with open('config.yaml', 'r') as f:
            ...     config = yaml.safe_load(f)
            >>> ssconfig = StorySageConfig.from_config(config)
            >>> print(ssconfig.completion_model)
            'gpt-3.5-turbo'
            >>> print(ssconfig.redis_ex)
            3600
        """
        required_keys = ['OPENAI_API_KEY', 'CHROMA_PATH', 'CHROMA_COLLECTION', 'CHROMA_FULL_TEXT_COLLECTION',
                         'SERIES_PATH', 'ENTITIES_PATH', 'N_CHUNKS', 'REDIS_URL',
                         'REDIS_EXPIRE', 'PROMPTS_PATH', 'COMPLETION_MODEL', 'COMPLETION_TEMPERATURE',
                         'COMPLETION_MAX_TOKENS']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Config is missing required key: {key}")

        ssconfig = StorySageConfig(
            openai_api_key=config['OPENAI_API_KEY'],
            chroma_path=config['CHROMA_PATH'],
            chroma_collection=config['CHROMA_COLLECTION'],
            chroma_full_text_collection=config['CHROMA_FULL_TEXT_COLLECTION'],
            n_chunks=config['N_CHUNKS'],
            completion_model=config['COMPLETION_MODEL'],
            completion_temperature=config['COMPLETION_TEMPERATURE'],
            completion_max_tokens=config['COMPLETION_MAX_TOKENS']
        )

        # Load entities from ENTITIES_PATH
        with open(config['ENTITIES_PATH'], 'r') as file:
            entities_dict = json.load(file)
            logger.debug(f"Loaded entities from {config['ENTITIES_PATH']}")
            ssconfig.entities = {key: StorySageEntityCollection.from_dict(value) for key, value in entities_dict.items()}

        # Load series from SERIES_PATH
        with open(config['SERIES_PATH'], 'r') as file:
            series_list = yaml.safe_load(file)
            ssconfig.series = [StorySageSeries.from_dict(series) for series in series_list]

        # Load redis connection
        try:
            ssconfig.redis_instance = redis.Redis.from_url(config['REDIS_URL'])
            ssconfig.redis_ex = config['REDIS_EXPIRE']
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")  

        # Load prompts from PROMPTS_PATH
        with open(config['PROMPTS_PATH'], 'r') as file:
            ssconfig.prompts = yaml.safe_load(file)
            logger.debug(f"Loaded prompts from {config['PROMPTS_PATH']}")

        return ssconfig