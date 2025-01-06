from dataclasses import dataclass, field
from typing import List, Dict
from .story_sage_series import StorySageSeries
from ..story_sage_entity import StorySageEntityCollection
import json
import yaml
import logging
import redis

logger = logging.getLogger()

@dataclass
class StorySageConfig:
    """Configuration manager for the StorySage application.

    This class handles all configuration aspects of StorySage, including OpenAI integration,
    vector database settings, entity management, and caching configurations.

    Attributes:
        openai_api_key (str): Authentication key for OpenAI API access.
        chroma_path (str): File system path to the Chroma vector database.
        chroma_collection (str): Name of the collection within Chroma DB.
        n_chunks (int): Number of text chunks for processing large documents.
        prompts (Dict[str, List[Dict[str, str]]]): Template prompts for various operations.
        entities (Dict[str, StorySageEntityCollection]): Named collections of story entities.
        series (List[StorySageSeries]): Book series configurations.
        redis_instance (redis.Redis): Connection to Redis cache server.
        redis_ex (int): Cache expiration time in seconds.

    Example:
        >>> config = {
        ...     'OPENAI_API_KEY': 'sk-...',
        ...     'CHROMA_PATH': './chromadb',
        ...     'CHROMA_COLLECTION': 'books',
        ...     'N_CHUNKS': 5,
        ...     'SERIES_PATH': 'series.yaml',
        ...     'ENTITIES_PATH': 'entities.json',
        ...     'REDIS_URL': 'redis://localhost:6379/0',
        ...     'REDIS_EXPIRE': 3600,
        ...     'PROMPTS_PATH': 'prompts.yaml'
        ... }
        >>> ssconfig = StorySageConfig.from_config(config)
    """
    openai_api_key: str
    chroma_path: str
    chroma_collection: str
    n_chunks: int
    prompts: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    entities: Dict[str, StorySageEntityCollection] = field(default_factory=dict)
    series: List[StorySageSeries] = field(default_factory=list)
    redis_instance: redis.Redis = None
    redis_ex: int = None

    def get_series_json(self) -> List[dict]:
        """Converts all series configurations to a JSON-compatible format.

        Returns:
            List[dict]: List of series configurations in dictionary format.

        Example:
            >>> ssconfig = StorySageConfig.from_config(config)
            >>> series_data = ssconfig.get_series_json()
            >>> print(series_data[0])
            {
                'series_id': 'SH001',
                'series_name': 'Sherlock Holmes',
                'series_metadata_name': 'sherlock_holmes',
                'books': [
                    {
                        'book_id': 'SH001-1',
                        'book_name': 'A Study in Scarlet',
                        'file_path': './books/study_in_scarlet.txt'
                    }
                ]
            }
        """
        return [series.to_metadata_json() for series in self.series]
    
    def get_series_by_meta_name(self, series_metadata_name: str) -> StorySageSeries:
        """Retrieves a specific series by its metadata name.

        Args:
            series_metadata_name (str): The unique metadata name of the series.

        Returns:
            StorySageSeries: The matching series configuration object.
            None: If no series matches the provided metadata name.

        Example:
            >>> ssconfig = StorySageConfig.from_config(config)
            >>> series = ssconfig.get_series_by_meta_name('sherlock_holmes')
            >>> print(series.series_name)
            'Sherlock Holmes'
        """
        return next((series for series in self.series if series.series_metadata_name == series_metadata_name), None)

    @classmethod
    def from_config(cls, config: dict) -> 'StorySageConfig':
        """Creates a StorySageConfig instance from a configuration dictionary.

        Initializes all components of StorySage including entity collections, series
        configurations, Redis cache, and prompt templates.

        Args:
            config (dict): Configuration parameters including:
                - OPENAI_API_KEY: OpenAI authentication key
                - CHROMA_PATH: Path to ChromaDB storage
                - CHROMA_COLLECTION: Name of ChromaDB collection
                - SERIES_PATH: Path to series configuration YAML
                - ENTITIES_PATH: Path to entities JSON file
                - N_CHUNKS: Number of text chunks for processing
                - REDIS_URL: Redis connection URL
                - REDIS_EXPIRE: Redis cache expiration time
                - PROMPTS_PATH: Path to prompt templates YAML

        Returns:
            StorySageConfig: Initialized configuration object.

        Raises:
            ValueError: If any required configuration key is missing.
            Exception: If Redis connection fails.

        Example:
            >>> with open('config.yaml', 'r') as f:
            ...     config = yaml.safe_load(f)
            >>> ssconfig = StorySageConfig.from_config(config)
            >>> print(ssconfig.chroma_collection)
            'books'
        """
        required_keys = ['OPENAI_API_KEY', 'CHROMA_PATH', 'CHROMA_COLLECTION', 
                         'SERIES_PATH', 'ENTITIES_PATH', 'N_CHUNKS', 'REDIS_URL',
                         'REDIS_EXPIRE', 'PROMPTS_PATH']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Config is missing required key: {key}")

        ssconfig = StorySageConfig(
            openai_api_key=config['OPENAI_API_KEY'],
            chroma_path=config['CHROMA_PATH'],
            chroma_collection=config['CHROMA_COLLECTION'],
            n_chunks=config['N_CHUNKS'],
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