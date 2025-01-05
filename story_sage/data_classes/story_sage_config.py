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
    """
    Represents the configuration for StorySage.

    Attributes:
        openai_api_key (str): API key for OpenAI.
        chroma_path (str): Path to the Chroma configuration.
        chroma_collection (str): Name of the Chroma collection.
        n_chunks (int): Number of chunks for processing.
        entities (Dict[str, StorySageEntityCollection]): Dictionary of entity collections.
        series (List[StorySageSeries]): List of series in the configuration.
        redis_instance (redis.Redis): Redis instance for caching.
        redis_ex (int): Redis expiration time in seconds.
    """
    openai_api_key: str
    chroma_path: str
    chroma_collection: str
    n_chunks: int
    entities: Dict[str, StorySageEntityCollection] = field(default_factory=dict)
    series: List[StorySageSeries] = field(default_factory=list)
    redis_instance: redis.Redis = None
    redis_ex: int = None

    def get_series_json(self) -> List[dict]:
        """
        Retrieves the series data as a list of dictionaries.

        Returns:
            List[dict]: A list of dictionaries containing series data.
        
        Example:
            [
                {
                    'series_id': '1',
                    'series_name': 'Series One',
                    'series_metadata_name': 'Metadata One',
                    'books': [
                        {'book_id': '1', 'book_name': 'Book One', ...},
                        {'book_id': '2', 'book_name': 'Book Two', ...}
                    ]
                },
                ...
            ]
        """
        return [series.to_metadata_json() for series in self.series]
    
    def get_series_by_meta_name(self, series_metadata_name: str) -> StorySageSeries:
        """
        Retrieves a StorySageSeries object based on the series_metadata_name provided.

        Returns:
            StorySageSeries: A StorySageSeries object.
        
        Example:
            series = ssconfig.get_series_by_meta_name('sherlock_holmes')
        """
        return next((series for series in self.series if series.series_metadata_name == series_metadata_name), None)

    @classmethod
    def from_config(cls, config: dict) -> 'StorySageConfig':
        """
        Creates an instance of StorySageConfig from a dictionary.

        Args:
            config (dict): A dictionary containing configuration data.

        Returns:
            StorySageConfig: An instance of StorySageConfig.

        Raises:
            ValueError: If a required key is missing from the config.

        Example:
            config = {
                'OPENAI_API_KEY': 'your-api-key',
                'CHROMA_PATH': '/path/to/chroma',
                'CHROMA_COLLECTION': 'collection_name',
                'SERIES_PATH': '/path/to/series.yaml',
                'ENTITIES_PATH': '/path/to/entities.json',
                'N_CHUNKS': 10,
                'REDIS_URL': 'redis://localhost:6379/0',
                'REDIS_EXPIRE': 3600
            }
            ssconfig = StorySageConfig.from_config(config)
        """
        required_keys = ['OPENAI_API_KEY', 'CHROMA_PATH', 'CHROMA_COLLECTION', 
                         'SERIES_PATH', 'ENTITIES_PATH', 'N_CHUNKS', 'REDIS_URL',
                         'REDIS_EXPIRE']
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

        return ssconfig