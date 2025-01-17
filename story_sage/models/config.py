from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .series import StorySageSeries
from .entity import StorySageEntityCollection
import json
import yaml
import logging
import redis
import time

logger = logging.getLogger(__name__)

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
        ...     'RAPTOR_COLLECTION': 'raptor',
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
    raptor_collection: str
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
        logger.debug("Converting %d series to JSON format", len(self.series))
        return [series.to_metadata_json() for series in self.series]
    
    def get_series_by_meta_name(self, series_metadata_name: str) -> Optional[StorySageSeries]:
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
        logger.debug("Looking up series by metadata name: %s", series_metadata_name)
        series = next((s for s in self.series if s.series_metadata_name == series_metadata_name), None)
        if series is None:
            logger.warning("No series found with metadata name: %s", series_metadata_name)
        return series

    @classmethod
    def from_config(cls, config: dict, sparse: bool = False) -> 'StorySageConfig':
        """Creates a new StorySageConfig instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing all required settings:
                - OPENAI_API_KEY (str): OpenAI API authentication key
                - CHROMA_PATH (str): Path to ChromaDB storage
                - RAPTOR_COLLECTION (str): Name of collection for raptor retrieval
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
        logger.info("Initializing StorySageConfig from dictionary")
        start_time = time.time()

        # Validate required configuration keys
        required_keys = [
            'OPENAI_API_KEY', 'CHROMA_PATH', 'SERIES_PATH', 'ENTITIES_PATH', 
            'N_CHUNKS', 'REDIS_URL', 'RAPTOR_COLLECTION', 'REDIS_EXPIRE', 
            'PROMPTS_PATH', 'COMPLETION_MODEL', 'COMPLETION_TEMPERATURE',
            'COMPLETION_MAX_TOKENS'
        ]
        
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            logger.error("Missing required configuration keys: %s", missing_keys)
            raise ValueError(f"Config is missing required keys: {', '.join(missing_keys)}")

        # Initialize base configuration
        ssconfig = StorySageConfig(
            openai_api_key=config['OPENAI_API_KEY'],
            chroma_path=config['CHROMA_PATH'],
            n_chunks=config['N_CHUNKS'],
            completion_model=config['COMPLETION_MODEL'],
            completion_temperature=config['COMPLETION_TEMPERATURE'],
            completion_max_tokens=config['COMPLETION_MAX_TOKENS'],
            raptor_collection=config['RAPTOR_COLLECTION']
        )

        # Load series configurations
        if not sparse:
            try:
                logger.info("Loading series from %s", config['SERIES_PATH'])
                series_start = time.time()
                with open(config['SERIES_PATH'], 'r') as file:
                    series_list = yaml.safe_load(file)
                    ssconfig.series = [StorySageSeries.from_dict(series) for series in series_list]
                logger.info("Loaded %d series in %.2fs", 
                        len(ssconfig.series), time.time() - series_start)
            except (yaml.YAMLError, FileNotFoundError) as e:
                logger.error("Failed to load series: %s", str(e), exc_info=True)
                raise

        # Initialize Redis connection
        if not sparse:    
            try:
                logger.info("Connecting to Redis at %s", config['REDIS_URL'])
                redis_start = time.time()
                ssconfig.redis_instance = redis.Redis.from_url(config['REDIS_URL'])
                ssconfig.redis_ex = config['REDIS_EXPIRE']
                # Test connection
                ssconfig.redis_instance.ping()
                logger.info("Redis connection established in %.2fs", 
                        time.time() - redis_start)
            except redis.RedisError as e:
                logger.error("Failed to connect to Redis: %s", str(e), exc_info=True)
                raise

        # Load prompt templates
        if not sparse:
            try:
                logger.info("Loading prompts from %s", config['PROMPTS_PATH'])
                prompt_start = time.time()
                with open(config['PROMPTS_PATH'], 'r') as file:
                    ssconfig.prompts = yaml.safe_load(file)
                logger.info("Loaded %d prompt templates in %.2fs", 
                        len(ssconfig.prompts), time.time() - prompt_start)
            except (yaml.YAMLError, FileNotFoundError) as e:
                logger.error("Failed to load prompts: %s", str(e), exc_info=True)
                raise

        total_time = time.time() - start_time
        logger.info("StorySageConfig initialization completed in %.2fs", total_time)
        return ssconfig
    
    @classmethod
    def from_file(cls, config_path: str) -> 'StorySageConfig':
        """Creates a new StorySageConfig instance from a configuration file.

        Args:
            config_path (str): Path to the configuration file (YAML format)

        Returns:
            StorySageConfig: Fully initialized configuration instance

        Raises:
            FileNotFoundError: If the configuration file is not found
            Exception: If the configuration file is invalid or missing required keys

        Example:
            >>> ssconfig = StorySageConfig.from_file('config.yaml')
            >>> print(ssconfig.openai_api_key)
            'sk-...'
        """
        logger.info("Loading configuration from file: %s", config_path)
        try:
            start_time = time.time()
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.debug("Configuration file loaded in %.2fs", time.time() - start_time)
            return cls.from_config(config)
        except FileNotFoundError:
            logger.error("Configuration file not found: %s", config_path)
            raise
        except yaml.YAMLError as e:
            logger.error("Invalid YAML in configuration file: %s", str(e), exc_info=True)
            raise