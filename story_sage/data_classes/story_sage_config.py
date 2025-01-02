from dataclasses import dataclass
from typing import List, Dict
from .story_sage_series import StorySageSeries
from ..story_sage_entity import StorySageEntityCollection
import json
import yaml
import logging

logger = logging.getLogger()

@dataclass
class StorySageConfig:
    """
    Represents the configuration for StorySage.

    Attributes:
        series (List[StorySageSeries]): List of series in the configuration.
    """
    series: List[StorySageSeries]
    openai_api_key: str
    chroma_path: str
    chroma_collection: str
    entities: Dict[str, StorySageEntityCollection]
    n_chunks: int

    def get_series_metadata_names(self) -> List[str]:
        """
        Retrieves the metadata names for all series in the configuration.

        Returns:
            List[str]: A list of metadata names.
        """
        return [series.series_metadata_name for series in self.series]

    def get_series_json(self) -> List[dict]:
        """
        Retrieves the series data as a list of dictionaries.

        Returns:
            List[dict]: A list of dictionaries containing series data.
        """
        result = []
        for series in self.series:
            series_dict = {}
            series_dict['series_id'] = series.series_id
            series_dict['series_name'] = series.series_name
            series_dict['series_metadata_name'] = series.series_metadata_name
            series_dict['books'] = []
            for book in series.books:
                series_dict['books'].append(book.to_json())
            result.append(series_dict)

        return result

    @classmethod
    def from_config(cls, config: dict) -> 'StorySageConfig':
        """
        Creates an instance of StorySageConfig from a dictionary.

        Args:
            config (dict): A dictionary containing configuration data.

        Returns:
            StorySageConfig: An instance of StorySageConfig.
        """
        required_keys = ['OPENAI_API_KEY', 'CHROMA_PATH', 'CHROMA_COLLECTION', 
                         'SERIES_PATH', 'ENTITIES_PATH', 'N_CHUNKS']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Config is missing required key: {key}")

        ssconfig = StorySageConfig(
            series=[],
            openai_api_key=config['OPENAI_API_KEY'],
            chroma_path=config['CHROMA_PATH'],
            chroma_collection=config['CHROMA_COLLECTION'],
            entities={},
            n_chunks=config['N_CHUNKS']
        )

        # Load entities from ENTITIES_PATH
        with open(config['ENTITIES_PATH'], 'r') as file:
            entities_dict = json.load(file)
            logger.debug(f"Loaded entities from {config['ENTITIES_PATH']}")
            logger.debug(f"Entities: {entities_dict}")
            ssconfig.entities = {key: StorySageEntityCollection.from_dict(value) for key, value in entities_dict.items()}

        # Load series from SERIES_PATH
        with open(config['SERIES_PATH'], 'r') as file:
            series_list = yaml.safe_load(file)
            ssconfig.series = [StorySageSeries.from_dict(series) for series in series_list]

        return ssconfig