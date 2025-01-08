import pytest
import json
import yaml
from unittest.mock import mock_open, patch, MagicMock
from story_sage.models.story_sage_config import StorySageConfig
from story_sage.models.story_sage_series import StorySageSeries
from story_sage.models.story_sage_entity import StorySageEntityCollection

def test_config_initialization(mock_config_dict):
    config = StorySageConfig(
        openai_api_key=mock_config_dict['OPENAI_API_KEY'],
        chroma_path=mock_config_dict['CHROMA_PATH'],
        chroma_collection=mock_config_dict['CHROMA_COLLECTION'],
        chroma_full_text_collection=mock_config_dict['CHROMA_FULL_TEXT_COLLECTION'],
        n_chunks=mock_config_dict['N_CHUNKS']
    )
    
    assert config.openai_api_key == 'test-key'
    assert config.chroma_path == './test_chromadb'
    assert config.n_chunks == 5

def test_from_config(mock_config_dict, mock_series_data, mock_entities_data, mock_prompts_data):
    with patch('builtins.open', mock_open()) as mock_file:
        with patch('yaml.safe_load') as mock_yaml_load:
            with patch('json.load') as mock_json_load:
                with patch('redis.Redis.from_url') as mock_redis:
                    mock_yaml_load.side_effect = [mock_series_data, mock_prompts_data]
                    mock_json_load.return_value = mock_entities_data
                    mock_redis.return_value = MagicMock()

                    config = StorySageConfig.from_config(mock_config_dict)

                    assert config.openai_api_key == 'test-key'
                    assert isinstance(config.series[0], StorySageSeries)
                    assert len(config.series) == 2
                    assert 'harry_potter' in config.entities
                    assert 'sherlock_holmes' in config.entities
                    assert config.redis_ex == 3600

def test_get_series_json(mock_config_dict, mock_series_data, mock_entities_data, mock_prompts_data):
    with patch('builtins.open', mock_open()) as mock_file:
        with patch('yaml.safe_load') as mock_yaml_load:
            with patch('json.load') as mock_json_load:
                with patch('redis.Redis.from_url') as mock_redis:
                    mock_yaml_load.side_effect = [mock_series_data, mock_prompts_data]
                    mock_json_load.return_value = mock_entities_data
                    mock_redis.return_value = MagicMock()

                    config = StorySageConfig.from_config(mock_config_dict)
                    series_json = config.get_series_json()

                    assert len(series_json) == 2
                    assert series_json[0]['series_id'] == '1'
                    assert series_json[0]['series_metadata_name'] == 'sherlock_holmes'
                    assert series_json[1]['series_id'] == '2'
                    assert series_json[1]['series_metadata_name'] == 'harry_potter'

def test_get_series_by_meta_name(mock_config_dict, mock_series_data, mock_entities_data, mock_prompts_data):
    with patch('builtins.open', mock_open()) as mock_file:
        with patch('yaml.safe_load') as mock_yaml_load:
            with patch('json.load') as mock_json_load:
                with patch('redis.Redis.from_url') as mock_redis:
                    mock_yaml_load.side_effect = [mock_series_data, mock_prompts_data]
                    mock_json_load.return_value = mock_entities_data
                    mock_redis.return_value = MagicMock()

                    config = StorySageConfig.from_config(mock_config_dict)
                    
                    # Test Sherlock Holmes series
                    sherlock_series = config.get_series_by_meta_name('sherlock_holmes')
                    assert sherlock_series is not None
                    assert sherlock_series.series_id == '1'
                    assert len(sherlock_series.books) == 2
                    
                    # Test Harry Potter series
                    harry_potter_series = config.get_series_by_meta_name('harry_potter')
                    assert harry_potter_series is not None
                    assert harry_potter_series.series_id == '2'
                    assert len(harry_potter_series.books) == 1
                    
                    # Test non-existent series
                    non_existent = config.get_series_by_meta_name('non_existent')
                    assert non_existent is None

def test_from_config_missing_key(mock_config_dict):
    incomplete_config = mock_config_dict.copy()
    del incomplete_config['OPENAI_API_KEY']
    
    with pytest.raises(ValueError) as exc_info:
        StorySageConfig.from_config(incomplete_config)
    
    assert "Config is missing required key" in str(exc_info.value)

def test_from_config_redis_failure(mock_config_dict, mock_series_data, mock_entities_data, mock_prompts_data):
    with patch('builtins.open', mock_open()) as mock_file:
        with patch('yaml.safe_load') as mock_yaml_load:
            with patch('json.load') as mock_json_load:
                with patch('redis.Redis.from_url') as mock_redis:
                    mock_yaml_load.side_effect = [mock_series_data, mock_prompts_data]
                    mock_json_load.return_value = mock_entities_data
                    mock_redis.side_effect = Exception("Redis connection failed")

                    config = StorySageConfig.from_config(mock_config_dict)
                    assert config.redis_instance is None
