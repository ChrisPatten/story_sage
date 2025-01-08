import pytest
import json
import yaml
from unittest.mock import mock_open, patch, MagicMock
from story_sage.models.story_sage_config import StorySageConfig
from story_sage.models.story_sage_series import StorySageSeries
from story_sage.models.story_sage_entity import StorySageEntityCollection

@pytest.fixture
def mock_config_dict():
    return {
        'OPENAI_API_KEY': 'test-key',
        'CHROMA_PATH': './test_chromadb',
        'CHROMA_COLLECTION': 'test_embeddings',
        'CHROMA_FULL_TEXT_COLLECTION': 'test_texts',
        'N_CHUNKS': 5,
        'COMPLETION_MODEL': 'gpt-3.5-turbo',
        'COMPLETION_TEMPERATURE': 0.7,
        'COMPLETION_MAX_TOKENS': 2000,
        'SERIES_PATH': 'test_series.yaml',
        'ENTITIES_PATH': 'test_entities.json',
        'REDIS_URL': 'redis://localhost:6379/0',
        'REDIS_EXPIRE': 3600,
        'PROMPTS_PATH': 'test_prompts.yaml'
    }

@pytest.fixture
def mock_series_data():
    return [
        {
            'series_id': '1',
            'series_name': 'Sherlock Holmes',
            'series_metadata_name': 'sherlock_holmes',
            'entity_settings': {
                'names_to_skip': ['light'],
                'person_titles': ['doctor', 'my dear']
            },
            'books': [
                {
                    'number_in_series': 1,
                    'title': 'A Study in Scarlet',
                    'book_metadata_name': '01_a_study_in_scarlet',
                    'number_of_chapters': 14,
                    'cover_image': '1.1.jpg'
                },
                {
                    'number_in_series': 2,
                    'title': 'The Sign of the Four',
                    'book_metadata_name': '02_the_sign_of_four',
                    'number_of_chapters': 12,
                    'cover_image': '1.2.jpg'
                }
            ]
        },
        {
            'series_id': '2',
            'series_name': 'Harry Potter',
            'series_metadata_name': 'harry_potter',
            'books': [
                {
                    'number_in_series': 1,
                    'title': "Harry Potter and the Sorcerer's Stone",
                    'book_metadata_name': '01_the_sourcerers_stone',
                    'number_of_chapters': 17,
                    'cover_image': '2.1.jpg'
                }
            ]
        }
    ]

@pytest.fixture
def mock_entities_data():
    return {
        "harry_potter": {
            "entity_groups": [
                {
                    "entity_group_id": "3f0253390ac5fa4923a56cfd1358219d",
                    "entities": [
                        {
                            "entity_name": "harry",
                            "entity_type": "entity",
                            "entity_id": "28d0509a63c2fcca54f8924e0c5ba6d0",
                            "entity_group_id": "3f0253390ac5fa4923a56cfd1358219d"
                        },
                        {
                            "entity_name": "potter",
                            "entity_type": "entity",
                            "entity_id": "cb5d80eb797d9f3d6004433371747f5d",
                            "entity_group_id": "3f0253390ac5fa4923a56cfd1358219d"
                        }
                    ]
                },
                {
                    "entity_group_id": "1cce1ea09da65a3d323c3266fd0ca2ee",
                    "entities": [
                        {
                            "entity_name": "voldemort",
                            "entity_type": "entity",
                            "entity_id": "e266edc732a6af8fc92ddcd62a12a85e",
                            "entity_group_id": "1cce1ea09da65a3d323c3266fd0ca2ee"
                        },
                        {
                            "entity_name": "dark lord",
                            "entity_type": "entity",
                            "entity_id": "ad22a7f5ef9b84a8d576115a6c0a6202",
                            "entity_group_id": "1cce1ea09da65a3d323c3266fd0ca2ee"
                        }
                    ]
                }
            ]
        },
        "sherlock_holmes": {
            "entity_groups": [
                {
                    "entity_group_id": "19600491749be1edb2a38d8a66b4bde7",
                    "entities": [
                        {
                            "entity_name": "holmes",
                            "entity_type": "entity",
                            "entity_id": "fa499166e4a9e56069c55f44e9f9304b",
                            "entity_group_id": "19600491749be1edb2a38d8a66b4bde7"
                        },
                        {
                            "entity_name": "sherlock",
                            "entity_type": "entity",
                            "entity_id": "d1606a758db86cb60daacac0b9556b0b",
                            "entity_group_id": "19600491749be1edb2a38d8a66b4bde7"
                        }
                    ]
                }
            ]
        }
    }

@pytest.fixture
def mock_prompts_data():
    return {
        'generate_prompt': [
            {
                'role': 'developer',
                'prompt': 'You are a helpful GPT named Story Sage. Although you are not the author, you should use the voice of an author talking about their own work...'
            },
            {
                'role': 'user',
                'prompt': '# Relevant Context\n{context}'
            },
            {
                'role': 'user',
                'prompt': '# Question\n{question}'
            }
        ],
        'relevant_chunks_prompt': [
            {
                'role': 'developer',
                'prompt': 'You are a helpful GPT named Story Sage...'
            },
            {
                'role': 'user',
                'prompt': '# Chunk Summaries\n{summaries}'
            },
            {
                'role': 'user',
                'prompt': '# Question\n{question}'
            }
        ],
        'generate_keywords_prompt': [
            {
                'role': 'developer',
                'prompt': 'You are a helpful GPT named Story Sage...'
            },
            {
                'role': 'user',
                'prompt': '# Question\n{question}'
            }
        ]
    }
