import pytest
from story_sage.models.story_sage_series import (
    Book,
    EntitySettings,
    StorySageSeries,
    StorySageSeriesCollection
)

@pytest.fixture
def book():
    return Book(
        number_in_series=1,
        title="Leviathan Wakes",
        book_metadata_name="01_leviathan_wakes",
        number_of_chapters=55,
        cover_image="4.1.jpg"
    )

@pytest.fixture
def another_book():
    return Book(
        number_in_series=2,
        title="Caliban’s War",
        book_metadata_name="02_calibans_war",
        number_of_chapters=54,
        cover_image="4.2.jpg"
    )

@pytest.fixture
def entity_settings():
    return EntitySettings(
        names_to_skip=["Mr.", "Mrs.", "Dr.", "light"],
        person_titles=["King", "Queen", "Prince", "doctor", "my dear", "Detective", "Inspector"]
    )

@pytest.fixture
def story_sage_series(entity_settings, book, another_book):
    return StorySageSeries(
        series_id=4,
        series_name="The Expanse",
        series_metadata_name="the_expanse",
        entity_settings=entity_settings,
        books=[book, another_book]
    )

@pytest.fixture
def additional_series(entity_settings):
    book3 = Book(
        number_in_series=3,
        title="Abaddon’s Gate",
        book_metadata_name="03_abaddons_gate",
        number_of_chapters=53,
        cover_image="4.3.jpg"
    )
    return StorySageSeries(
        series_id=4,
        series_name="The Expanse",
        series_metadata_name="the_expanse",
        entity_settings=entity_settings,
        books=[book3]
    )

@pytest.fixture
def story_sage_series_collection(story_sage_series, additional_series):
    collection = StorySageSeriesCollection()
    collection.add_series(story_sage_series)
    return collection

def test_book_initialization(book):
    assert book.number_in_series == 1
    assert book.title == "Leviathan Wakes"  
    assert book.book_metadata_name == "01_leviathan_wakes"  
    assert book.number_of_chapters == 55  
    assert book.cover_image == "4.1.jpg"  

def test_book_to_json(book):
    expected_json = {
        'number_in_series': 1,
        'title': "Leviathan Wakes", 
        'book_metadata_name': "01_leviathan_wakes", 
        'number_of_chapters': 55, 
        'cover_image': "4.1.jpg" 
    }
    assert book.to_json() == expected_json

def test_entity_settings_initialization(entity_settings):
    assert entity_settings.names_to_skip == ["Mr.", "Mrs.", "Dr.", "light"]
    assert entity_settings.person_titles == ["King", "Queen", "Prince", "doctor", "my dear", "Detective", "Inspector"]

def test_story_sage_series_initialization(story_sage_series):
    assert story_sage_series.series_id == 4  
    assert story_sage_series.series_name == "The Expanse"  
    assert story_sage_series.series_metadata_name == "the_expanse"
    assert len(story_sage_series.books) == 2  
    assert story_sage_series.books[0].title == "Leviathan Wakes"  

def test_story_sage_series_from_dict():
    data = {
        'series_id': 2,
        'series_name': 'Mystery Tales',
        'series_metadata_name': 'mystery_tales',
        'entity_settings': {
            'names_to_skip': ['Ms.', 'Miss'],
            'person_titles': ['Detective', 'Inspector']
        },
        'books': [
            {
                'number_in_series': 1,
                'title': 'The Silent Witness',
                'book_metadata_name': 'the_silent_witness',
                'number_of_chapters': 10,
                'cover_image': 'silence.jpg'
            }
        ]
    }
    series = StorySageSeries.from_dict(data)
    assert series.series_id == 2
    assert series.series_name == 'Mystery Tales'
    assert series.series_metadata_name == 'mystery_tales'
    assert series.entity_settings.names_to_skip == ['Ms.', 'Miss']
    assert series.entity_settings.person_titles == ['Detective', 'Inspector']
    assert len(series.books) == 1
    assert series.books[0].title == 'The Silent Witness'

def test_story_sage_series_to_metadata_json(story_sage_series):
    metadata_json = story_sage_series.to_metadata_json()
    expected_json = {
        'series_id': 4, 
        'series_name': "The Expanse", 
        'series_metadata_name': "the_expanse", 
        'books': [
            {
                'number_in_series': 1,
                'title': "Leviathan Wakes", 
                'book_metadata_name': "01_leviathan_wakes", 
                'number_of_chapters': 55, 
                'cover_image': "4.1.jpg" 
            },
            {
                'number_in_series': 2,
                'title': "Caliban’s War",
                'book_metadata_name': "02_calibans_war",
                'number_of_chapters': 54,
                'cover_image': "4.2.jpg"
            }
        ]
    }
    assert metadata_json == expected_json

def test_story_sage_series_collection_initialization():
    collection = StorySageSeriesCollection()
    assert len(collection.series_list) == 0

def test_story_sage_series_collection_get_series_by_name(story_sage_series_collection, story_sage_series):
    series = story_sage_series_collection.get_series_by_name("The Expanse") 
    assert series is not None
    assert series.series_name == "The Expanse"

    non_existent = story_sage_series_collection.get_series_by_name("Non Existent Series")
    assert non_existent is None

def test_story_sage_series_collection_to_metadata_json(story_sage_series_collection):
    metadata_json = story_sage_series_collection.to_metadata_json()
    expected_json = {
        'series_list': [
            {
                'series_id': 4,  
                'series_name': "The Expanse",  
                'series_metadata_name': "the_expanse",  
                'books': [
                    {
                        'number_in_series': 1,
                        'title': "Leviathan Wakes",
                        'book_metadata_name': "01_leviathan_wakes",
                        'number_of_chapters': 55,
                        'cover_image': "4.1.jpg"
                    },
                    {
                        'number_in_series': 2,
                        'title': "Caliban’s War",
                        'book_metadata_name': "02_calibans_war",
                        'number_of_chapters': 54,
                        'cover_image': "4.2.jpg"
                    }
                ]
            }
        ]
    }
    print(metadata_json)
    print(expected_json)
    assert metadata_json == expected_json

def test_story_sage_series_collection_from_metadata_json():
    metadata = {
        'series_list': [
            {
                'series_id': 1,
                'series_name': "Epic Fantasy",
                'series_metadata_name': "epic_fantasy",
                'entity_settings': {
                    'names_to_skip': ["Mr.", "Mrs."],
                    'person_titles': ["King", "Queen"]
                },
                'books': [
                    {
                        'number_in_series': 1,
                        'title': "The Beginning",
                        'book_metadata_name': "the_beginning",
                        'number_of_chapters': 12,
                        'cover_image': "beginning.jpg"
                    }
                ]
            },
            {
                'series_id': 2,
                'series_name': "Mystery Tales",
                'series_metadata_name': "mystery_tales",
                'entity_settings': {
                    'names_to_skip': ["Ms.", "Miss"],
                    'person_titles': ["Detective", "Inspector"]
                },
                'books': []
            }
        ]
    }
    collection = StorySageSeriesCollection.from_metadata_json(metadata)
    assert len(collection.series_list) == 2
    assert collection.series_list[0].series_name == "Epic Fantasy"
    assert collection.series_list[1].series_name == "Mystery Tales"

def test_story_sage_series_initialization_with_mock_data(mock_series_data):
    series = StorySageSeries.from_dict(mock_series_data[0])
    assert series.series_id == '1'
    assert series.series_name == 'Sherlock Holmes'
    assert series.series_metadata_name == 'sherlock_holmes'
    assert series.entity_settings.names_to_skip == ['light']
    assert series.entity_settings.person_titles == ['doctor', 'my dear']
    assert len(series.books) == 2
    assert series.books[0].title == 'A Study in Scarlet'

def test_story_sage_series_collection_add_series(story_sage_series_collection, story_sage_series):
    new_series = StorySageSeries(
        series_id=2,  
        series_name="Mystery Tales",
        series_metadata_name="mystery_tales",
        entity_settings=EntitySettings(
            names_to_skip=["Ms.", "Miss"],
            person_titles=["Detective", "Inspector"]
        ),
        books=[]
    )
    updated_collection = StorySageSeriesCollection()
    for series in story_sage_series_collection.series_list:
        updated_collection.add_series(series)
    updated_collection.add_series(new_series)
    assert len(updated_collection.series_list) == 2
    assert updated_collection.series_list[1].series_name == "Mystery Tales"