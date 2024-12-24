from story_sage.story_sage import StorySage
import yaml
import json
import markdown
from tqdm import tqdm

# Define paths to configuration files
config_path = './config.yml'
test_config_path = './test_config.yml'

try:
    # Load configuration and data files
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    with open(config['SERIES_PATH'], 'r') as file:
        series_list = yaml.safe_load(file)
    with open(config['ENTITIES_PATH'], 'r') as file:
        entities = json.load(file)
    with open(test_config_path, 'r') as file:
        test_config = yaml.safe_load(file)
except Exception as e:
    raise e

# Extract configuration settings
api_key = config['OPENAI_API_KEY']
chroma_path = config['CHROMA_PATH']
chroma_collection = config['CHROMA_COLLECTION']

# Initialize StorySage instance with configuration settings
story_sage = StorySage(
    api_key=api_key,
    chroma_path=chroma_path,
    chroma_collection_name=chroma_collection,
    entities=entities,
    series=series_list,
    n_chunks=10  # Number of text chunks to process
)

def get_answer(question, series_id, book_number, chapter_number):
    """
    Get an answer from the StorySage instance for a given question and book details.
    
    Args:
        question (str): The question to ask.
        series_id (int): The ID of the series.
        book_number (int): The number of the book in the series.
        chapter_number (int): The number of the chapter in the book.
    
    Returns:
        str: The answer from the StorySage instance.
    """
    data = {
        'question': question,
        'book_number': book_number, 
        'chapter_number': chapter_number,
        'series_id': series_id
    }
    result, context, request_id = story_sage.invoke(**data)
    return result

def get_html_results(results):
    """
    Convert the results into an HTML table format.
    
    Args:
        results (list): A list of tuples containing questions, book numbers, and answers.
    
    Returns:
        str: An HTML string representing the results in a table format.
    """
    html_table = f'<table><tr><th>Question</th>'
    for book_number in results[0][1]:
        html_table += f'<th>Book {book_number}</th>'
    html_table += "</tr>"
    for question, book_numbers, answers in results:
        html_table += f'<tr><td>{question}</td>'
        for answer in answers:
            html_table += f'<td>{markdown.markdown(answer)}</td>'
        html_table += "</tr>"
    html_table += "</table>"
    return f'<html><head><link rel="stylesheet" href="test_result_styles.css"></head><body><div>{html_table}</div></body></html>'

# Load test configuration settings
question_list = test_config['question_list']
series_number = test_config['series_number']
book_numbers = test_config['book_numbers']

results = []

# Iterate over each question and get answers for each book number
for question in tqdm(question_list, desc='Getting responses to questions'):
    question_result = (question, book_numbers, [])
    for book in book_numbers:
        result = get_answer(question, series_number, book, chapter_number=99)
        question_result[2].append(result)
    results.append(question_result)

# Write the results to an HTML file
with open('quality_test_results.html', 'w') as file:
    file.write(get_html_results(results))