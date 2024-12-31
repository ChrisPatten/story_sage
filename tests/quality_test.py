import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from story_sage.story_sage import StorySage
import yaml
import json
import markdown
from tqdm import tqdm
from story_sage.story_sage import StorySage, StorySageEntityCollection

# Define paths to configuration files
config_path = Path(__file__).parent.parent / 'config.yml'
test_config_path = Path(__file__).parent / 'test_config.yml'
results_path = Path(__file__).parent / 'quality_test_results.html'

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
    entities_dict=entities,
    series_list=series_list
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
    return result, context

def get_html_results(results):
    """
    Convert the results into an HTML table format.
    
    Args:
        results (list): A list of tuples containing questions, book numbers, and answers.
    
    Returns:
        str: An HTML string representing the results in a table format.
    """

    page_css = """
body {
    font-family: Arial, sans-serif;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    border: 1px solid #ddd;
    padding: 8px;
}

th {
    background-color: #f2f2f2;
    text-align: left;
}
    """

    output = f'<html><head><style>{page_css}</style></head><body><div>'
    html_table = f'<table><tr><th>Question</th>'
    context_table = f'<table><tr><th>Question</th>'
    for book_number in results[0][1]:
        html_table += f'<th>Book {book_number}</th>'
        context_table += f'<th>Book {book_number}</th>'
    html_table += "</tr>"
    context_table += "</tr>"
    for question, book_numbers, answers in results:
        html_table += f'<tr><td>{question}</td>'
        context_table += f'<tr><td>{question}</td>'
        for answer in answers:
            answer_list = '\n\n'.join(answer[1])
            html_table += f'<td>{markdown.markdown(answer[0])}</td>'
            context_table += f'<td>{markdown.markdown(answer_list)}</td>'
        html_table += "</tr>"
        context_table += "</tr>"
    html_table += "</table>"
    context_table += "</table>"
    output += html_table + '</div><div>' + context_table + '</div></body></html>'
    return output

# Load test configuration settings
question_list = test_config['question_list']
series_number = test_config['series_number']
book_numbers = test_config['book_numbers']

results = []

# Iterate over each question and get answers for each book number
for question in tqdm(question_list, desc='Getting responses to questions'):
    question_result = (question, book_numbers, [])
    for book in book_numbers:
        result, context = get_answer(question, series_number, book, chapter_number=99)
        question_result[2].append((result, context))
    results.append(question_result)

# Write the results to an HTML file
with open(results_path, 'w') as file:
    file.write(get_html_results(results))