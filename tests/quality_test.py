import sys
from pathlib import Path
import os
from typing import Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(str(Path(__file__).parent.parent))
import yaml
import logging
import markdown
from tqdm import tqdm
from story_sage.story_sage import StorySage
from story_sage.types import StorySageConfig, StorySageContext

# Define paths to configuration files
config_path = Path(__file__).parent.parent / 'config.yml'
test_config_path = Path(__file__).parent / 'test_config.yml'
results_path = Path(__file__).parent / 'quality_test_results.html'

try:
    # Load configuration and data files
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    STORY_SAGE_CONFIG = StorySageConfig.from_config(config)
    with open(test_config_path, 'r') as file:
        test_config = yaml.safe_load(file)
except Exception as e:
    raise e

test_config = test_config[0]

# Initialize StorySage instance with configuration settings
story_sage = StorySage(
    config=STORY_SAGE_CONFIG,
    log_level=logging.DEBUG
)

logger = logging.getLogger("story_sage")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("story_sage_debug.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
#logger.addHandler(console_handler)

def get_answer(question, series_id, book_number, chapter_number) -> Tuple[str, list[StorySageContext]]:
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
    answer, context, _, _ = story_sage.invoke(**data)
    return answer, context

def get_html_results(results: list[dict]) -> str:
    """
    Convert the results into an HTML table format.
    
    Args:
        results (list): A list of tuples containing questions, book numbers, and answers.
    
    Returns:
        str: An HTML string representing the results in a table format.
    """

    page_css = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: var(--background-color);
    color: var(--text-color);
}

table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    border: 1px solid var(--border-color);
    padding: 12px;
}

th {
    background-color: var(--header-background);
    text-align: left;
}

@media (prefers-color-scheme: dark) {
    :root {
        --background-color: #121212;
        --text-color: #e0e0e0;
        --border-color: #333;
        --header-background: #1f1f1f;
    }
}

@media (prefers-color-scheme: light) {
    :root {
        --background-color: #ffffff;
        --text-color: #000000;
        --border-color: #ddd;
        --header-background: #f2f2f2;
    }
}
    """

    output = f'<html><head><style>{page_css}</style></head><body><div><h1>StorySage Quality Test Results</h1>'
    html_table = f'<table><tr><th>Question</th>'
    context_table = f'<table><tr><th>Question</th>'
    for book_number in results[0]['book_numbers']:
        html_table += f'<th>Book {book_number}</th>'
        context_table += f'<th>Book {book_number}</th>'
    html_table += "</tr>"
    context_table += "</tr>"
    for result in results:
        html_table += f"<tr><td>{result['question']}</td>"
        context_table += f"<tr><td>{result['question']}</td>"

        answer: str
        context: list[StorySageContext]
        for answer, context in result['answers']:
            html_table += f'<td>{markdown.markdown(answer)}</td>'
            context_string = '\n'.join([c.format_for_llm() for c in context])
            context_table += f"<td>{markdown.markdown(context_string)}</td>"
        html_table += "</tr>"
        context_table += "</tr>"
    html_table += "</table>"
    context_table += "</table>"
    output += html_table + '</div><h1>Context Returned:</h1><div>' + context_table + '</div></body></html>'
    return output

# Load test configuration settings
question_list = test_config['question_list']
series_number = test_config['series_number']
book_numbers = test_config['book_numbers']

results = []

# Iterate over each question and get answers for each book number
for question in tqdm(question_list, desc='Getting responses to questions'):
    question_result = {
        'question': question, 
        'book_numbers': book_numbers, 
        'answers': []
    }
    for book in book_numbers:
        answer, context = get_answer(question, series_number, book, chapter_number=99)
        question_result['answers'].append((answer, context))
    results.append(question_result)

# Write the results to an HTML file
with open(results_path, 'w') as file:
    file.write(get_html_results(results))

exit(0)