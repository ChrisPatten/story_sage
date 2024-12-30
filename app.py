"""
app.py

Provides the Flask application for the StorySage project, including RESTful endpoints
for user queries and feedback, as well as a web interface.

Example usage:
    $ python app.py

After running:
    1. Visit http://localhost:5010/ in your browser to view the web interface.
    2. Send POST requests to /invoke to query the StorySage engine.

Example results:
    - Renders the index page at '/'.
    - The '/invoke' endpoint returns JSON with an answer, context, and request_id.
    - The '/feedback' endpoint accepts and logs user feedback.

Note:
    - Check config.yml, series.yml, and entities.json for correct setup.
    - Install all dependencies (Flask, etc.) before running.
"""

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin
from story_sage.story_sage import StorySage
import yaml
import os
import json
import logging
from logging.handlers import TimedRotatingFileHandler
import warnings

CONFIG_PATH = './config.yml'



warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for all routes

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG for detailed output

# Set up logging formatter and handler
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
os.makedirs('logs', exist_ok=True)  # Create logs directory if it doesn't exist
log_handler = TimedRotatingFileHandler('logs/story_sage.log', when='midnight', backupCount=30)
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)

# Configure feedback logging
feedback_logger = logging.getLogger('feedback')
feedback_logger.setLevel(logging.INFO)
feedback_handler = TimedRotatingFileHandler('logs/feedback.log', when='midnight', backupCount=30)
feedback_handler.setFormatter(log_formatter)
feedback_logger.addHandler(feedback_handler)

logger.debug('Loading config and data files')
try:
    # Load configuration and data files
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
        logger.debug(f'Loaded {CONFIG_PATH}')
    with open(config['SERIES_PATH'], 'r') as file:
        series_list = yaml.safe_load(file)
        logger.debug(f'LOADED {config["SERIES_PATH"]}')
except Exception as e:
    # Log any errors that occur during the loading of configuration files
    logger.error(f"Error loading configuration files: {e}")
    raise
    
# Extract configuration settings
api_key = config['OPENAI_API_KEY']
chroma_path = config['CHROMA_PATH']
chroma_collection = config['CHROMA_COLLECTION']

def collect_entities(series_metadata_names):
    entities_dict = {}
    for name in series_metadata_names:
        try:
            if not os.path.exists(f'./entities/{name}/entities.json'):
                logger.warning(f'Entities file for {name} does not exist.')
                continue
            with open(f'./entities/{name}/entities.json', 'r') as file:
                entities_data = json.load(file)
                entities_dict[name] = entities_data
                logger.debug(f'Loaded entities for {name}')
        except Exception as e:
            logger.error(f"Error loading entities for {name}: {e}")
    return entities_dict

# Collect series metadata names from the series list
series_metadata_names = [series['series_metadata_name'] for series in series_list]

# Load the entity files using the collect_entities function
with open(config['ENTITIES_PATH'], 'r') as file:
    entities_dict = json.load(file)
    logger.debug(f'Loaded {config["ENTITIES_PATH"]}')

# Initialize the StorySage engine with the provided configurations
story_sage = StorySage(
    api_key=api_key,
    chroma_path=chroma_path,
    chroma_collection_name=chroma_collection,
    entities_dict=entities_dict,
    series_list=series_list,
    n_chunks=15  # Number of text chunks to process
)

@app.route('/')
def index():
    """Renders the main index page.

    Returns:
        Response: An HTML page rendered from the index template.
    """
    return render_template('./index.html')

@app.route('/invoke', methods=['POST'])
@cross_origin()
def invoke_story_sage():
    """Handles POST requests to invoke the StorySage engine.

    Expects a JSON payload with:
        question (str): The question to ask.
        book_number (int): The book number for context.
        chapter_number (int): The chapter number for context.
        series_id (int): The series ID for context.

    Returns:
        Response: JSON containing the result, context, and request_id.

    Example:
        curl -X POST http://localhost:5010/invoke -H "Content-Type: application/json" -d '{
            "question": "Who is the Half-Blood Prince?",
            "book_number": 6,
            "chapter_number": 14,
            "series_id": 2
        }'
    """
    data = request.get_json()
    required_keys = ['question', 'book_number', 'chapter_number', 'series_id']
    if not all(key in data for key in required_keys):
        # Return an error if any required parameter is missing
        return jsonify({'error': f'Missing parameter! Request must include {", ".join(required_keys)}'}), 400

    try:
        # Invoke the StorySage engine with the provided parameters
        result, context, request_id = story_sage.invoke(**data)
        return jsonify({'result': result, 'context': context, 'request_id': request_id})
    except Exception as e:
        # Log the error and return a server error response
        logger.error(f"Error invoking StorySage: {e}")
        return jsonify({'error': 'Internal server error.'}), 500

@app.route('/invoke', methods=['GET'])
@cross_origin()
def get_series():
    """Handles GET requests to retrieve available series information.

    Returns:
        Response: JSON list of series with their metadata.

    Example:
        curl http://localhost:5010/invoke
        [
           {
               "series_id": 1,
               "series_name": "Title",
               ...
           },
           ...
        ]
    """
    return jsonify(series_list)

@app.route('/feedback', methods=['POST'])
@cross_origin()
def feedback():
    """Handles user feedback submission via POST requests.

    Expects a JSON payload with:
        request_id (str): The request identifier.
        feedback (str): The user's comments or evaluation.
        type (str): The category or type of feedback.

    Returns:
        Response: A JSON success message or an error response.

    Example:
        curl -X POST http://localhost:5010/feedback -H "Content-Type: application/json" -d '{
            "request_id": "123e4567-e89b-12d3-a456-426614174000",
            "feedback": "Great answer!",
            "type": "praise"
        }'
    """
    data = request.get_json()
    required_keys = ['request_id', 'feedback', 'type']
    if not all(key in data for key in required_keys):
        # Return an error if any required parameter is missing
        return jsonify({'error': f'Missing parameter! Request must include {", ".join(required_keys)}'}), 400

    try:
        # Process the feedback here (e.g., save to a database or log)
        feedback_logger.info(f"Feedback received for request {data['request_id']}|{data['feedback']}|{data['type']}")
        return jsonify({'message': 'Feedback received.'})
    except Exception as e:
        # Log the error and return a server error response
        logger.error(f"Error processing feedback: {e}")
        return jsonify({'error': 'Internal server error.'}), 500

if __name__ == '__main__':
    # Run the Flask application on port 5010 with debugging enabled
    app.run(port=5010, debug=True)