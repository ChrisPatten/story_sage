"""
app.py

This module sets up a Flask web application for the StorySage project.

It provides RESTful API endpoints for invoking the StorySage engine,
handling user feedback, and serving the main web interface.

Example Usage:
    To run the application, execute the following command:
    $ python app.py

    Access the web interface by navigating to:
    http://localhost:5010/

    Example API call to invoke the StorySage engine:
    ```bash
    curl -X POST http://localhost:5010/invoke -H "Content-Type: application/json" -d '{
        "question": "What is the significance of the Mirror of Erised?",
        "book_number": 1,
        "chapter_number": 12,
        "series_id": 2
    }'
    ```

Example Results:
    - The application renders the main page at '/'.
    - The '/invoke' endpoint processes user questions and returns answers.
    - The '/feedback' endpoint accepts user feedback on the responses.

Note:
    - Ensure 'config.yml', 'series.yml', and 'entities.json' are properly configured.
    - Dependencies must be installed, including Flask and related packages.
"""

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin
from story_sage.story_sage import StorySage
import yaml
import pickle
import glob
import os
import re
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

try:
    # Load configuration and data files
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
    with open(config['SERIES_PATH'], 'r') as file:
        series_list = yaml.safe_load(file)
    with open(config['ENTITIES_PATH'], 'r') as file:
        entities = yaml.safe_load(file)
except Exception as e:
    # Log any errors that occur during the loading of configuration files
    logger.error(f"Error loading configuration files: {e}")
    raise

# Extract configuration settings
api_key = config['OPENAI_API_KEY']
chroma_path = config['CHROMA_PATH']
chroma_collection = config['CHROMA_COLLECTION']

# Initialize the StorySage engine with the provided configurations
story_sage = StorySage(
    api_key=api_key,
    chroma_path=chroma_path,
    chroma_collection_name=chroma_collection,
    entities=entities,
    series_yml_path='series.yml',
    n_chunks=10  # Number of text chunks to process
)

@app.route('/')
def index():
    """
    Render the main index page.

    Returns:
        A rendered HTML template for the index page.
    """
    return render_template('./index.html')

@app.route('/invoke', methods=['POST'])
@cross_origin()
def invoke_story_sage():
    """
    Handle POST requests to invoke the StorySage engine.

    Expected JSON payload:
        {
            "question": str,
            "book_number": int,
            "chapter_number": int,
            "series_id": int
        }

    Returns:
        A JSON response containing the result, context, and request_id.

    Example Request:
        curl -X POST http://localhost:5010/invoke -H "Content-Type: application/json" -d '{
            "question": "Who is the Half-Blood Prince?",
            "book_number": 6,
            "chapter_number": 14,
            "series_id": 2
        }'

    Example Response:
        {
            "result": "The Half-Blood Prince is revealed to be Severus Snape.",
            "context": "...",
            "request_id": "unique_request_id"
        }
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
    """
    Handle GET requests to retrieve available series information.

    Returns:
        A JSON response containing the list of series.

    Example Request:
        curl http://localhost:5010/invoke

    Example Response:
        [
            {
                "series_id": 1,
                "series_name": "The Lord of the Rings",
                "series_metadata_name": "lord_of_the_rings",
                ...
            },
            ...
        ]
    """
    return jsonify(series_list)

@app.route('/feedback', methods=['POST'])
@cross_origin()
def feedback():
    """
    Handle POST requests to submit user feedback.

    Expected JSON payload:
        {
            "request_id": str,
            "feedback": str
        }

    Returns:
        A JSON response confirming receipt of the feedback.

    Example Request:
        curl -X POST http://localhost:5010/feedback -H "Content-Type: application/json" -d '{
            "request_id": "unique_request_id",
            "feedback": "The answer was very helpful!"
        }'

    Example Response:
        {
            "message": "Feedback received."
        }
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