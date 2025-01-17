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

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS, cross_origin
from story_sage import StorySage, StorySageConversation, StorySageConfig
import yaml
import os
import logging
from logging.handlers import TimedRotatingFileHandler
import warnings

CONFIG_PATH = './config.yml'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizers parallelism to avoid warnings


warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for all routes


def get_loggers_by_prefix(prefix: str) -> list[logging.Logger]:
    """Retrieve all loggers that start with the given prefix.

    Args:
        prefix (str): The prefix to filter loggers by.

    Returns:
        list[logging.Logger]: List of loggers matching the prefix.
    """
    loggers = []
    for name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and name.startswith(prefix):
            loggers.append(logger)
    return loggers


# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
os.makedirs('logs', exist_ok=True)  # Create logs directory if it doesn't exist
log_handler = TimedRotatingFileHandler('logs/story_sage.log', when='midnight', backupCount=30)
log_handler.setFormatter(log_formatter)

# Set up the root logger to only show messages from story_sage
root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)  # Set higher level to suppress other package logs

# Configure the story_sage logger
logger = logging.getLogger('story_sage')
logger.addHandler(log_handler)
logger.setLevel(logging.DEBUG)

# Configure feedback logging
feedback_logger = logging.getLogger('story_sage.feedback')
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
    STORY_SAGE_CONFIG = StorySageConfig.from_config(config)
except Exception as e:
    logger.error(f"Error loading configuration: {e}", exc_info=True)
    raise

# Initialize the StorySage engine with the provided configurations
story_sage = StorySage(
    config=STORY_SAGE_CONFIG,
    log_level=logging.DEBUG
)

loggers_with_prefix = get_loggers_by_prefix('story_sage')
for component_logger in loggers_with_prefix:
    component_logger.setLevel(logging.DEBUG)

loggers_with_prefix = get_loggers_by_prefix('openai')
for component_logger in loggers_with_prefix:
    component_logger.setLevel(logging.DEBUG)

@app.route('/')
def index():
    """Renders the main index page.

    Returns:
        Response: An HTML page rendered from the index template.
    """
    logger.debug("Rendering index page")
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
        logger.warning("Missing parameter in request: %s", data)
        return jsonify({'error': f'Missing parameter! Request must include {", ".join(required_keys)}'}), 400
    if 'conversation_id' in data and data['conversation_id']:
        logger.debug(f'Conversation ID: {data["conversation_id"]}')
        conversation = StorySageConversation(conversation_id=data['conversation_id'], 
                                             redis=STORY_SAGE_CONFIG.redis_instance, 
                                             redis_ex=STORY_SAGE_CONFIG.redis_ex)
    else:
        logger.debug('No conversation ID provided')
        conversation = StorySageConversation(redis=STORY_SAGE_CONFIG.redis_instance, 
                                             redis_ex=STORY_SAGE_CONFIG.redis_ex)

    try:
        # Invoke the StorySage engine with the provided parameters
        payload = {
            'question': data['question'],
            'book_number': data['book_number'],
            'chapter_number': data['chapter_number'],
            'series_id': data['series_id'],
            'conversation': conversation
        }
        logger.info("Invoking StorySage with payload: %s", payload)
        result, context, request_id, entities = story_sage.invoke(**payload)
        try:
            conversation.add_turn(question=data['question'], detected_entities=entities, 
                                context=context, response=result)
        except Exception as e:
            logger.error(f"Error adding turn to conversation: {e}")
            raise e
        logger.info("Successfully processed request with request_id: %s", request_id)
        return jsonify({'result': result, 'context': context, 'request_id': 'request_id', 'conversation_id': str(conversation.conversation_id)})
    except Exception as e:
        # Log the error and return a server error response
        logger.error(f"Error invoking StorySage: {e}", exc_info=True)
        return jsonify({'error': "I'm sorry, but I experienced an error trying to answer your question. Please let my creator know!"}), 500

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
    logger.debug("Retrieving series information")
    return jsonify(STORY_SAGE_CONFIG.get_series_json())

@app.route('/feedback', methods=['POST'])
@cross_origin()
def feedback():
    """Handles user feedback submission via POST requests.

    Expects a JSON payload with:
        conversation_id (str): The request identifier.
        feedback (str): The user's comments or evaluation.
        type (str): The category or type of feedback.

    Returns:
        Response: A JSON success message or an error response.

    Example:
        curl -X POST http://localhost:5010/feedback -H "Content-Type: application/json" -d '{
            "feedback": "Great answer!",
            "type": "positive",
            "conversation_id": "123e4567-e89b-12d3-a456-426614174000"
        }'
    """
    data = request.get_json()
    required_keys = ['conversation_id', 'feedback', 'type']
    if not all(key in data for key in required_keys):
        logger.warning("Missing parameter in feedback request: %s", data)
        return jsonify({'error': f'Missing parameter! Request must include {", ".join(required_keys)}'}), 400

    try:
        conversation = StorySageConversation(conversation_id=data['conversation_id'], redis=STORY_SAGE_CONFIG.redis_instance, redis_ex=STORY_SAGE_CONFIG.redis_ex)
        # Process the feedback here (e.g., save to a database or log)
        feedback_logger.info(f"Feedback received for request {data['conversation_id']}|{data['feedback']}|{data['type']}")
        feedback_logger.info(conversation.get_log())
        logger.info("Feedback successfully processed for conversation_id: %s", data['conversation_id'])
        return jsonify({'message': 'Feedback received.'})
    except Exception as e:
        # Log the error and return a server error response
        logger.error(f"Error processing feedback: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error.'}), 500

@app.route('/images/<filename>')
def serve_image(filename):
    """Serves image files from the 'images' directory.

    Args:
        filename (str): The name of the image file to serve.

    Returns:
        Response: The requested image file.
    """
    logger.debug("Serving image: %s", filename)
    return send_from_directory('images', filename)

if __name__ == '__main__':
    # Run the Flask application on port 5010 with debugging enabled
    logger.info("Starting Flask application on port 5010")
    app.run(port=5010, debug=True)