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

warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)
CORS(app)

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Default level, can be made configurable

log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
os.makedirs('logs', exist_ok=True)
log_handler = TimedRotatingFileHandler('logs/story_sage.log', when='midnight', backupCount=30)
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)

try:
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    with open('series.yml', 'r') as file:
        series_list = yaml.safe_load(file)
    with open('entities.json', 'r') as file:
        entities = yaml.safe_load(file)
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise

api_key = config['OPENAI_API_KEY']
chroma_path = config['CHROMA_PATH']
chroma_collection = config['CHROMA_COLLECTION']

story_sage = StorySage(
    api_key=api_key,
    chroma_path=chroma_path,
    chroma_collection_name=chroma_collection,
    entities=entities,
    series_yml_path='series.yml',
    n_chunks=10
)

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/invoke', methods=['POST'])
@cross_origin()
def invoke_story_sage():
    logger.info("Received /invoke POST request.")
    data = request.get_json()
    required_keys = ['question', 'book_number', 'chapter_number', 'series_id']
    if not all(key in data for key in required_keys):
        logger.warning("Missing parameters in /invoke request.")
        return jsonify({'error': f'Missing parameter! Request must include {", ".join(required_keys)}'}), 400

    try:
        result, context = story_sage.invoke(**data)
        logger.info("Successfully invoked StorySage.")
        logger.debug(f"Result: {result}, Context: {context}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error invoking StorySage: {e}")
        return jsonify({'error': 'Internal server error.'}), 500

@app.route('/invoke', methods=['GET'])
@cross_origin()
def get_series():
    logger.info("Received /invoke GET request.")
    return jsonify(series_list)

if __name__ == '__main__':
    app.run(port=5010, debug=True)