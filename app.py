from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin
from story_sage.story_sage import StorySage
import yaml
import pickle
import glob
import os
import re

app = Flask(__name__)
CORS(app)

# Dummy graph object with an invoke method
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

api_key = config['OPENAI_API_KEY']
chroma_path = config['CHROMA_PATH']
chroma_collection = config['CHROMA_COLLECTION']

# Load series.yml to create a mapping from series_metadata_name to series_id
with open('series.yml', 'r') as file:
    series_list = yaml.safe_load(file)
metadata_to_id = {series['series_metadata_name']: series['series_id'] for series in series_list}

# Load all character dictionaries and merge them using the metadata_to_id mapping
character_dict = {}
for filepath in glob.glob('./characters/*_characters.pkl'):
    with open(filepath, 'rb') as f:
        series_characters = pickle.load(f)
        # Extract series_metadata_name from filename
        filename = os.path.basename(filepath)
        match = re.match(r'(.+)_characters\.pkl', filename)
        if match:
            series_metadata_name = match.group(1)
            series_id = metadata_to_id.get(series_metadata_name)
            if series_id is not None:
                character_dict[series_id] = series_characters
            else:
                print(f'Warning: No series_id found for series_metadata_name "{series_metadata_name}"')
        else:
            print(f'Warning: Filename "{filename}" does not match the expected pattern.')

story_sage = StorySage(
    api_key=api_key,
    chroma_path=chroma_path,
    chroma_collection_name=chroma_collection,
    character_dict=character_dict,
    n_chunks=10
)

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/invoke', methods=['POST'])
@cross_origin()
def invoke_story_sage():
    data = request.get_json()
    required_keys = ['question', 'book_number', 'chapter_number', 'series_id']
    if not all(key in data for key in required_keys):
        return jsonify({'error': f'Missing parameter! Request must include {", ".join(required_keys)}'}), 400
    
    # Lookup series_name based on series_id
    series_id = data.get('series_id')
    series_entry = next((s for s in series_list if s['series_id'] == series_id), None)
    if series_entry:
        data['series_name'] = series_entry['series_metadata_name']
    else:
        return jsonify({'error': f'Invalid series_id: {series_id}'}), 400
    
    data.pop('series_id', None)

    result, context = story_sage.invoke(**data)
    return jsonify(result)

@app.route('/invoke', methods=['GET'])
@cross_origin()
def get_series():
    return jsonify(series_list)

if __name__ == '__main__':
    app.run(port=5010, debug=True)