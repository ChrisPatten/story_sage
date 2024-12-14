from flask import Flask, jsonify, request, render_template
from story_sage.story_sage import StorySage
import yaml
import pickle

app = Flask(__name__)

# Dummy graph object with an invoke method
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

api_key = config['OPENAI_API_KEY']
chroma_path = config['CHROMA_PATH']
chroma_collection = config['CHROMA_COLLECTION']
merged_characters_path = config['MERGED_CHARACTERS_PATH']


with open(merged_characters_path, 'rb') as f:
    character_dict = pickle.load(f)

story_sage = StorySage(api_key=api_key, chroma_path=chroma_path, 
                       chroma_collection_name=chroma_collection,
                       character_dict=character_dict, n_chunks=10)

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/invoke', methods=['POST'])
def invoke_story_sage():
    data = request.get_json()
    required_keys = ['question', 'book_number', 'chapter_number', 'series_id']
    if not all(key in data for key in required_keys):
        return jsonify({'error': f'Missing parameter! Request must include {", ".join(required_keys)}'}), 400
    result, context = story_sage.invoke(**data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)