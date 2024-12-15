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
merged_characters_path = f'./characters/{chroma_collection}_characters.pkl'


with open(merged_characters_path, 'rb') as f:
    character_dict = pickle.load(f)

story_sage = StorySage(api_key=api_key, chroma_path=chroma_path, 
                       chroma_collection_name=chroma_collection,
                       character_dict=character_dict, n_chunks=15)

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

@app.route('/invoke', methods=['GET'])
def get_series():
    series = [
        {
            'series_id': 2,
            'series_name': 'Harry Potter',
            'books': [
                {
                    'number_in_series': 1,
                    'title': "Harry Potter and the Sorcerer's Stone",
                    'number_of_chapters': 17
                },
                {
                    'number_in_series': 2,
                    'title': 'Harry Potter and the Chamber of Secrets',
                    'number_of_chapters': 18
                },
                {
                    'number_in_series': 3,
                    'title': 'Harry Potter and the Prisoner of Azkaban',
                    'number_of_chapters': 22
                },
                {
                    'number_in_series': 4,
                    'title': 'Harry Potter and the Goblet of Fire',
                    'number_of_chapters': 37
                },
                {
                    'number_in_series': 5,
                    'title': 'Harry Potter and the Order of the Phoenix',
                    'number_of_chapters': 38
                },
                {
                    'number_in_series': 6,
                    'title': 'Harry Potter and the Half-Blood Prince',
                    'number_of_chapters': 30
                },
                {
                    'number_in_series': 7,
                    'title': 'Harry Potter and the Deathly Hallows',
                    'number_of_chapters': 36
                }
            ]
        },
        {
            'series_id': 1,
            'series_name': 'Sherlock Holmes',
            'books': [
                {
                    'number_in_series': 1,
                    'title': 'A Study in Scarlet',
                    'number_of_chapters': 14
                },
                {
                    'number_in_series': 2,
                    'title': 'The Sign of the Four',
                    'number_of_chapters': 12
                },
                {
                    'number_in_series': 3,
                    'title': 'The Hound of the Baskervilles',
                    'number_of_chapters': 15
                },
                {
                    'number_in_series': 4,
                    'title': 'The Valley of Fear',
                    'number_of_chapters': 14
                }
            ]
        }
    ]
    return jsonify(series)

if __name__ == '__main__':
    app.run(port=5010)