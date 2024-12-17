
import yaml
import os
import pickle
import openai

# Load the series YAML file
with open('series.yml', 'r') as file:
    series_data = yaml.safe_load(file)

# Initialize the result dictionary
result = {}

# Set your OpenAI API key
openai.api_key = 'YOUR_API_KEY'

# Iterate through each series
for series in series_data:
    series_id = str(series['series_id'])
    result[series_id] = {
        'series_metadata_name': series['series_metadata_name'],
        'books': []
    }
    # Iterate through each book in the series
    for book in series['books']:
        book_number = str(book['number_in_series'])
        book_entry = {
            book_number: {
                'book_name': book['title'],
                'chapters': {}
            }
        }
        book_metadata_name = book['book_metadata_name']
        chunks_dir = os.path.join('chunks', book_metadata_name)
        # Check if the chunks directory exists
        if not os.path.isdir(chunks_dir):
            continue
        # Iterate through each chunk file
        for chunk_file in os.listdir(chunks_dir):
            if chunk_file.endswith('.pkl'):
                chunk_path = os.path.join(chunks_dir, chunk_file)
                with open(chunk_path, 'rb') as cf:
                    chunk_data = pickle.load(cf)
                # Use OpenAI to extract characters from the chunk
                response = openai.Completion.create(
                    model='gpt-4o-mini',
                    prompt=f"Extract the character names from the following text:\n{chunk_data}\nNames:",
                    max_tokens=50,
                    temperature=0
                )
                characters = response['choices'][0]['text'].strip().split(', ')
                # Get chapter number from chunk file name
                chapter_number = os.path.splitext(chunk_file)[0]
                book_entry[book_number]['chapters'][chapter_number] = {
                    'characters': characters
                }
        result[series_id]['books'].append(book_entry)

# Optionally, save the result to a file
# with open('characters.json', 'w') as outfile:
#     json.dump(result, outfile, indent=4)