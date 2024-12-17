import pandas as pd
import pickle
import glob
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List
import yaml


def get_character_set(series_name: str, book_name: str, book_number: str) -> pd.DataFrame:
    """Extract and process character names from a single book's entity data.
    
    Args:
        series_name (str): Name of the book series.
        book_name (str): Name of the specific book.
        book_number (str): Book's number in the series.
    
    Returns:
        pd.DataFrame: DataFrame containing processed character information.
    """
    min_len = 3  # Minimum number of characters in a character name to consider
    df_entities = pd.read_csv(f'./booknlp/output/{series_name}/{book_name}.entities', sep='\t')
    df_entities = df_entities[(df_entities['cat'] == 'PER') & (df_entities['prop'] == 'PROP')]
    df_entities['text'] = df_entities['text'].str.lower()
    df_entities = df_entities.rename(columns={'COREF': 'character_id', 'text': 'characters'})
    df_entities['character_id'] = book_number + '_' + df_entities['character_id'].astype(str)
    df_entities = df_entities.groupby('character_id')['characters'].apply(lambda x: list(set(x))).reset_index()
    df_entities = df_entities[df_entities['characters'].apply(lambda x: len(x) >= min_len)]
    return df_entities

def collect_characters(series_name: str, book_titles: list) -> pd.DataFrame:
    """Collect and combine character data from multiple books in a series.
    
    Args:
        series_name (str): Name of the book series.
        book_titles (list): List of book titles to process.
    
    Returns:
        pd.DataFrame: DataFrame containing characters from all books.
    """
    result_df = pd.DataFrame(columns=['book_title', 'character_id', 'characters'])
    for book_title in book_titles:
        print(f'Processing {book_title}')
        book_number = book_title.split('_')[0]
        df = get_character_set(series_name, book_title, book_number)
        print(f'Found {len(df)} characters')
        df['book_title'] = book_title
        result_df = pd.concat([result_df, df])
    return result_df

def merge_characters(df1, df2, threshold=0.8):
    """Merge character lists using semantic similarity.
    
    Utilizes sentence transformers to compute embeddings and merges character mentions
    that likely refer to the same character across different books.
    
    Args:
        df1 (pd.DataFrame): First DataFrame containing character data.
        df2 (pd.DataFrame): Second DataFrame to merge with.
        threshold (float, optional): Similarity threshold for merging. Defaults to 0.8.
    
    Returns:
        dict: Dictionary of merged character sets.
    """
    merged_characters = {}
    df2_merged_ids = set()
    similarity_scores = []
    
    # Encode characters in each dataframe using the SentenceTransformer model
    df1['embeddings'] = df1['characters'].apply(lambda x: model.encode(x))
    df2['embeddings'] = df2['characters'].apply(lambda x: model.encode(x))

    for idx1, row1 in df1.iterrows():
        characters1 = row1['characters']
        embeddings1 = np.mean(row1['embeddings'], axis=0)
        primary_character_id = row1['character_id']

        merged_characters[primary_character_id] = set(characters1)

        for idx2, row2 in df2.iterrows():
            characters2 = row2['characters']
            embeddings2 = np.mean(row2['embeddings'], axis=0)

            # Calculate cosine similarity between character embeddings
            similarity_matrix = cosine_similarity(embeddings1.reshape(1, -1), embeddings2.reshape(1, -1))
            similarity_score = similarity_matrix[0][0]
            similarity_scores.append(similarity_score)

            if similarity_score > threshold:
                merged_characters[primary_character_id].update(characters2)
                df2_merged_ids.update(row2['character_id'])

    # Add characters from the second dataframe that were not merged
    for idx, row in df2.iterrows():
        if row['character_id'] not in df2_merged_ids:
            merged_characters[row['character_id']] = set(row['characters'])

    return merged_characters

def process_series(series_data: List[dict], series_name: str) -> pd.DataFrame:
    """Process an entire book series to identify and merge character mentions.
    
    Args:
        series_data (List[dict]): List of dictionaries containing series information.
        series_name (str): Name of the series to process.
    
    Returns:
        pd.DataFrame: DataFrame with merged character information for the series.
    """
    def get_book_titles(series_data: List[dict], series_name: str) -> List[str]:
        """Helper function to extract book titles from series data."""
        books = [series['books'] for series in series_data if series['series_metadata_name'] == series_name]
        return [book['book_metadata_name'] for book in books[0]]
    
    book_titles = get_book_titles(series_data, series_name)
    all_characters = collect_characters(series_name, book_titles)
    
    merged_characters = all_characters[all_characters['book_title'] == book_titles[0]].copy()

    for book_title in book_titles[1:]:
        print(f'Merging characters from {book_title}')
        next_book_characters = all_characters[all_characters['book_title'] == book_title].copy()
        merged_characters_dict = merge_characters(merged_characters, next_book_characters)
        merged_characters = pd.DataFrame(
            [(k, list(v)) for k, v in merged_characters_dict.items()],
            columns=['character_id', 'characters']
        )

    return merged_characters

def main():
    """Main execution function for the character processing pipeline.
    
    Steps:
        1. Initializes the SentenceTransformer model.
        2. Loads series configuration from a YAML file.
        3. Processes each series and saves merged character data.
    """
    # Initialize the SentenceTransformer model for semantic similarity
    global model
    print('Loading SentenceTransformer model')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model = model.to('mps')  # Utilize Metal Performance Shaders for acceleration on Mac
    print('Model loaded')

    # Load series configuration from the YAML file
    with open('series.yml', 'r') as file:
        series_data = yaml.safe_load(file)
    
    # Iterate through each series and process character data
    for series in series_data:
        merged_characters = process_series(series_data, series['series_metadata_name'])

        # Convert the merged characters to a dictionary and save using pickle
        merged_characters_dict = merged_characters.set_index('character_id')['characters'].to_dict()
        pickle.dump(merged_characters_dict, open(f"./characters/{series['series_metadata_name']}_characters.pkl", 'wb'))

# Execute the main function when the script is run directly
main()
