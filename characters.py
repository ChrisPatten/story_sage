import pandas as pd
import pickle
import glob
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List

SERIES_NAME = 'harry_potter'
BOOK_TITLES = ['01_the_sourcerers_stone', '02_the_chamber_of_secrets', '03_the_prisoner_of_azkaban', '04_the_goblet_of_fire', '05_the_order_of_the_phoenix', '06_the_half_blood_prince', '07_the_deathly_hallows']

def get_character_set(book_name: str, book_number: str) -> pd.DataFrame:
    min_len = 3
    df_entities = pd.read_csv(f'./booknlp/output/{SERIES_NAME}/{book_name}.entities', sep='\t')
    df_entities = df_entities[(df_entities['cat'] == 'PER') & (df_entities['prop'] == 'PROP')]
    df_entities['text'] = df_entities['text'].str.lower()
    df_entities = df_entities.rename(columns={'COREF': 'character_id', 'text': 'characters'})
    df_entities['character_id'] = book_number + '_' + df_entities['character_id'].astype(str)
    df_entities = df_entities.groupby('character_id')['characters'].apply(lambda x: list(set(x))).reset_index()
    df_entities = df_entities[df_entities['characters'].apply(lambda x: len(x) >= min_len)]
    return df_entities

def collect_characters(book_titles: list) -> pd.DataFrame:
    result_df = pd.DataFrame(columns=['book_title', 'character_id', 'characters'])
    for book_title in book_titles:
        print(f'Processing {book_title}')
        book_number = book_title.split('_')[0]
        df = get_character_set(book_title, book_number)
        print(f'Found {len(df)} characters')
        df['book_title'] = book_title
        result_df = pd.concat([result_df, df])
    return result_df

def merge_characters(df1, df2, threshold=0.8):
    merged_characters = {}
    df2_merged_ids = set()
    similarity_scores = []
    
    # Encode characters in each dataframe
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

def main():
    # Initialize the SentenceTransformer model
    global model
    print('Loading SentenceTransformer model')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model = model.to('mps')
    print('Model loaded')

    # Define book titles
    book_titles = BOOK_TITLES

    # Collect and merge characters
    print('Collecting characters')
    all_characters = collect_characters(book_titles)
    
    # Initialize with first book's characters
    merged_characters = all_characters[all_characters['book_title'] == book_titles[0]].copy()

    # Merge with remaining books
    for book_title in book_titles[1:]:
        print(f'Merging characters from {book_title}')
        next_book_characters = all_characters[all_characters['book_title'] == book_title].copy()
        merged_characters_dict = merge_characters(merged_characters, next_book_characters)
        merged_characters = pd.DataFrame(
            [(k, list(v)) for k, v in merged_characters_dict.items()],
            columns=['character_id', 'characters']
        )

    # Convert to dictionary and save
    merged_characters_dict = merged_characters.set_index('character_id')['characters'].to_dict()
    pickle.dump(merged_characters_dict, open(f'./characters/{SERIES_NAME}_characters.pkl', 'wb'))

main()
