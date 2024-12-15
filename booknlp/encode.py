from booknlp.booknlp import BookNLP
import pandas as pd
import pickle
import glob

model_params={
		"pipeline":"entity", 
		"model":"big"
	}

#model_params["pipeline"]="entity"
	
booknlp=BookNLP("en", model_params)

def extract_characters(file, book_name, output_dir, char_output_dir):
    booknlp.process(file, output_dir, book_name)
    df_entities = pd.read_csv(f'{output_dir}/{book_name}.entities', sep='\t')
    character_set = df_entities[(df_entities['cat'] == 'PER') & (df_entities['prop'] == 'PROP')] \
        .groupby('COREF')['text'].unique().drop_duplicates()
    filtered_character_set = character_set[character_set.apply(lambda x: len(x) >= 4)]
    character_dict = {name: idx for idx, names in filtered_character_set.items() for name in names}
    with open(f'{char_output_dir}/{book_name}_characters.pkl', 'wb') as f:
        pickle.dump(character_dict, f)
# Output directory to store resulting files in


# File within this directory will be named ${book_id}.entities, ${book_id}.tokens, etc.

output_directory="./output/harry_potter"
char_output_dir="../characters/harry_potter"
for file in glob.glob('../books/harry_potter/*.txt'):
	print(f'Extracting characters from {file}')
	book_name = file.split('/')[-1].replace('.txt', '')
	extract_characters(file, book_name, output_directory, char_output_dir)