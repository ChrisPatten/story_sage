from booknlp.booknlp import BookNLP
import pandas as pd
import pickle

book_name = '06_lord_of_chaos'

model_params={
		"pipeline":"entity,quote,supersense,event,coref", 
		"model":"big"
	}

model_params={
		"pipeline":"entity", 
		"model":"big"
	}

#model_params["pipeline"]="entity"
	
booknlp=BookNLP("en", model_params)

# Input file to process
input_file=f"../books/{book_name}.txt"

# Output directory to store resulting files in
output_directory="./output/"

# File within this directory will be named ${book_id}.entities, ${book_id}.tokens, etc.

booknlp.process(input_file, output_directory, book_name)

df_entities = pd.read_csv(f'./output/{book_name}.entities', sep='\t')
character_set = df_entities[(df_entities['cat'] == 'PER') & (df_entities['prop'] == 'PROP')] \
  .groupby('COREF')['text'].unique().drop_duplicates()
filtered_character_set = character_set[character_set.apply(lambda x: len(x) >= 4)]

character_dict = {name: idx for idx, names in filtered_character_set.items() for name in names}
with open(f'{book_name}_characters.pkl', 'wb') as f:
  pickle.dump(character_dict, f)