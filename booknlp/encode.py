from booknlp.booknlp import BookNLP

model_params={
		"pipeline":"entity,quote,supersense,event,coref", 
		"model":"big"
	}

#model_params["pipeline"]="entity"
	
booknlp=BookNLP("en", model_params)

# Input file to process
input_file="../books/01_the_eye_of_the_world.txt"

# Output directory to store resulting files in
output_directory="./output/"

# File within this directory will be named ${book_id}.entities, ${book_id}.tokens, etc.
book_id="the_eye_of_the_world"

booknlp.process(input_file, output_directory, book_id)