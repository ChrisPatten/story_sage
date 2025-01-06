# Story Sage

![Story Sage Logo](story_sage.png)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Preparing Book Data](#preparing-book-data)
- [Usage](#usage)
- [Current Issues](#current-issues)
- [Contributing](#contributing)
- [License](#license)

## Introduction

**Story Sage** is a tool that helps users interact with their books through natural conversation. It uses AI to provide relevant answers about book content while avoiding spoilers.

## Features

- **Interactive Q&A:** Question and answer system that preserves plot surprises
- **Semantic Search:** Uses advanced embedding models to understand and retrieve relevant information across book content.
- **Customizable Filters:** Filter responses based on book, chapter, or specific entities like characters and places.
- **Persistent Storage:** Stores and retrieves embeddings efficiently using ChromaDB.
- **Extensible Architecture:** Easily extendable components for additional functionalities.

## Architecture

Story Sage uses a modular architecture with Retrieval-Augmented Generation (RAG) and chain-of-thought logic to deliver accurate and context-aware responses.

```
+------------------+          +------------------+
|  User Interface  | <------> |    Story Sage    |
+------------------+          +------------------+
                                     |
                                     |
                                     v
                         +-------------------------+
                         |   Retrieval Module      |
                         | - StorySageRetriever    |
                         | - ChromaDB Integration  |
                         +-------------------------+
                                     |
                                     |
                                     v
                         +-------------------------+
                         |   Generation Module     |
                         | - StorySageChain        |
                         | - Language Model (LLM)  |
                         +-------------------------+
                                     |
                                     |
                                     v
                         +-------------------------+
                         |   State Management      |
                         | - StorySageState        |
                         +-------------------------+
```

### Major Components

#### StorySageRetriever

Handles the retrieval of relevant text chunks from the book based on user queries using ChromaDB. The basic pattern is:

1. The user's question is passed to the `first_pass_query` method.
1. The retriever builds up a where condition using the `get_where_filter` method to filter the ChromaDB query
1. The retriever queries ChromaDB with the question and filters and returns the relevant chunks containing the summarized versions of the chunks.
1. `StorySageChain` manages evaluating the summarized chunks to find the IDs that
are likely to be relevant to the question.
1. If IDs are returned, the chain fetches the full text of those chunks using the `get_by_ids` method and returns them to `StorySageChain`.
1. If no IDs are returned, the chain sends the question to the `retrieve_chunks` method to get the full text of the chunks and returns them to `StorySageChain`.

#### StorySageChain

Manages the generation of responses by processing retrieved information through a language model. There are several steps in the chain:

1. RouterFunction: Anayzes the question to determine if it's past or present tense and sets the sort order appropriately.
1. GetCharacters: Searches the question for references to any characters and adds the character ids to the state.
1. GetContextFilters: Builds a context filter object based on information in the state.
1. GetInitialContext: Retrieves the summarized chunks returned by StorySageRetriever.
1. IdentifyRelevantChunks: Evaluates the summarized chunks to find the IDs that are likely to be relevant to the question.
1. GetContextByIDs: Fetches the full text of the relevant chunks if any IDs were captured.
1. GetContext: Fetches the full text of chunks based on a naive semantic search of the summaries if no IDs were captured.
1. Generate: Generates a response using the language model and the context.

#### StorySageState

Maintains the state of user interactions, including context and extracted entities.

## Installation

### Prerequisites

- Python 3.11.4
- pyenv
- Redis
- [ChromaDB](https://www.chromadb.com/docs/)
- [Sentence Transformers](https://www.sbert.net/)
- [LangChain](https://langchain.com/)

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/chrispatten/story_sage.git
   cd story_sage
   ```

2. **Run Setup**
   ```bash
   make setup
   ```
   This will:
   - Install pyenv if needed
   - Set up Python 3.11.4
   - Create virtual environment
   - Install Redis if needed
   - Create default configuration files

3. **Configure the Application**
   Update the following configuration files:
   - `config.yml`: Set your OpenAI API key and other settings
   - `redis_config.conf`: Configure Redis settings if needed
   
   Example config.yml:
   ```yaml
   OPENAI_API_KEY: "your-api-key"
   CHROMA_PATH: './chroma_data'
   CHROMA_COLLECTION: 'story_sage'
   ENTITIES_PATH: './entities/entities.json'
   SERIES_PATH: './series_prod.yml'
   N_CHUNKS: 15
   REDIS_URL: 'redis://localhost:6379/0'
   REDIS_EXPIRE: 86400
   ```

## Running the Application

1. **Start Redis**
   ```bash
   make redis
   ```

2. **Run the Application**
   ```bash
   make app
   ```

## Preparing Book Data

### Series Configuration

Create a `series_prod.yml` file to configure your book series. Example structure:

```yaml
- series_id: 2
  series_name: 'Series Name'
  series_metadata_name: 'series_name'
  entity_settings:
    names_to_skip:
      - 'common_word'
    person_titles:
      - 'title1'
      - 'title2'
    base_characters:
      - name: 'Character Name'
        other_names:
          - 'alias1'
          - 'alias2'
  books:
    - number_in_series: 1
      title: 'Book Title'
      book_metadata_name: '01_book_name'
      number_of_chapters: 17
```

### Book File Preparation

Each book in the series should be stored as a separate text file. The text file should be named using the `book_metadata_name` from the series.yml file. For example, the text file for the first book in the Harry Potter series would be named `01_the_sourcerers_stone.txt`.

Place books in a subdirectory titled with the `series_metadata_name` from the series.yml file. For example, the Harry Potter books would be stored in the directory named `./books/harry_potter`.

Strip out any non-essential content from the text files, such as table of contents, author notes, etc. The text should only contain the main content of the book.

### Chunking

Chunk the book data into semantic chunks for efficient processing. Use the `create_chunks.py` script to generate these chunks. Follow these steps:

#### Using create_chunks.py
Use this script to split book text files into semantically coherent chunks:
1. Confirm your series and book files are organized in ./books/<series_name>/*.txt.
2. Update the SERIES_NAME variable in create_chunks.py to match your directory.
3. Run the script:
   ```bash
   python create_chunks.py
   ```
4. Confirm JSON files are generated in ./chunks/<series_name>/semantic_chunks/ for each chapter.

### Entity Extraction

#### Using extract_entities.py
Use this script to extract named entities from your semantic chunks:
1. Ensure the required text chunks are already generated in ./chunks/<series_name>/semantic_chunks/.
2. Open extract_entities.py and set TARGET_SERIES_ID to match the correct series_id in series.yml.
4. Run the script:
   ```bash
   python extract_entities.py
   ```
5. Check the ./entities/<series_name>/ directory for generated JSON files containing extracted entities.

### Embedding

#### Using embed_chunks.py
Use this script to embed your semantic chunks into the ChromaDB vector store:
1. Ensure you have successfully run `create_chunks.py` and `extract_entities.py`.
2. Open `embed_chunks.py` and set the `series_metadata_name` variable to match your series.
3. Verify that `entities.json` and `series_prod.yml` are correctly configured.
4. Run the script:
   ```bash
   python embed_chunks.py
   ```
5. Confirm that the embedded documents are stored in the `./chroma_data` directory by checking the ChromaDB collection.

## Usage

```python
from story_sage import StorySage

# Initialize Story Sage
story_sage = StorySage(
   api_key='your-openai-api-key',
   chroma_path='./chroma_db',
   chroma_collection_name='books_collection',
   entities_dict={'series': {...}},  # Your entities data
   series_list=[{'title': 'Series Title'}],  # Your series data
   n_chunks=5
)

# Ask a question
question = "What motivates the main character in Book 1?"
answer, context = story_sage.invoke(question)

print("Answer:", answer)
print("Context:", context)
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with clear messages.
4. Open a pull request detailing your changes.

For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the [MIT License](LICENSE.md).

It was created with the help of GitHub Copilot and [Connor Tyrell](https://github.com/angusmccloud/story-sage)