# Story Sage

![Story Sage Logo](story_sage.png)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Preparing Book Data](#preparing-book-data)
- [Usage](#usage)
  - [Basic Example](#basic-example)
  - [Advanced Configuration](#advanced-configuration)
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

### Component Breakdown

- **StorySageRetriever:** Handles the retrieval of relevant text chunks from the book based on user queries using ChromaDB.
- **StorySageChain:** Manages the generation of responses by processing retrieved information through a language model.
- **StorySageState:** Maintains the state of user interactions, including context and extracted entities.
- **ChromaDB:** Serves as the vector store for efficient storage and retrieval of text embeddings.
- **Language Model (LLM):** Generates human-like responses based on the provided context.

## Installation

### Prerequisites

- Python 3.8 or higher
- [ChromaDB](https://www.chromadb.com/docs/)
- [Sentence Transformers](https://www.sbert.net/)
- [LangChain](https://langchain.com/)
- [PyYAML](https://pyyaml.org/)

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/chrispatten/story_sage.git
   cd story_sage
   ```

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure ChromaDB**
   Ensure ChromaDB is installed and running. Configure the `chroma_path` and `chroma_collection_name` in the configuration files as needed.

5. **Prepare Series Data**
   Populate the `series.yml` file with your book series information.

6. **Embed in ChromaDB**
   Embed your book series data into ChromaDB (add info about final scrips here).

## Preparing Book Data

### Series Configuration

The series.yml file is used to configure the book series data for Story Sage. Each series is defined with a unique series_id, series_name, and series_metadata_name. Each series contains a list of books, where each book has its own metadata.

#### Example Series Configuration

```yaml
- series_id: 2
  series_name: 'Harry Potter'
  series_metadata_name: 'harry_potter'
  books:
    - number_in_series: 1
      title: "Harry Potter and the Sorcerer's Stone"
      book_metadata_name: '01_the_sourcerers_stone'
      number_of_chapters: 17
    - number_in_series: 2
      title: 'Harry Potter and the Chamber of Secrets'
      book_metadata_name: '02_the_chamber_of_secrets'
      number_of_chapters: 18
    # Additional books...
```

#### Keys

- `series_id`: A unique identifier for the series. This should be an integer.
- `series_name`: The name of the book series.
- `series_metadata_name`: A metadata-friendly name for the series, typically in lowercase and with underscores instead of spaces.
- `books`: A list of books in the series. Each book has the following keys:
- `number_in_series`: The position of the book in the series.
- `title`: The title of the book.
- `book_metadata_name`: A metadata-friendly name for the book, typically in lowercase and with underscores instead of spaces.
- `number_of_chapters`: The total number of chapters in the book.

### Book File Preparation

Each book in the series should be stored as a separate text file. The text file should be named using the `book_metadata_name` from the series.yml file. For example, the text file for the first book in the Harry Potter series would be named `01_the_sourcerers_stone.txt`.

Place books in a subdirectory titled with the `series_metadata_name` from the series.yml file. For example, the Harry Potter books would be stored in the directory named `./books/harry_potter`.

Strip out any non-essential content from the text files, such as table of contents, author notes, etc. The text should only contain the main content of the book.

### Chunking

#### Using the StorySageChunker Class

The chunking functionality has been encapsulated in the `StorySageChunker` class within the `story_sage` module. This class helps in splitting your book text into semantically coherent chunks.

1. **Import the Class**
   ```python
   from story_sage.utils.chunker import StorySageChunker
   ```

2. **Initialize the Chunker**
   ```python
   chunker = StorySageChunker(model_name='all-MiniLM-L6-v2')
   ```

3. **Process Text**
   ```python
   chunks = chunker.process_file(
       text=full_text,
       context_window=2,
       percentile_threshold=85,
       min_chunk_size=3
   )
   ```

4. **Configure Parameters**
   - `model_name`: Specify the sentence transformer model to use.
   - `context_window`: Number of sentences to include for context.
   - `percentile_threshold`: Threshold for identifying where to break chunks.
   - `min_chunk_size`: Minimum number of sentences per chunk.

5. **Verify the Output**
   After processing, you'll receive a list of text chunks that can be used for further processing like entity extraction or embedding.

### Entity Extraction

#### Running the Entity Extraction Script

To extract named entities from your book data, use the `entity_extractor.py` script located in the root directory. This process involves two main components: `entity_extractor.py` and `entity_processor.py`. Follow these steps:

1. **Ensure Dependencies are Installed**
   Make sure all required Python packages are installed. If not, install them using:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the Script**
   - **API Key:** Ensure your OpenAI API key is set in the `config.yml` file under `OPENAI_API_KEY`.
   - **Target Series and Book:**
     - Open `extract_entities.py` and set the `TARGET_SERIES_ID` and `TARGET_BOOK_NUMBER` to specify which series and book you want to process.
     - Ensure that the `series.yml` file contains the correct metadata for your target series and book.

3. **Prepare Book Chunks**
   - Run the `chunking.py` script to generate semantic chunks of your book if you haven't done so already.
   - Ensure the chunks are saved in the `./chunks/{series_metadata_name}/semantic_chunks/` directory.

4. **Run the Entity Extraction**
   Execute the `extract_entities.py` script to extract entities:
   ```bash
   python extract_entities.py
   ```
   This script will:
   - Load the semantic chunks of the specified book.
   - Use `StorySageEntityExtractor` to extract named entities from each chapter.
   - Save the extracted entities to `./entities/{series_metadata_name}/{book_metadata_name}.json`.

5. **Process Extracted Entities**
   The extracted entities are further processed using `entity_processor.py` to aggregate and group similar entities. This step ensures that entities like character names are consistently identified across the book.

6. **Verify the Output**
   After running, check the `./entities/{series_metadata_name}/` directory for the generated JSON file containing the extracted entities.

#### Parameters

The `entity_extractor.py` script has several configurable parameters:

- **API Key:** Set your OpenAI API key in the `config.yml` file.
- **Target Series ID and Book Number:** Modify `TARGET_SERIES_ID` and `TARGET_BOOK_NUMBER` in `extract_entities.py` to specify which book to process.
- **Token Limits and Cooldown:** Adjust `token_per_min_limit` and `cooldown_secs` in `entity_extractor.py` to manage API rate limits.

You can modify these parameters in the respective scripts to tailor the entity extraction process to your needs.

### Entity Preparation

#### Running the Entity Preparation Script

To prepare and consolidate extracted entities, use the `prepare_entities.py` script located in the root directory. Follow these steps:

1. **Ensure Dependencies are Installed**
   Make sure all required Python packages are installed. If not, install them using:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the Script**
   - **API Key:** Ensure your OpenAI API key is set in the `config.yml` file under `OPENAI_API_KEY`.
   - **Series IDs to Process:** Modify the `SERIES_IDS_TO_PROCESS` list in `prepare_entities.py` to include the series IDs you want to process.

3. **Run the Script**
   Execute the script using:
   ```bash
   python prepare_entities.py
   ```
   This script will:
   - Read entity JSON files from `./entities/<series_metadata_name>/`.
   - Clean and aggregate entities using `StorySageEntityProcessor`.
   - Generate a consolidated `entities.json` file with grouped and unique entities.

4. **Verify the Output**
   After running, check the `entities.json` file in the root directory for the consolidated and processed entities.

#### Parameters

The `prepare_entities.py` script has several configurable parameters:

- **Series IDs to Process:** Modify the `SERIES_IDS_TO_PROCESS` list to specify which series to process.
- **API Key:** Ensure the API key in `config.yml` is correct.
- **Other Configuration:** Adjust any additional settings as needed in the script.

You can modify these parameters in `prepare_entities.py` to tailor the entity preparation process to your needs.

### Embedding

#### Running the Embedding Script

To generate and store embeddings for your book data, use the `embedding.py` script located in the root directory. Follow these steps:

1. **Ensure Dependencies are Installed**
   Make sure all required Python packages are installed. If not, install them using:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the Script**
   - **Configuration File:** Ensure that the `config.yml` file is properly configured with the necessary settings, including `CHROMA_COLLECTION` and other relevant parameters.
   - **Series and Entity Data:** Ensure that `series.yml` and `entities.json` are present and correctly populated with your series and entity information.

3. **Run the Script**
   Execute the script using:
   ```bash
   python embedding.py
   ```
   This script will:
   - Load text chunks from `./chunks/{series_metadata_name}/semantic_chunks/`.
   - Generate embeddings using the SentenceTransformer model.
   - Store the embeddings in the ChromaDB vector store at the specified path (`./chroma_data`).

4. **Verify the Output**
   After running, verify that the embeddings have been successfully added to the ChromaDB vector store by checking the `./chroma_data` directory and ensuring that the `books_collection` contains the embedded documents.

#### Parameters

The `embedding.py` script has several configurable parameters:

- **Chroma Collection Name:** Ensure `CHROMA_COLLECTION` in `config.yml` matches the desired collection name in ChromaDB.
- **Model Name:** The SentenceTransformer model used for generating embeddings. You can modify the model in the script if needed.
- **Series to Process:** Modify the `series_to_process` list in the script to specify which series to embed.

You can modify these parameters in `embedding.py` and `config.yml` to tailor the embedding process to your needs.

## Usage

### Basic Example

```python
from story_sage import StorySage

# Initialize Story Sage
story_sage = StorySage(
    api_key='your-openai-api-key',
    chroma_path='./chroma_db',
    chroma_collection_name='books_collection',
    entities={'series': {...}},  # Your entities data
    series_yml_path='series.yml',
    n_chunks=5
)

# Ask a question
question = "What motivates the main character in Book 1?"
answer, context = story_sage.invoke(question)

print("Answer:", answer)
print("Context:", context)
```

### Advanced Configuration

Customize filters and extend functionalities by modifying the `story_sage.py` and related modules. Refer to the [Architecture](#architecture) section for guidance on extending components.

## Current Issues

 - request_id isn't being emitted in the logs that are captured on disk.

## To Do

- [ ] Update content processing to be better organized/parameterized (issue #3)

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