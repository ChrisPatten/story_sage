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
   Embed your book series data into ChromaDB.

## Preparing Book Data

### Series Configuration

The series.yml file is used to configure the book series data for Story Sage. Each series is defined with a unique series_id, series_name, and series_metadata_name. Each series contains a list of books, where each book has its own metadata.

#### Example Series Configuration

```yaml
- series_id: 2
  series_name: 'Harry Potter'
  series_metadata_name: 'harry_potter'
  entity_settings:
    names_to_skip:
      - 'light'
    person_titles:
      - 'miss'
      - 'professor'
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
- `entity_settings`: Settings for entity extraction. This section contains the following keys:
  - `names_to_skip`: A list of names to skip during entity extraction.
  - `person_titles`: A list of person titles to consider during entity extraction.
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