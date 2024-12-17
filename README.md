# Story Sage

![Story Sage Logo](story_sage.png)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Example](#basic-example)
  - [Advanced Configuration](#advanced-configuration)
- [Current Issues](#current-issues)
- [Contributing](#contributing)
- [License](#license)

## Introduction

**Story Sage** is a cutting-edge application that leverages Retrieval-Augmented Generation (RAG) and chain-of-thought logic to allow users to interact with their books seamlessly. Designed to provide insightful conversations without revealing spoilers, Story Sage enhances the reading experience by enabling users to ask questions and receive detailed answers based on the book's content.

## Features

- **Interactive Q&A:** Engage in conversations about your books without encountering spoilers.
- **Semantic Search:** Utilizes advanced embedding models to understand and retrieve relevant information.
- **Customizable Filters:** Filter responses based on book, chapter, or specific entities like characters and places.
- **Persistent Storage:** Stores and retrieves embeddings efficiently using ChromaDB.
- **Extensible Architecture:** Easily extendable components for additional functionalities.

## Architecture

Story Sage employs a modular architecture encompassing Retrieval-Augmented Generation (RAG) and chain-of-thought logic to deliver accurate and context-aware responses.

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

**Entity Filtering:** 

I need to make sure all the entities only exist in one category. 
Currenly something like "Trolloc" is getting picked up as both an animal and a group.
I need to decide an order of precedence for the entity types.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with clear messages.
4. Open a pull request detailing your changes.

For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the [MIT License](LICENSE).
