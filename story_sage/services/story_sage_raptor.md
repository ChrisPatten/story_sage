# StorySage Raptor

## Introduction
The RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) system 
implements a hierarchical text analysis pipeline that processes books at multiple 
levels using clustering and summarization. It's designed to support both efficient 
similarity search and controlled information access based on reading progress.

## Core Components

### Chunk Management
- **ChunkMetadata**: Tracks book number, chapter number, hierarchy level, and chunk index
- **Chunk**: Container for text with metadata, embeddings, and hierarchical relationships
- **ChunkHierarchy**: Utility for managing the tree structure during processing

### Processing Pipeline
1. **Text Chunking**: 
   - Uses StorySageChunker's sentence_splitter
   - Configurable chunk size and overlap
   - Preserves book and chapter structure
   - Assigns unique chunk identifiers

2. **Two-Phase Clustering**:
   - Global Phase:
     - UMAP dimensionality reduction with larger neighborhood
     - Initial broad grouping using Gaussian Mixture Models
   - Local Phase:
     - Secondary clustering within global clusters
     - Tighter UMAP parameters for local relationships
     - Supports soft clustering (chunks can belong to multiple clusters)

3. **Hierarchical Summarization**:
   - Generates summaries for each cluster using GPT models
   - Builds parent-child relationships between levels
   - Creates progressively higher-level abstractions

## Data Structures

### Chunk Key Format
Unique identifiers follow the pattern:
```
book_{n}|chapter_{m}|level_{l}|chunk_{i}
```

### Hierarchy Tree
Results are organized as:
```python
{
    'book_filepath.txt': {
        'chapter_1': {
            'level_1': [Chunk, ...],  # Original chunks
            'level_2': [Chunk, ...],  # First-level summaries
            'level_3': [Chunk, ...]   # Second-level summaries
        },
        'chapter_2': {
            # Same structure as chapter_1
        }
    }
}
```

### Chunk Object
```python
{
    'text': str,              # Content
    'metadata': {             # ChunkMetadata
        'book_number': int,
        'chapter_number': int,
        'level': int,
        'chunk_index': int
    },
    'chunk_key': str,         # Unique identifier
    'parents': List[str],     # Parent chunk keys
    'children': List[str],    # Child chunk keys
    'embedding': np.ndarray,  # Vector representation
    'is_summary': bool        # True for summary chunks
}
```

## Configuration

### Required Settings
- OpenAI API key (via StorySageConfig)
- Model name (e.g., 'gpt-4o-mini')
- Chunk size and overlap
- Clustering parameters:
  - Threshold (default: 0.5)
  - Target dimensions (default: 10)
  - Maximum clusters (default: 50)
  - UMAP metric (default: 'cosine')

### Optional Settings
- Random seed (default: 8675309)
- Summary parameters:
  - Max tokens
  - Temperature
  - Stop sequence
- GMM initialization attempts

## Usage Example
```python
processor = RaptorProcessor(
    config_path="config.yaml",
    model_name="gpt-4o-mini",
    chunk_size=1000,
    chunk_overlap=50,
    clustering_threshold=0.5,
    metric='cosine'
)

results = processor.process_texts(
    "path/to/books/*.txt",
    number_of_levels=3
)

# Access hierarchical results
for book_path, book_data in results.items():
    for chapter_key, levels in book_data.items():
        chunks = levels['level_1']      # Original text chunks
        summaries = levels['level_2']   # First-level summaries
        
        for summary in summaries:
            print(f"Summary: {summary.text[:100]}...")
            print(f"Summarizes chunks: {summary.children}")
```

## Credits

Credit for the RAPTOR algorithm goes to Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning in their original paper https://arxiv.org/html/2401.18059v1. Their original implementation is available here: https://github.com/parthsarthi03/raptor.

The StorySage Raptor system is a simplified adaptation for the StorySage platform, focusing on book processing and summarization with ability to constrain context by book and chapter number.

Several core components of the RaptorProcessor class are based on the example implementation
described by Vipul Maheshwari here: https://superlinked.com/vectorhub/articles/improve-rag-with-raptor.
