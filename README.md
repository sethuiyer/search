## Educational Purpose
This project focuses on building a high-quality search engine on custom data using [txtai](https://neuml.github.io/txtai/).
txtai is an all-in-one embeddings database for semantic search, LLM orchestration and language model workflows.

## Overview
The project includes preparing a text corpus, indexing it using txtai, and then performing advanced semantic searches. It leverages txtai's Textractor for text extraction and incorporates a custom `SemanticSearch` class for efficient searching.

## Prerequisites
- Python 3.6+
- txtai library

## Corpus Preparation
1. **Extract Text Data**:
   - Use [txtai's Textractor](https://neuml.github.io/txtai/pipeline/data/textractor/) to extract text from various materials. Ensure `sentences=True` is set.
   - Store the extracted list of sentences in separate text files for different materials.
   - Merge these files into a single text file named `database.txt`.

## `search.py`
This script uses txtai to process, index, and load the raw data present in `database.txt`. It sets up the infrastructure for the search engine.

## `SemanticSearch` Class Usage

### Step 1: Initialization
Create an instance of the `SemanticSearch` class. Specify the model path for embeddings.

```python
from src.search import SemanticSearch
semantic_search = SemanticSearch()
```

### Step 2: Download and Load the Index
Download the index file and load it into the `SemanticSearch` instance.

```bash
wget https://huggingface.co/<user>/<repo>/resolve/main/index.tar.gz # or any URL where your index lives
```

Then you can simply 
```python
# Load the index file
semantic_search.load_index(index_path)
```

or train the index on your custom data by using the create\_and\_save\_embeddings. Pass the data as list of strings in the first argument then the index.tar.gz as second.

```python
from src.search import SemanticSearch
semantic_search = SemanticSearch()
semantic_search.load_index('index.tar.gz')
print(semantic_search.search('Q4 performance forecast'))
```

### Step 3: Performing a Search
Perform semantic searches using the `search` method.

```python
query = "Your search query"
results = semantic_search.search(query, limit=5)

# Displaying results
for result in results:
    print(result)
```

## `llm_router.py`
This script uses txtai to determine the query type and the appropriate tools required for processing.

```python
result = classifier.classify_instructions(["Draft a poem which also proves that sqrt of 2 is irrational"])
print(result)
```

