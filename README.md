## Educational Purpose
This project focuses on building a high-quality search engine on custom data using [txtai](https://neuml.github.io/txtai/).
txtai is an all-in-one embeddings database for semantic search, LLM orchestration and language model workflows.

## Overview
The project includes preparing a text corpus, indexing it using txtai, and then performing advanced semantic searches. It leverages txtai's Textractor for text extraction and incorporates a custom `SemanticSearch` class for efficient searching.

## Prerequisites
- Python 3.6+
- txtai library
- Requests library for Python

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
from semantic_search import SemanticSearch

model_path = "sentence-transformers/all-mpnet-base-v2"
semantic_search = SemanticSearch(model_path=model_path)
```

### Step 2: Download and Load the Index
Download the index file and load it into the `SemanticSearch` instance.

```python
import requests

# URL of the index file
url = "https://huggingface.co/<user>/<repo>/resolve/main/index.tar.gz"
index_path = "index.tar.gz"

# Downloading the index file
response = requests.get(url)
with open(index_path, "wb") as file:
    file.write(response.content)

# Load the index file
semantic_search.load_index(index_path)
```
or train the index on your custom data by using the create\_and\_save\_embeddings.

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

