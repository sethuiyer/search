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
from src.search import SemanticSearch
semantic_search = SemanticSearch()
semantic_search.load_index('index.tar.gz')
```
or train the index on your custom data by using the create\_and\_save\_embeddings. Pass the data as list of strings in the first argument then the index.tar.gz as second.

```python
semantic_search.create_and_save_embeddings(dataset as list of segmented sentences, 'index.tar.gz')
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


## Example

Let's see the performance of this library on a custom dataset

```bash
python test.py 
Embeddings loaded in 4.84 seconds ⚡️
🔍 Query: kshipta avastha
Search completed in 1.69 seconds ⚡️
["When your mind doesn't want to budge, that is called the dull and listless state, when your mind does not want to budge It is called the Muda Avastha, which means The mind has not even come to the level of conscious thought It is just listless It is also called the donkey mind This is called the Muda Avastha The next state is Which also you go into, it is the Kshipta Avastha is kshitva avastha where you are very restless, very agitated, thinking many thoughts and unable to even listen or sit quiet mind.", 'I think this should be closed and after focused state there is a deepened ekagraha avastha which is called nirudha avastha which is possible only through yoga.', 'Agar successful hona ho to donkey monkey butterfly ko chodke ekagra avastha is normal to a healthy mind because it is not being carried away by some emotion, by silly thought process by some fear some anger no no this is not true.']
```

Extras:

## `llm_router.py`
This script uses txtai to determine the query type and the appropriate tools required for processing.

```python
result = classifier.classify_instructions(["Draft a poem which also proves that sqrt of 2 is irrational"])
print(result)
```

