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
   - Later, we can simply open('database.txt').readlines() to get the dataset as list of segmented sentences.

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
or train the index on your custom data by using the create\_and\_save\_embeddings. 
Pass the data as list of strings in the first argument then the index.tar.gz as second.

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
Embeddings loaded in 5.36 seconds ⚡️
🔍 Query: What is kshipta avashta

Search completed in 3.29 seconds ⚡️
['mind please see this carefully kshipta is the distracted or restless state when you know what you should do but well energy nahi hai somehow you want', 'a Avastha The next state is Which also you go into, it is the Kshipta Avastha is kshitva avastha where you are very restless, very agitated, thinking']
```

Then you can use the output from this to the language models

```python
from txtai.pipeline import LLM

# Create and run LLM pipeline
llm = LLM('google/flan-t5-large')
llm(
  """
  SYSTEM: You are Natasha, a friendly assistant who answers user's queries.

USER: what is kshipta avastha

CONTEXT:
 ['mind please see this carefully kshipta is the distracted or restless state when you know what you should do but well energy nahi hai somehow you want', 
 'a Avastha The next state is Which also you go into, it is the Kshipta Avastha is kshitva avastha where you are very restless, very agitated, thinking']

ASSISTANT:
  """
)
```
```text
Natasha: kshipta avastha is the distracted or restless state when you know what you should do but well energy nahi hai somehow you want
```

Pretty good response if you ask me.

Second example:

```bash
python test.py 
Embeddings loaded in 4.42 seconds ⚡️
🔍 Query: Who is Rene Descartes?

Search completed in 1.78 seconds ⚡️
['Descartes, or Cartesius (his Latinized name), is usually regarded as the founder of modern philosophy,and he was also a brilliant mathematician and a', 'According to Werner Heisenberg ( 1958 , p. 81), who struggled with the problem for many years, “This partition has penetrated deeply into the human mi']
```

and giving it to the LLM

```python
llm(
  """
  SYSTEM: You are Natasha, a friendly assistant who answers user's queries from the given context.

USER: Who is Rene Descartes?

CONTEXT:
['Descartes, or Cartesius (his Latinized name), is usually regarded as the founder of modern philosophy,and he was also a brilliant mathematician and a', 'According to Werner Heisenberg ( 1958 , p. 81), who struggled with the problem for many years, “This partition has penetrated deeply into the human mi']


ASSISTANT:
  """
)
```

```text
Descartes, or Cartesius (his Latinized name) is usually regarded as the founder of modern philosophy
```

Again, pretty good.

Extras:

## `llm_router.py`
This script uses txtai to determine the query type and the appropriate tools required for processing.

```python
result = classifier.classify_instructions(["Draft a poem which also proves that sqrt of 2 is irrational"])
print(result)
```

If you find this code useful in your research, please consider citing it as follows:

Sethurathienam Iyer, "Semantic Search with One-Shot Classification," 2024. Available at: https://github.com/sethuiyer/search

