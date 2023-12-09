import os
import time
from collections import OrderedDict
from txtai.embeddings import Embeddings
from txtai.pipeline import Segmentation
from txtai.pipeline import Labels
from src.util import process_text, filter_scores, clean_strings


class SemanticSearch:
    def __init__(self, config_dict=None):
        """
        Constructor for the SemanticSearch class.

        Parameters:
        - model_path (str): Path to the pre-trained model for embeddings.
        """
        # Load the embeddings index
        if not config_dict:
            self.config_dict = {"path": "sentence-transformers/all-mpnet-base-v2", "content": "sqlite", "backend": "numpy", "hybrid": True}
        self.embeddings = Embeddings(self.config_dict)
        self.__emb_loaded__ = False

        # Internal variables
        self.labels = Labels('facebook/bart-large-mnli')
        self.tags = ['purely_lexical', 'mostly_lexical', 'balanced', 'mostly_semantic', 'purely_semantic']
        self.weights = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.segmenter = Segmentation(sentences=True, minlength=150)

    def create_and_save_embeddings(self, processed_data, index_path="index.tar.gz"):
        """
        Create and save embeddings for processed data using the txtai library.

        Parameters:
        - processed_data (list): A list of units, each containing combined sentences.
        - model_path (str): Path to the pre-trained model for embeddings.
        - index_path (str): Path to save the index file.

        Returns:
        - None
        """
        if os.path.exists(index_path):
            print(f"Index file {index_path} already exists. Skipping embedding creation.")
            return

        # Index the processed data
        start_time = time.time()
        self.embeddings.index((x, text, None) for x, text in enumerate(processed_data))

        # Save the embeddings index
        self.embeddings.save(index_path)
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Embeddings generated and saved in {time_taken:.2f} seconds ⚡️")
        self.__emb_loaded__ = True

    def load_index(self, index_path):
        """
        Load the embeddings index file.

        Parameters:
        - index_path (str): Path to the embeddings index file.

        Returns:
        - None
        """
        if not os.path.exists(index_path):
            print(f"Index file {index_path} does not exist. Unable to load embeddings.")
            return
        start_time = time.time()
        self.embeddings.load(index_path)
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Embeddings loaded in {time_taken:.2f} seconds ⚡️")
        self.__emb_loaded__ = True

    def calculate_dynamic_threshold(self, query):
        """
        Calculate the dynamic threshold for the given query.

        Parameters:
        - query (str): Query for dynamic threshold calculation.

        Returns:
        - float: Dynamic threshold value.
        """
        # Get the labels and weights
        label_weights = self.labels(query, self.tags)

        # Calculate the dynamic threshold as the weighted sum
        dynamic_threshold = sum(weight * label_weight[1] for label_weight, weight in zip(label_weights, self.weights))

        return dynamic_threshold

    def search(self, query, limit=5):
        """
        Perform semantic search for the given query.

        Parameters:
        - query (str): Query for semantic search.
        - limit (int): Maximum number of results to return.

        Returns:
        - list: List of dictionaries with keys 'id', 'text', and 'score'.
        """
        if not self.__emb_loaded__:
            print("Embeddings are not loaded. Unable to perform search.")
            return None

        start_time = time.time()
        print(f"🔍 Query: {query}")

        # Calculate dynamic threshold
        dynamic_threshold = self.calculate_dynamic_threshold(query)

        # Perform search with dynamic threshold
        results = self.embeddings.search(query, weights=dynamic_threshold, limit=limit)

        # Concatenate and preprocess text
        relevant_results = ". ".join(result['text'] for result in results)
        
        # This is to be removed
        relevant_results = relevant_results.replace('keyflix_', '. keyflix_')

        # Split into sentences
        sentences = process_text(self.segmenter(relevant_results), max_length=150)

        # Index and search using keyword embeddings
        keyword_embedder = Embeddings(keyword=True)
        keyword_embedder.index([(x, sentence, None) for x, sentence in enumerate(sentences)])
        keyword_results = keyword_embedder.search(query, limit=8)
        keyword_results = filter_scores(keyword_results)
        relevant_results = [sentences[x[0]] for x in keyword_results]
        relevant_results = clean_strings(relevant_results)
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Search completed in {time_taken:.2f} seconds ⚡️")
        return relevant_results


