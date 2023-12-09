from src.search import SemanticSearch
semantic_search = SemanticSearch()
semantic_search.load_index('index.tar.gz')
search_query = open('query.txt').read()
print(semantic_search.search(search_query))
