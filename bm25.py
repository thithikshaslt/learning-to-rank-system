from rank_bm25 import BM25Okapi
import numpy as np

class BM25Retriever:
    def __init__(self, corpus):
        """
        Initialize BM25 with a given corpus of documents.
        """
        self.corpus = corpus
        # Tokenize corpus as a list of words
        tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def get_top_k(self, query, k=50):
        """
        Retrieve top K documents for a given query.
        Returns a list of tuples: (document_index, bm25_score)
        """
        tokenized_query = query.lower().split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get indices of top K documents sorted descending by score
        top_indices = np.argsort(doc_scores)[::-1][:k]
        
        return [(idx, doc_scores[idx]) for idx in top_indices]
