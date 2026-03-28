from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import pandas as pd
import numpy as np

class FeatureExtractor:
    def __init__(self, df):
        """
        Initialize feature extractors (TF-IDF, BERT bi-encoder, Cross-Encoder).
        """
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        
        all_text = df['title'].tolist() + df['abstract'].tolist() + df['query'].tolist()
        self.tfidf_vectorizer.fit(all_text)
        
        # Bi-encoder for fast similarity
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Cross-encoder for deep semantic scoring
        print("  Loading Cross-Encoder (ms-marco-MiniLM-L-6-v2)...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    def get_feature_names(self):
        return [
            'bm25_score', 'tfidf_sim', 'q_title_sim', 'q_abstract_sim',
            'keyword_overlap', 'bert_sim', 'cross_encoder_score'
        ]

    def extract_batch(self, queries, docs, titles, abstracts, bm25_scores):
        """
        Batch extraction with cross-encoder reranking score as a feature.
        """
        clean_docs = [f"{t} {a}" for t, a in zip(titles, abstracts)]
        
        # 1. BERT Bi-Encoder Embeddings
        q_embs = self.bert_model.encode(queries, convert_to_numpy=True, show_progress_bar=False)
        d_embs = self.bert_model.encode(clean_docs, convert_to_numpy=True, show_progress_bar=False)
        
        # 2. TF-IDF Vectors
        q_tfidf = self.tfidf_vectorizer.transform(queries)
        d_tfidf = self.tfidf_vectorizer.transform(docs)
        t_tfidf = self.tfidf_vectorizer.transform(titles)
        a_tfidf = self.tfidf_vectorizer.transform(abstracts)
        
        # 3. Cross-Encoder scores (deep semantic relevance)
        ce_pairs = list(zip(queries, clean_docs))
        ce_scores = self.cross_encoder.predict(ce_pairs, show_progress_bar=False)
        
        X = []
        for i in range(len(queries)):
            query = queries[i].lower()
            title = titles[i].lower()
            abstract = abstracts[i].lower()
            
            q_tokens = set(query.split())
            t_tokens = set(title.split())
            a_tokens = set(abstract.split())
            d_tokens = t_tokens.union(a_tokens)
            
            # Feature 1: BM25 score
            bm25 = bm25_scores[i]
            
            # Feature 2: TF-IDF Similarity (Query vs Full Text)
            tfidf_sim = float(np.dot(q_tfidf[i].toarray(), d_tfidf[i].toarray().T)[0][0])
            
            # Feature 3: Query-Title TF-IDF Similarity
            q_title_sim = float(np.dot(q_tfidf[i].toarray(), t_tfidf[i].toarray().T)[0][0])
            
            # Feature 4: Query-Abstract TF-IDF Similarity
            q_abstract_sim = float(np.dot(q_tfidf[i].toarray(), a_tfidf[i].toarray().T)[0][0])
            
            # Feature 5: Keyword overlap count
            overlap = len(q_tokens.intersection(d_tokens))
            
            # Feature 6: BERT Bi-Encoder Cosine Sim
            bert_sim = float(np.dot(q_embs[i], d_embs[i]) / (np.linalg.norm(q_embs[i]) * np.linalg.norm(d_embs[i]) + 1e-9))
            
            # Feature 7: Cross-Encoder score (deep semantic)
            ce_score = float(ce_scores[i])
            
            X.append([bm25, tfidf_sim, q_title_sim, q_abstract_sim, overlap, bert_sim, ce_score])
            
        return pd.DataFrame(X, columns=self.get_feature_names())
