from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import lightgbm as lgb
import pickle
import os
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import List, Optional

# Local imports
from bm25 import BM25Retriever
from feature_extractor import FeatureExtractor
from graph_utils import CitationGraph

app = FastAPI(title="LTR and Citation Graph Search API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 15
    use_graph: bool = True

@app.on_event("startup")
def load_models():
    global ltr_model, bm25_retriever, extractor, graph_service, idx_to_paper
    try:
        # Load LTR model
        ltr_model = lgb.Booster(model_file='ltr_model.txt')
        
        # Load BM25 and Corpus
        with open('bm25_corpus.pkl', 'rb') as f:
            bm25_data = pickle.load(f)
            # data structure: {'corpus': [...], 'idx_to_paper': {...}}
            bm25_corpus = bm25_data['corpus']
            idx_to_paper = bm25_data['idx_to_paper']
            bm25_retriever = BM25Retriever(bm25_corpus)
            
        # Initialize Feature Extractor (load TF-IDF from disk)
        # Assuming we saved 'tfidf_vectorizer.pkl' during training
        extractor = FeatureExtractor(vectorizer_path='tfidf_vectorizer.pkl')
        
        # Initialize Graph Service
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        graph_service = CitationGraph(api_key=api_key)
        
        print("Successfully loaded all models and services!")
    except Exception as e:
        print(f"Error loading models: {e}")
        # For demo purposes, if loading fails (e.g. file missing), we'll handle gracefully in the endpoint

@app.post("/search")
def search(request: QueryRequest):
    try:
        query = request.query
        
        # Step 1: BM25 retrieves top 50
        print(f"Step 1: BM25 retrieving top 50 for '{query}'...")
        top_50_indices = bm25_retriever.get_top_k(query, k=50)
        
        b_q, b_d, b_t, b_a, b_s = [], [], [], [], []
        p_ids = []
        for idx, score in top_50_indices:
            paper = idx_to_paper[idx]
            b_q.append(query)
            b_d.append(paper['document'])
            b_t.append(paper['title'])
            b_a.append(paper['abstract'])
            b_s.append(score)
            p_ids.append(paper['paper_id'])
            
        # Step 2: Feature Extraction
        print("Step 2: Extracting LTR features...")
        X_sample = extractor.extract_batch(b_q, b_d, b_t, b_a, b_s)
        
        # Step 3: LTR Reranking
        print("Step 3: Reranking with LTR model...")
        ltr_scores = ltr_model.predict(X_sample)
        
        # Merge scores back to paper list
        papers = []
        for i, idx in enumerate(top_50_indices):
            p_idx = idx[0]
            paper = idx_to_paper[p_idx].copy()
            paper['ltr_score'] = float(ltr_scores[i])
            paper['bm25_score'] = float(b_s[i])
            paper['rank_init'] = i + 1
            papers.append(paper)
            
        # Sort by LTR score
        papers.sort(key=lambda x: x['ltr_score'], reverse=True)
        
        # Step 4: Take top 10-15
        top_k = min(request.top_k, 50)
        top_papers = papers[:top_k]
        top_pids = [p['paper_id'] for p in top_papers]
        
        # Step 5-7: Graph Processing
        graph_data = {"nodes": [], "links": []}
        final_results = top_papers
        
        if request.use_graph:
            print("Step 5-7: Processing citation graph...")
            metrics, graph_data = graph_service.get_pipeline_data(top_pids)
            
            # Step 8: Combine scores
            # Final Score = normalized(LTR) + w1 * normalized(PageRank) + w2 * normalized(CitationCount)
            # Simple version for now:
            for p in top_papers:
                pid = p['paper_id']
                m = metrics.get(pid, {"pagerank": 0, "centrality": 0})
                p['pagerank'] = m['pagerank']
                p['centrality'] = m['centrality']
                
                # Combine (Weights: LTR 60%, PageRank 40%)
                # Normalizing pagerank (pagerank ranges from 0-1, but often very small)
                # We'll use a simple log-scale or max-norm for display
                p['final_score'] = p['ltr_score'] + (m['pagerank'] * 100) # Simple relative boost
            
            # Step 9: Re-sort by final score if graph metrics applied
            final_results.sort(key=lambda x: x.get('final_score', x['ltr_score']), reverse=True)
            for i, p in enumerate(final_results):
                p['final_rank'] = i + 1

        return {
            "query": query,
            "results": final_results,
            "graph": graph_data
        }
        
    except Exception as e:
        print(f"Search API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
