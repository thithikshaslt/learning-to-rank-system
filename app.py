from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import lightgbm as lgb
import pickle
import os
import numpy as np
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from .env
load_dotenv()

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

class SimilarRequest(BaseModel):
    paper_id: str
    top_k: int = 5

# ─── Global state ────────────────────────────────────────────────────
suggestion_queries: list = []
suggestion_titles: list = []
query_embeddings: np.ndarray = None
paper_embeddings: np.ndarray = None
bert_model: SentenceTransformer = None

@app.on_event("startup")
def load_models():
    global ltr_model, bm25_retriever, extractor, graph_service, idx_to_paper
    global suggestion_queries, suggestion_titles
    global query_embeddings, paper_embeddings, bert_model
    try:
        # Load LTR model
        ltr_model = lgb.Booster(model_file='ltr_model.txt')
        
        # Load BM25 and Corpus
        with open('bm25_corpus.pkl', 'rb') as f:
            bm25_data = pickle.load(f)
            bm25_corpus = bm25_data['corpus']
            idx_to_paper = bm25_data['idx_to_paper']
            bm25_retriever = BM25Retriever(bm25_corpus)
            
        # Initialize Feature Extractor (load TF-IDF from disk)
        extractor = FeatureExtractor(vectorizer_path='tfidf_vectorizer.pkl')
        
        # Initialize Graph Service
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        graph_service = CitationGraph(api_key=api_key)
        
        # ─── Suggestion & similarity setup ────────────────────────
        # Load queries for autocomplete + "Did You Mean?"
        try:
            with open('queries.txt', 'r') as f:
                suggestion_queries = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            suggestion_queries = []
        
        # Collect unique paper titles for autocomplete
        suggestion_titles = list(set(
            p['title'] for p in idx_to_paper.values() if p.get('title')
        ))
        
        # Pre-compute BERT embeddings for queries (for "Did You Mean?")
        bert_model = extractor.bert_model
        if suggestion_queries:
            query_embeddings = bert_model.encode(suggestion_queries, convert_to_numpy=True, show_progress_bar=False)
        else:
            query_embeddings = np.array([])
        
        # Pre-compute BERT embeddings for all papers (for "Similar Papers")
        if os.path.exists('paper_embeddings.pkl'):
            print("Loading pre-computed paper embeddings...")
            with open('paper_embeddings.pkl', 'rb') as f:
                paper_embeddings = pickle.load(f)
        else:
            print("Pre-computing paper embeddings for similarity search (First run only)...")
            paper_texts = [
                f"{p['title']} {p['abstract']}" for p in idx_to_paper.values()
            ]
            paper_embeddings = bert_model.encode(paper_texts, convert_to_numpy=True, show_progress_bar=True, batch_size=128)
            # Cache them for next time
            with open('paper_embeddings.pkl', 'wb') as f:
                pickle.dump(paper_embeddings, f)
        
        print("Successfully loaded all models and services!")
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        

@app.post("/search")
def search(request: QueryRequest):
    try:
        query = request.query
        
        # ─── "Did You Mean?" ──────────────────────────────────────
        suggestion = None
        if query_embeddings is not None and len(query_embeddings) > 0 and bert_model is not None:
            q_emb = bert_model.encode([query], convert_to_numpy=True)
            sims = np.dot(query_embeddings, q_emb.T).flatten()
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            best_query = suggestion_queries[best_idx]
            
            # Only suggest if similar but not identical
            if best_sim > 0.6 and best_query.lower().strip() != query.lower().strip():
                suggestion = best_query
        
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
            # Ensure authors field exists
            paper['authors'] = paper.get('authors', '')
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
            
            for p in top_papers:
                pid = p['paper_id']
                m = metrics.get(pid, {"pagerank": 0, "centrality": 0})
                p['pagerank'] = m['pagerank']
                p['centrality'] = m['centrality']
                
                if m.get('year'):
                    p['year'] = m['year']
                
                p['final_score'] = p['ltr_score'] + (m['pagerank'] * 100)
            
            final_results.sort(key=lambda x: x.get('final_score', x['ltr_score']), reverse=True)
            for i, p in enumerate(final_results):
                p['final_rank'] = i + 1

        return {
            "query": query,
            "results": final_results,
            "graph": graph_data,
            "suggestion": suggestion  # "Did You Mean?" suggestion
        }
        
    except Exception as e:
        print(f"Search API Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/suggest")
def suggest(q: str = ""):
    """Autocomplete suggestions based on stored queries and paper titles."""
    if not q or len(q) < 2:
        return {"suggestions": []}
    
    q_lower = q.lower().strip()
    results = []
    seen = set()
    
    # 1. Match from curated queries first (higher priority)
    for query in suggestion_queries:
        if q_lower in query.lower() and query.lower() not in seen:
            results.append({"text": query, "type": "query"})
            seen.add(query.lower())
            if len(results) >= 4:
                break
    
    # 2. Match from paper titles
    for title in suggestion_titles:
        if len(results) >= 8:
            break
        if q_lower in title.lower() and title.lower() not in seen:
            # Truncate long titles for display
            display = title if len(title) <= 80 else title[:77] + "..."
            results.append({"text": display, "type": "paper"})
            seen.add(title.lower())
    
    return {"suggestions": results}


@app.post("/similar")
def find_similar(request: SimilarRequest):
    """Find papers similar to a given paper_id using BERT cosine similarity."""
    try:
        # Find the paper index
        target_idx = None
        target_paper = None
        for idx, paper in idx_to_paper.items():
            if paper['paper_id'] == request.paper_id:
                target_idx = idx
                target_paper = paper
                break
        
        if target_idx is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        # Get the pre-computed embedding for this paper
        target_emb = paper_embeddings[target_idx]
        
        # Compute cosine similarity against all papers
        sims = np.dot(paper_embeddings, target_emb) / (
            np.linalg.norm(paper_embeddings, axis=1) * np.linalg.norm(target_emb) + 1e-9
        )
        
        # Get top K+1 (since the paper itself will be #1)
        top_indices = np.argsort(sims)[::-1][:request.top_k + 1]
        
        similar_papers = []
        for idx in top_indices:
            idx = int(idx)
            if idx == target_idx:
                continue  # Skip self
            paper = idx_to_paper[idx].copy()
            paper['similarity'] = round(float(sims[idx]), 4)
            paper['authors'] = paper.get('authors', '')
            similar_papers.append(paper)
            if len(similar_papers) >= request.top_k:
                break
        
        return {
            "source": target_paper['title'],
            "similar": similar_papers
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Similar Papers Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
