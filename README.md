# Learning-to-Rank Information Retrieval System

This repository provides a complete Learning-to-Rank (LTR) Information Retrieval system using Python. 
It implements a 2-stage retrieval pipeline:
1. **Stage 1 (Candidate Retrieval):** Uses BM25 (via `rank_bm25`) to fetch the top 50 documents for a given query.
2. **Stage 2 (Re-ranking):** Uses a LightGBM LambdaRank trained model to re-rank the top K candidates using extracted features.

## Architecture & Modules
- `data_loader.py`: Loads the MS MARCO dataset via HuggingFace's `datasets` library.
- `bm25.py`: Wrapper for BM25 Okapi algorithm to retrieve initial candidates.
- `feature_extractor.py`: Generates machine learning features (TF-IDF cosine similarity, BM25 score, Query/Doc lengths, Keyword overlap, BERT Embedding similarity).
- `train_ltr.py`: Trains the LightGBM point-wise/pair-wise tree model.
- `evaluate.py`: Calculates NDCG@10, Precision@K, and Recall@K.
- `main.py`: Full execution pipeline connecting all the above.
- `app.py`: Simple FastAPI stub endpoint (bonus).

## Setup Instructions

1. Install requirements
```bash
pip install -r requirements.txt
```

2. Run the main processing, training, and evaluation pipeline
```bash
python main.py --samples 500
```
This command loads 500 query-document pairs, builds the knowledge corpus, computes features, trains the model, and prints out NDCG@10 evaluations.

3. Run an Interactive Search
```bash
python main.py --query "what is the largest animal on earth?"
```

4. Fast API Endpoint
```bash
uvicorn app:app --reload
```
