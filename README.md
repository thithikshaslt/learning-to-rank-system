# 🔍 ArxivSense: AI-Powered Research Discovery Engine

ArxivSense is an advanced research paper search engine that utilizes a **2-stage retrieval pipeline** and a **citation graph engine** to deliver highly relevant results. By combining traditional keyword matching (BM25) with modern machine learning re-ranking (Learning to Rank) and graph-based metrics, it provides a superior search experience for academic papers.

---

## 🚀 Key Features

- **2-Stage Retrieval Pipeline**: 
    - **Stage 1 (Recall)**: Rapidly retrieves 50-100 candidates using the BM25 Okapi algorithm.
    - **Stage 2 (Precision)**: Re-ranks candidates using a **LightGBM LambdaRank** model trained on semantic and lexical features.
- **Advanced Feature Extraction**: Incorporates BM25 scores, TF-IDF similarities, BERT semantic embeddings, and Cross-Encoder relevance scores.
- **Citation Graph Analysis**: Integrates with the **Semantic Scholar API** to fetch citation data and calculate **PageRank** and **Centrality** for further ranking refinement.
- **Interactive Visualization**: A premium React-based dashboard featuring a **D3-powered 2D Force Graph** to explore paper relationships.
- **Performance Evaluation**: Comprehensive metrics including **NDCG@10**, **Precision@10**, and **Recall@10** to measure ranking quality.

---

## 🏗️ Project Structure

```text
├── backend/ (Root Directory)
│   ├── app.py              # FastAPI Search API (Backend Entry Point)
│   ├── main.py             # CLI Tool for Training & Evaluation
│   ├── bm25.py             # BM25 Retrieval Implementation
│   ├── feature_extractor.py # ML Feature Generation (TF-IDF, BERT, etc)
│   ├── graph_utils.py       # Citation Graph & Metric Logic
│   ├── train_ltr.py        # LightGBM Training Logic
│   ├── evaluate.py         # Ranking Evaluation Metrics
│   └── requirements.txt    # Python Dependencies
├── frontend/               # React (Vite) Application
│   ├── src/                # Frontend Source Code
│   │   ├── App.jsx         # Main Dashboard UI
│   │   └── index.css       # Premium Glassmorphism Styling
│   └── package.json        # Frontend Dependencies
└── ltr_model.txt           # Pre-trained LambdaRank Model
```

---

## 🛠️ Setup Instructions

### 1. Prerequisites
- **Python 3.9+**
- **Node.js 18+**
- **Semantic Scholar API Key** (Optional, but recommended for graph features)

### 2. Backend Setup
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/Scripts/activate # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the root directory and add your API key:
   ```env
   SEMANTIC_SCHOLAR_API_KEY="your_api_key_here"
   ```
3. (Optional) Train the LTR model if not already present:
   ```bash
   python main.py --samples 3000
   ```

### 3. Frontend Setup
1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```

---

##  Running the Application

### Step 1: Start the Backend (FastAPI)
From the root directory:
```bash
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```
The API will be available at `http://localhost:8001`.

### Step 2: Start the Frontend (Vite)
From the `frontend` directory:
```bash
npm run dev
```
The dashboard will be available at `http://localhost:5173`.

---

## 🧬 How It Works (The Pipeline)

1. **User Query**: User enters a search term in the dashboard.
2. **Recall (BM25)**: The backend fetches the top 50 matches from the indexed arXiv corpus.
3. **Reranking (LTR)**: For each candidate, 15+ features are extracted (e.g., Cross-Encoder similarity, BM25 score). The LightGBM model predicts a relevance score for each.
4. **Graph Enrichment**: The system fetches citations for the top LTR results and calculates **PageRank** to boost highly influential papers.
5. **Final Ranking**: Papers are sorted by a weighted combination of LTR and Graph scores.
6. **UI Display**: Results are shown in a modern UI with an interactive citation network graph.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, Scikit-Learn, LightGBM, Rank-BM25, Sentence-Transformers, NetworkX.
- **Frontend**: React 19, Vite, Framer Motion, Lucide React, React-Force-Graph, D3.js.
- **Data Source**: arXiv (via Hugging Face Datasets), Semantic Scholar API.

---

