import argparse
import pandas as pd
import numpy as np
import pickle
import os
import random
from data_loader import load_research_papers
from bm25 import BM25Retriever
from feature_extractor import FeatureExtractor
from train_ltr import train_lambdarank
from evaluate import evaluate_ranking

def main():
    parser = argparse.ArgumentParser(description="Learning to Rank Research Paper Search Engine")
    parser.add_argument("--samples", type=int, default=3000, help="Number of papers to load from dataset")
    parser.add_argument("--query", type=str, default="", help="Custom query to test")
    parser.add_argument("--query_file", type=str, default="", help="Path to a file containing custom queries")
    args = parser.parse_args()
    
    # 1. Load Data
    print(f"1. Loading arXiv research papers (limit: {args.samples})...")
    df = load_research_papers(samples=args.samples)
    
    # Preprocess corpus for BM25 and Indexing
    unique_papers_df = df.drop_duplicates(subset=['paper_id'])
    idx_to_paper = {i: {
        'title': row['title'],
        'abstract': row['abstract'],
        'document': row['document'],
        'paper_id': row['paper_id']
    } for i, row in enumerate(unique_papers_df.to_dict('records'))}
    
    corpus = [p['document'] for p in idx_to_paper.values()]
    # Reverse map: paper_id -> corpus index
    paperid_to_idx = {p['paper_id']: i for i, p in idx_to_paper.items()}
    
    # 2. Stage 1 Retrieval (BM25)
    print("\n2. Initializing Stage 1 Retriever (BM25)...")
    bm25_retriever = BM25Retriever(corpus)
    
    # 3. Feature Extraction & Data Preparation
    print("\n3. Initializing Feature Extractor (TF-IDF, BERT & Cross-Encoder)...")
    extractor = FeatureExtractor(df)
    
    # Split queries — use ANY positive label (1 or 2)
    queries_with_pos = df[df['label'] > 0]['query_id'].unique()
    random.seed(42)
    query_list = list(queries_with_pos)
    random.shuffle(query_list)
    
    split_idx = int(0.8 * len(query_list))
    train_qids = set(query_list[:split_idx])
    test_qids = set(query_list[split_idx:])
    
    print(f"Split: {len(train_qids)} train queries, {len(test_qids)} test queries.")

    def prepare_data(qids):
        X_list = []
        y_list = []
        group_list = []
        
        subset_df = df[df['query_id'].isin(qids)]
        grouped = subset_df.groupby('query_id')
        
        processed = 0
        for qid, group in grouped:
            query = group['query'].iloc[0]
            # Stage 1: Retrieve top 50 candidates via BM25
            top_docs = bm25_retriever.get_top_k(query, k=50)
            
            # Build label lookup from the balanced training data
            rel_ids = {row['paper_id']: row['label'] for _, row in group.iterrows()}
            
            # Prepare batch for feature extraction
            b_queries = []
            b_docs = []
            b_titles = []
            b_abstracts = []
            b_bm25_scores = []
            labels = []
            
            for doc_idx, bm25_score in top_docs:
                paper = idx_to_paper[doc_idx]
                labels.append(rel_ids.get(paper['paper_id'], 0))
                b_queries.append(query)
                b_docs.append(paper['document'])
                b_titles.append(paper['title'])
                b_abstracts.append(paper['abstract'])
                b_bm25_scores.append(bm25_score)
            
            # Must have at least one positive AND one negative for meaningful training
            has_pos = any(l > 0 for l in labels)
            has_neg = any(l == 0 for l in labels)
            
            if has_pos and has_neg and len(labels) >= 5:
                group_X = extractor.extract_batch(b_queries, b_docs, b_titles, b_abstracts, b_bm25_scores)
                X_list.append(group_X)
                y_list.extend(labels)
                group_list.append(len(labels))
                processed += 1
            
            if processed % 10 == 0 and processed > 0:
                print(f"  Processed {processed} queries...")
                
        if not X_list:
            return None, None, None
            
        X_all = pd.concat(X_list, ignore_index=True)
        y_all = np.array(y_list)
        
        # Verify feature variance (catch constant features)
        for col in X_all.columns:
            if X_all[col].std() < 1e-10:
                print(f"  WARNING: Feature '{col}' has near-zero variance!")
        
        print(f"  Label distribution in this split: {dict(zip(*np.unique(y_all, return_counts=True)))}")
        return X_all, y_all, group_list

    print("\nPreparing Training Data...")
    X_train, y_train, g_train = prepare_data(train_qids)
    
    print("\nPreparing Testing Data...")
    X_test, y_test, g_test = prepare_data(test_qids)
    
    # 4. Train LTR Model
    if X_train is not None and g_train:
        print(f"\n4. Training Stage 2 LTR Model (LightGBM LambdaRank) on {len(g_train)} queries...")
        print(f"   Features: {list(X_train.columns)}")
        model = train_lambdarank(X_train, y_train, g_train)
        model.booster_.save_model('ltr_model.txt')
        
        # --- SAVE PERSISTENCE ARTIFACTS FOR Search API ---
        print("\nSaving Search API persistence artifacts...")
        extractor.save_vectorizer('tfidf_vectorizer.pkl')
        with open('bm25_corpus.pkl', 'wb') as f:
            pickle.dump({
                'corpus': corpus,
                'idx_to_paper': idx_to_paper
            }, f)
        print("Artifacts saved: ltr_model.txt, tfidf_vectorizer.pkl, bm25_corpus.pkl")
        
        # Print feature importances
        importances = model.feature_importances_
        for name, imp in sorted(zip(X_train.columns, importances), key=lambda x: -x[1]):
            print(f"   {name}: {imp}")
    else:
        print("Not enough training data. Exiting.")
        return
        
    # 5. Evaluate
    print("\n5. Evaluating Performance (NDCG@10, P@10, R@10)...")
    bm25_metrics = {"ndcg": [], "p10": [], "r10": []}
    ltr_metrics = {"ndcg": [], "p10": [], "r10": []}
    
    if X_test is not None and g_test:
        curr_idx = 0
        for g_size in g_test:
            y_true = y_test[curr_idx : curr_idx + g_size]
            X_group = X_test.iloc[curr_idx : curr_idx + g_size]
            
            bm25_scores = X_group['bm25_score'].values
            p, r, n = evaluate_ranking(y_true, bm25_scores, k=10)
            bm25_metrics["ndcg"].append(n)
            bm25_metrics["p10"].append(p)
            bm25_metrics["r10"].append(r)
            
            ltr_scores = model.predict(X_group)
            p, r, n = evaluate_ranking(y_true, ltr_scores, k=10)
            ltr_metrics["ndcg"].append(n)
            ltr_metrics["p10"].append(p)
            ltr_metrics["r10"].append(r)
            
            curr_idx += g_size
        
        print("\n" + "="*60)
        print("          RESEARCH PAPER RANKING EVALUATION")
        print("="*60)
        print(f"{'Metric':<10} | {'BM25':<10} | {'LTR (Our Model)':<16} | {'Gain':<10}")
        print("-" * 60)
        for m in ["ndcg", "p10", "r10"]:
            b_val = np.mean(bm25_metrics[m])
            l_val = np.mean(ltr_metrics[m])
            gain = ((l_val - b_val) / (b_val + 1e-9)) * 100
            print(f"{m.upper():<10} | {b_val:<10.4f} | {l_val:<16.4f} | {gain:>+7.2f}%")
        print("="*60)
    else:
        print("  No test data available for evaluation.")
        
    # 6. Interactive Search
    test_queries = []
    if args.query:
        test_queries.append(args.query)
    elif args.query_file and os.path.exists(args.query_file):
        with open(args.query_file, 'r') as f:
            test_queries = [line.strip() for line in f if line.strip()]
    else:
        test_queries = ["large language models", "deep learning", "computer vision"]

    for query_to_test in test_queries:
        print(f"\n{'='*60}")
        print(f"  Search Results for: '{query_to_test}'")
        print(f"{'='*60}")
        
        top_docs_bm25 = bm25_retriever.get_top_k(query_to_test, k=50)
        
        b_q, b_d, b_t, b_a, b_s = [], [], [], [], []
        for doc_idx, score in top_docs_bm25:
            paper = idx_to_paper[doc_idx]
            b_q.append(query_to_test)
            b_d.append(paper['document'])
            b_t.append(paper['title'])
            b_a.append(paper['abstract'])
            b_s.append(score)
            
        X_sample = extractor.extract_batch(b_q, b_d, b_t, b_a, b_s)
        final_scores = model.predict(X_sample)
        
        # Also get cross-encoder scores for display
        ce_scores = X_sample['cross_encoder_score'].values
        
        top_indices = np.argsort(final_scores)[::-1][:10]
        print(f"\n{'Rank':<5} | {'LTR':>8} | {'CE':>8} | {'BM25':>8} | Paper Title")
        print("-" * 80)
        for rank, idx in enumerate(top_indices):
            title = b_t[idx]
            abstract = b_a[idx]
            print(f"{rank+1:<5} | {final_scores[idx]:>8.4f} | {ce_scores[idx]:>8.4f} | {b_s[idx]:>8.2f} | {title}")
            print(f"       Abstract: {abstract[:150]}...\n")

if __name__ == "__main__":
    main()
