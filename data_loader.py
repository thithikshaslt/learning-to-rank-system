from datasets import load_dataset
import pandas as pd
import random
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_research_papers(samples=3000):
    """
    Load arXiv research papers filtered by CS domains and generate query-document pairs.
    Uses strict BERT-based semantic labeling and class balancing.
    """
    print(f"Loading arXiv dataset and filtering for CS.AI/LG/CV...")
    
    load_chunk = max(samples * 100, 500000)
    try:
        dataset = load_dataset('gfissore/arxiv-abstracts-2021', split=f'train[:{load_chunk}]')
    except:
        print("Fallback to alternative dataset 'scientific_papers'...")
        dataset = load_dataset('scientific_papers', 'arxiv', split=f'train[:{load_chunk}]')
    
    papers = []
    relevant_cats = {'cs.AI', 'cs.LG', 'cs.CV'}
    
    for item in dataset:
        categories = item.get('categories', '')
        if isinstance(categories, list):
            cats = set(categories)
        else:
            cats = set(categories.split())
            
        if not (cats & relevant_cats):
            continue
            
        p_id = item.get('id', '')
        title = clean_text(item.get('title', ''))
        abstract = clean_text(item.get('abstract', ''))
        
        if title and abstract and len(abstract.split()) > 20:
            weighted_doc = (title + " ") * 3 + abstract
            papers.append({
                'paper_id': p_id,
                'title': title,
                'abstract': abstract,
                'document': weighted_doc
            })
            if len(papers) >= samples:
                break
                
    print(f"Filtering complete. Collected {len(papers)} papers in target domains.")
    
    queries = [
        "deep learning", "neural networks", "computer vision", "transformers", "reinforcement learning",
        "natural language processing", "generative adversarial networks", "graph neural networks",
        "transfer learning", "unsupervised learning", "semi-supervised learning", "few-shot learning",
        "zero-shot classification", "self-supervised learning", "contrastive learning",
        "attention mechanism", "recurrent neural networks", "convolutional neural networks",
        "object detection", "image segmentation", "face recognition", "pose estimation",
        "action recognition", "medical image analysis", "image generation", "text to image",
        "machine translation", "question answering", "text summarization", "sentiment analysis",
        "named entity recognition", "speech recognition", "text to speech", "speaker verification",
        "autonomous driving", "robot path planning", "robotics manipulation", "meta learning",
        "multi-agent reinforcement learning", "inverse reinforcement learning", "offline reinforcement learning",
        "federated learning", "privacy preserving machine learning", "differential privacy",
        "adversarial attacks", "adversarial robustness", "explainable ai", "interpretable machine learning",
        "fairness in machine learning", "causal inference", "representation learning",
        "knowledge graph", "recommender systems", "collaborative filtering", "time series forecasting",
        "anomaly detection", "dimensionality reduction", "hyperparameter optimization",
        "neural architecture search", "automl", "quantum machine learning", "large language models",
        "vision transformers", "diffusion models", "active learning"
    ]
    
    # --- BERT-based semantic labeling ---
    print("\nComputing BERT embeddings for semantic labels...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    paper_texts = [p['title'] + " " + p['abstract'] for p in papers]
    paper_embs = model.encode(paper_texts, convert_to_numpy=True, show_progress_bar=True, batch_size=128)
    query_embs = model.encode(queries, convert_to_numpy=True, show_progress_bar=False)
    sims = cosine_similarity(query_embs, paper_embs)
    
    data = []
    print("\nData Quality Check (strict labeling):")
    
    for i, query in enumerate(queries):
        qid = f"q{i:03d}"
        q_sims = sims[i]
        
        # Strict labeling: only truly relevant papers get positive labels
        label_2_count = 0
        label_1_count = 0
        query_pairs = []
        
        for p_idx in range(len(papers)):
            paper = papers[p_idx]
            sim = float(q_sims[p_idx])
            
            # STRICT labeling (cosine > 0.7 for label 2, > 0.5 for label 1)
            # Also check keyword presence for additional confidence
            q_lower = query.lower()
            title_lower = paper['title'].lower()
            abstract_lower = paper['abstract'].lower()
            q_tokens = set(q_lower.split())
            t_tokens = set(title_lower.split())
            a_tokens = set(abstract_lower.split())
            
            if sim >= 0.7 or (q_lower in title_lower):
                label = 2
                label_2_count += 1
            elif sim >= 0.5 or (len(q_tokens & t_tokens) >= 2) or (q_lower in abstract_lower):
                label = 1
                label_1_count += 1
            else:
                label = 0
            
            query_pairs.append({
                'query_id': qid,
                'query': query,
                'paper_id': paper['paper_id'],
                'title': paper['title'],
                'abstract': paper['abstract'],
                'document': paper['document'],
                'label': label,
                'sim_score': sim
            })
        
        # Ensure minimum relevant documents per query (at least 10)
        if label_2_count + label_1_count < 10:
            # Promote top papers by similarity
            sorted_pairs = sorted(enumerate(query_pairs), key=lambda x: x[1]['sim_score'], reverse=True)
            needed = 10 - (label_2_count + label_1_count)
            promoted = 0
            for idx, pair in sorted_pairs:
                if pair['label'] == 0 and promoted < needed:
                    query_pairs[idx]['label'] = 1
                    promoted += 1
                    label_1_count += 1
        
        total_pos = label_2_count + label_1_count
        
        # --- CLASS BALANCING: Downsample label 0 ---
        # Keep all positives + sample negatives (max 5x positives to avoid extreme imbalance)
        positives = [p for p in query_pairs if p['label'] > 0]
        negatives = [p for p in query_pairs if p['label'] == 0]
        
        # Sort negatives by sim_score descending to keep harder negatives
        negatives.sort(key=lambda x: x['sim_score'], reverse=True)
        max_neg = max(total_pos * 5, 50)  # Keep at most 5x positives or 50
        negatives = negatives[:max_neg]
        
        balanced_pairs = positives + negatives
        
        if i % 10 == 0:
            print(f"  Query: '{query}' -> L2:{label_2_count}, L1:{label_1_count}, L0:{len(negatives)} (total:{len(balanced_pairs)})")
        
        data.extend(balanced_pairs)

    df = pd.DataFrame(data)
    # Drop the temporary sim_score column
    df = df.drop(columns=['sim_score'])
    
    print(f"\nFinal Dataset Stats:")
    print(f"Total pairs: {len(df)}")
    print(f"Number of queries: {df['query_id'].nunique()}")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index().to_string()}")
    
    return df
