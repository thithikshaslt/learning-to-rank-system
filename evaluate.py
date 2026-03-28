import numpy as np
from sklearn.metrics import ndcg_score

def precision_at_k(y_true, y_pred, k=10):
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = np.array(y_true)[order]
    relevant_count = np.sum(y_true_sorted[:k] > 0)
    return relevant_count / k

def recall_at_k(y_true, y_pred, k=10):
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = np.array(y_true)[order]
    total_relevant = np.sum(np.array(y_true) > 0)
    if total_relevant == 0:
        return 0.0
    relevant_count = np.sum(y_true_sorted[:k] > 0)
    return relevant_count / total_relevant

def evaluate_ranking(y_true, y_pred, k=10):
    """
    Evaluates ranking outputs given true labels and predicted scores.
    Returns Precision@K, Recall@K, NDCG@K.
    """
    if len(y_true) < 2:
        return 0.0, 0.0, 0.0
    
    try:
        # ndcg_score expects true and predicted scores in 2D array: (n_samples, n_items)
        # Here n_samples = 1 (we evaluate 1 query at a time)
        ndcg = ndcg_score([y_true], [y_pred], k=k)
    except Exception as e:
        ndcg = 0.0
        
    p_at_k = precision_at_k(y_true, y_pred, k)
    r_at_k = recall_at_k(y_true, y_pred, k)
    
    return p_at_k, r_at_k, ndcg
