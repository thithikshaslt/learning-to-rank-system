import lightgbm as lgb
import numpy as np

def train_lambdarank(X_train, y_train, group_train):
    """
    Train a LightGBM LambdaRank Learn-to-Rank model.
    Tuned hyperparameters to handle small/medium datasets and avoid 'no further splits' warnings.
    """
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[10],
        learning_rate=0.05,
        n_estimators=200,
        num_leaves=31,
        min_child_samples=5,
        min_child_weight=1e-3,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1
    )
    
    model.fit(
        X_train,
        y_train,
        group=group_train
    )
    
    return model
