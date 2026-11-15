from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
from src.config import RANDOM_SEED, N_FOLDS
from src.metrics import f1_positive

def get_catboost_model(**kwargs):
    """Возвращает настроенный CatBoostClassifier."""
    default_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'auto_class_weights': 'Balanced',
        'eval_metric': 'F1',
        'random_seed': RANDOM_SEED,
        'verbose': 0
    }
    default_params.update(kwargs)
    return CatBoostClassifier(**default_params)

def evaluate_cv(X, y, model, n_folds=N_FOLDS, random_state=RANDOM_SEED):
    """Кросс-валидация с расчетом F1."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, verbose=0)
        y_pred = model.predict(X_val)
        cv_scores.append(f1_positive(y_val, y_pred))
        
    return np.mean(cv_scores), np.std(cv_scores)