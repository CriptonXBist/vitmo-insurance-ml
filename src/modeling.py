from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from src.metrics import f1_positive
from src.config import N_FOLDS, RANDOM_SEED

def get_catboost_model():
    return CatBoostClassifier(
        iterations=1000, 
        learning_rate=0.05, 
        depth=6,
        auto_class_weights='Balanced', 
        eval_metric='F1',
        random_seed=RANDOM_SEED, 
        verbose=0
    )
