import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
from src.config import POSITIVE_CLASS

def f1_positive(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS)

def find_best_threshold(y_true, probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])
