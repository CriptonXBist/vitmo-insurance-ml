"""
Расчёт целевых метрик и оптимизация порога классификации.
"""
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
from src.config import POSITIVE_CLASS

def f1_positive(y_true, y_pred) -> float:
    """Единая функция расчёта главной метрики: F1-score для класса 1."""
    return f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS)

def find_best_threshold(y_true, probs):
    """
    Подбирает порог, максимизирующий F1 класса 1.
    Вызывать ТОЛЬКО на валидационной выборке, никогда на hold-out test.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1[:-1]))
    return float(thresholds[best_idx]), float(f1[best_idx])
