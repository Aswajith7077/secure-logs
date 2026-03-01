import numpy as np
from sklearn.metrics import precision_recall_curve

def optimal_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1)
    best_threshold = thresholds[best_idx]
    print("Best threshold:", best_threshold)
    return best_threshold
