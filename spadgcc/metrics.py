import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
METRICS = ('ACC', 'ARI', 'NMI', 'AMI')

def clustering_metrics(truth, prediction):
    (truth, prediction) = (np.asarray(truth), np.asarray(prediction))
    if truth.ndim != 1 or truth.shape != prediction.shape:
        raise ValueError('Truth and predictions must be aligned one-dimensional vectors.')
    valid = ~pd.isna(truth)
    if not valid.any() or pd.isna(prediction[valid]).any():
        raise ValueError('Evaluation requires labeled spots and nonmissing predictions.')
    truth = truth[valid].astype(str)
    prediction = prediction[valid].astype(str)
    (_, a) = np.unique(truth, return_inverse=True)
    (_, b) = np.unique(prediction, return_inverse=True)
    counts = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(counts, (a, b), 1)
    (rows, cols) = linear_sum_assignment(-counts)
    return {'ACC': float(counts[rows, cols].sum() / len(a)), 'ARI': float(adjusted_rand_score(a, b)), 'NMI': float(normalized_mutual_info_score(a, b, average_method='arithmetic')), 'AMI': float(adjusted_mutual_info_score(a, b, average_method='arithmetic'))}
