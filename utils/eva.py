

import numpy as np
from sklearn.metrics import roc_auc_score
import math

def rHR(sorted_s, k, T2H_id, item_num):

    T2H_bias_ratio = len(T2H_id)/item_num
    num_T2H_bias_in_k = 0.
    for itemid in sorted_s[:k]:
        if itemid in T2H_id:
            num_T2H_bias_in_k += 1
    T2H_bias_ratio_in_k = num_T2H_bias_in_k/k
    return abs(T2H_bias_ratio_in_k-T2H_bias_ratio)


def rND(sorted_s, k, T2H_id, item_num):
    T2H_bias_ratio = len(T2H_id) / item_num
    num_T2H_bias_in_k = 0.
    for itemid in sorted_s[:k]:
        if itemid in T2H_id:
            num_T2H_bias_in_k += 1/(math.log2(k)+1e-6)
    T2H_bias_ratio_in_k = num_T2H_bias_in_k / k
    return abs(T2H_bias_ratio_in_k - T2H_bias_ratio)

def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, ground_truth, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
        Low but correct defination
    """
    GT = set(ground_truth)
    if len(GT) > k :
        sent_list = [1.0] * k
    else:
        sent_list = [1.0]*len(GT) + [0.0]*(k-len(GT))
    dcg_max = dcg_at_k(sent_list, k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num

def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

