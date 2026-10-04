"""Choosing the number of groups and assigning responses to them."""

import numpy as np


def kmeans_labels(embeddings, k, seed=42):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(np.asarray(embeddings))


def choose_k(embeddings, k_min=2, k_max=12, seed=42):
    """Pick the cluster count with the best cosine silhouette score.

    Returns (k, scores) where scores maps each tried k to its silhouette. Tiny inputs get k=1
    (silhouette needs at least 2 clusters and more points than clusters).
    """
    from sklearn.metrics import silhouette_score
    X = np.asarray(embeddings)
    n = len(X)
    k_max = min(k_max, n - 1)
    if n < 4 or k_max < k_min:
        return max(1, min(n, k_min - 1)), {}
    scores = {}
    for k in range(k_min, k_max + 1):
        labels = kmeans_labels(X, k, seed)
        if len(set(labels)) < 2:
            continue
        scores[k] = float(silhouette_score(X, labels, metric='cosine'))
    if not scores:
        return 1, {}
    return max(scores, key=scores.get), scores


def parse_clusters(value):
    """CLI value for --clusters: 'auto' or a positive integer."""
    if str(value).lower() == 'auto':
        return 'auto'
    try:
        n = int(value)
    except ValueError:
        raise ValueError("--clusters must be 'auto' or a positive whole number")
    if n < 1:
        raise ValueError("--clusters must be 'auto' or a positive whole number")
    return n


def cluster(embeddings, setting='auto', seed=42):
    """Cluster assignments for `setting` ('auto' or an int). Returns (labels, k, scores)."""
    n = len(embeddings)
    if setting == 'auto':
        k, scores = choose_k(embeddings, seed=seed)
    else:
        k, scores = max(1, min(n, setting)), {}
    if k <= 1:
        return np.zeros(n, dtype=int), 1, scores
    return kmeans_labels(embeddings, k, seed), k, scores
