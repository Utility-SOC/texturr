"""Evaluate texturr's grouping step on a public labeled dataset (Banking77 customer queries).

Each trial samples a few intents, hides the labels, groups the queries, and scores the
grouping against the true intents. This measures grouping quality only; it does not
evaluate the LLM-written labels (that needs a model; see the README).

    python eval/run_eval.py [--trials 5] [--intents 6] [--per-intent 60]
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import clustering  # noqa: E402


def load_banking77():
    from datasets import load_dataset
    d = load_dataset('mteb/banking77', split='train')
    return list(d['text']), np.array(d['label']), list(d['label_text'])


def tfidf_kmeans(texts, k, seed):
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit_transform(texts)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=5)
    ap.add_argument('--intents', type=int, default=6, help='distinct true intents per trial')
    ap.add_argument('--per-intent', type=int, default=30)
    ap.add_argument('--model', default='all-MiniLM-L6-v2')
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
    texts, labels, names = load_banking77()
    model = SentenceTransformer(args.model)
    methods = ['TF-IDF + KMeans (true k)', 'texturr, fixed k=5 (old default)', 'texturr, auto k', 'texturr, true k (upper bound)']
    scores = {m: {'ARI': [], 'NMI': []} for m in methods}
    chosen = []
    for trial in range(args.trials):
        rng = np.random.default_rng(trial)
        picked = rng.choice(sorted(set(labels)), args.intents, replace=False)
        idx = np.concatenate([rng.choice(np.where(labels == c)[0], min(args.per_intent, int((labels == c).sum())), replace=False) for c in picked])
        sub, truth = [texts[i] for i in idx], labels[idx]
        emb = model.encode(sub, show_progress_bar=False)
        k_true = args.intents
        runs = {
            methods[0]: tfidf_kmeans(sub, k_true, trial),
            methods[1]: clustering.cluster(emb, 5, trial)[0],
            methods[2]: None,
            methods[3]: clustering.cluster(emb, k_true, trial)[0],
        }
        auto, k, _ = clustering.cluster(emb, 'auto', trial)
        runs[methods[2]] = auto
        chosen.append(k)
        for m, pred in runs.items():
            scores[m]['ARI'].append(ari(truth, pred))
            scores[m]['NMI'].append(nmi(truth, pred))
    print(f"\nBanking77 train split (labels used only for scoring), {args.trials} trials x {args.intents} intents x {args.per_intent} queries "
          f"(embedding model {args.model}). True number of groups = {args.intents}.\n")
    print("| Method | ARI (mean ± sd) | NMI (mean ± sd) |\n|---|---|---|")
    for m in methods:
        a, n = np.array(scores[m]['ARI']), np.array(scores[m]['NMI'])
        print(f"| {m} | {a.mean():.2f} ± {a.std():.2f} | {n.mean():.2f} ± {n.std():.2f} |")
    print(f"\nAuto k chose: {chosen} (true {args.intents}).")


if __name__ == '__main__':
    t = time.time()
    main()
    print(f"({time.time() - t:.0f}s)")
