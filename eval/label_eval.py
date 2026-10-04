"""Score how well generated group labels identify what a group is about (Banking77 intents).

Grouping quality is measured by run_eval.py. This measures the LABELS: for groups whose true
intent is known, is the generated label closest (by embedding) to its own intent among all the
intents in the trial? Using the true groups isolates label quality from clustering quality.

    python eval/label_eval.py --provider none                     # keyphrase baseline, no LLM needed
    python eval/label_eval.py --provider ollama --model llama3.1  # a local model
    python eval/label_eval.py --provider anthropic --allow-remote # a hosted model (sends sampled queries out)

Metrics (higher is better):
  top-1   fraction of groups whose label is nearer to their own intent text than to any other intent's
  sim     mean cosine similarity between the label (+ summary) and the group's own intent text
Reference points: random labels land near 1/intents on top-1; labels copied from the true intent score 1.0.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def humanize(intent):
    return intent.replace('_', ' ').replace('-', ' ')


def score(labels, intents, embed):
    """labels[i] describes the group whose true intent is intents[i]. Returns (top1, mean_sim)."""
    L = np.asarray(embed(labels), dtype=float)
    T = np.asarray(embed([humanize(i) for i in intents]), dtype=float)
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    sims = L @ T.T
    return float((sims.argmax(axis=1) == np.arange(len(intents))).mean()), float(np.diag(sims).mean())


def build_trials(texts, label_ids, names, trials, n_intents, per_intent, sample):
    for t in range(trials):
        rng = np.random.default_rng(t)
        picked = rng.choice(sorted(set(label_ids)), n_intents, replace=False)
        groups = []
        for c in picked:
            idx = np.where(np.asarray(label_ids) == c)[0]
            idx = rng.choice(idx, min(per_intent, len(idx)), replace=False)
            groups.append({'intent': names[c], 'texts': [texts[i] for i in idx[:sample]]})
        yield groups


def keyphrase_labeler(groups):
    """The no-LLM baseline: the top TF-IDF terms of each group, the same signal texturr shows without a model."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [' '.join(g['texts']) for g in groups]
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), sublinear_tf=True)
    m = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())
    return [', '.join(vocab[np.argsort(m[i].toarray()[0])[::-1][:3]]) for i in range(len(docs))]


def llm_labeler(config):
    import llm

    def run(groups):
        out = []
        for g in groups:
            try:
                r = llm.label_cluster(config, g['texts'], [])
            except llm.LLMError as e:
                print(f"  labeling failed: {e}", file=sys.stderr)
                r = None
            out.append(f"{r['label']}. {r['summary']}" if r else '')
        return out
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--provider', default='none', help="'none' for the keyphrase baseline, or any texturr --llm provider")
    ap.add_argument('--model'); ap.add_argument('--base-url'); ap.add_argument('--allow-remote', action='store_true')
    ap.add_argument('--trials', type=int, default=5)
    ap.add_argument('--intents', type=int, default=8)
    ap.add_argument('--per-intent', type=int, default=30)
    ap.add_argument('--sample', type=int, default=8, help='queries shown to the labeler per group')
    ap.add_argument('--embedding-model', default='all-MiniLM-L6-v2')
    args = ap.parse_args()

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    d = load_dataset('mteb/banking77', split='train')
    names = {}
    for lid, name in zip(d['label'], d['label_text']):
        names[lid] = name
    st = SentenceTransformer(args.embedding_model)
    embed = lambda xs: st.encode(list(xs), show_progress_bar=False)

    labelers = {'keyphrases (no LLM)': keyphrase_labeler}
    if args.provider != 'none':
        import llm
        cfg = llm.resolve_config(args.provider, args.model, args.base_url)
        if cfg is None:
            sys.exit("No local LLM server found.")
        llm.check_policy(cfg, args.allow_remote, offline=False)
        labelers = {f'{cfg.preset.name} / {cfg.model}': llm_labeler(cfg), **labelers}
    rows = {k: ([], []) for k in labelers}
    refs = {'random labels (chance)': ([], []), 'true intent text (ceiling)': ([], [])}
    for groups in build_trials(list(d['text']), list(d['label']), names, args.trials, args.intents, args.per_intent, args.sample):
        intents = [g['intent'] for g in groups]
        for name, fn in labelers.items():
            t1, sim = score(fn(groups), intents, embed)
            rows[name][0].append(t1); rows[name][1].append(sim)
        rng = np.random.default_rng(len(refs['random labels (chance)'][0]))
        shuffled = [humanize(i) for i in rng.permutation(intents)]
        t1, sim = score(shuffled, intents, embed)
        refs['random labels (chance)'][0].append(t1); refs['random labels (chance)'][1].append(sim)
        t1, sim = score([humanize(i) for i in intents], intents, embed)
        refs['true intent text (ceiling)'][0].append(t1); refs['true intent text (ceiling)'][1].append(sim)
    print(f"\nBanking77 label identification, {args.trials} trials x {args.intents} intents (true groups, so this isolates label quality).\n")
    print("| Labels | top-1 (mean ± sd) | sim (mean ± sd) |\n|---|---|---|")
    for name, (t1, sim) in {**rows, **refs}.items():
        print(f"| {name} | {np.mean(t1):.2f} ± {np.std(t1):.2f} | {np.mean(sim):.2f} ± {np.std(sim):.2f} |")


if __name__ == '__main__':
    main()
