"""Refinable analysis sessions: the state that an agent (or any client) inspects and edits.

A session holds the responses, their embeddings and the current grouping. Clients can look at
groups, read examples, merge groups that mean the same thing, split mixed ones, and write
labels. Every change is recorded in `history`, so the final grouping has an audit trail.
Nothing here touches the network or the disk.
"""

import re
import secrets
import threading
import time

import numpy as np

import clean
import clustering

MAX_TEXT = 2000
LIMITS = {'label': 80, 'summary': 400, 'action': 300}
NONANSWER_ID = -1


class SessionError(Exception):
    """A problem with the request itself. The message is safe to show to an agent."""


def _clean_text(value, limit):
    text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', ' ', str(value)).strip()
    return re.sub(r'\s+', ' ', text)[:limit]


def _normalize(rows):
    rows = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.where(norms == 0, 1, norms)


class Session:
    def __init__(self, texts, embeddings, labels, name='', clock=time.time):
        if not (len(texts) == len(embeddings) == len(labels)):
            raise SessionError("texts, embeddings and labels must have the same length")
        self.id = secrets.token_urlsafe(9)
        self.name = _clean_text(name, 80)
        self.texts = [str(t)[:MAX_TEXT] for t in texts]
        self.emb = _normalize(embeddings) if len(texts) else np.zeros((0, 1), dtype=np.float32)
        self.assign = np.asarray(labels, dtype=int).copy()
        self.meta = {}                      # cluster id -> {'label','summary','action'}
        self.history = []
        self.finished = False
        self.source = None                  # set when created from an uploaded file (see server.App)
        self._clock = clock
        self.created = self.touched = clock()
        self.lock = threading.RLock()

    # -- construction --------------------------------------------------------------------------

    @classmethod
    def from_texts(cls, texts, embed, clusters='auto', keep_nonanswers=False, name=''):
        """Cluster `texts` using `embed(list_of_str) -> array`. Boilerplate answers become group -1."""
        texts = [str(t) for t in texts]
        keep = [i for i, t in enumerate(texts) if keep_nonanswers or not clean.is_nonanswer(t)]
        labels = np.full(len(texts), NONANSWER_ID, dtype=int)
        emb = np.zeros((len(texts), 1), dtype=np.float32)
        if keep:
            vectors = _normalize(embed([texts[i] for i in keep]))
            emb = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
            emb[keep] = vectors
            sub, _, _ = clustering.cluster(vectors, clusters)
            labels[keep] = sub
        s = cls(texts, emb, labels, name)
        s.log('created', responses=len(texts), groups=len(s.cluster_ids()), non_answers=int((labels == NONANSWER_ID).sum()))
        return s

    # -- helpers -------------------------------------------------------------------------------

    def log(self, op, **args):
        self.touched = self._clock()
        self.history.append({'op': op, 'args': args, 'at': round(self.touched - self.created, 1)})

    def cluster_ids(self):
        return sorted(int(c) for c in set(self.assign.tolist()) if c != NONANSWER_ID)

    def _members(self, cid):
        if cid == NONANSWER_ID:
            raise SessionError("Group -1 holds non-answers and cannot be edited.")
        idx = np.where(self.assign == cid)[0]
        if not len(idx):
            raise SessionError(f"No group {cid}. Existing groups: {self.cluster_ids()}")
        return idx

    def _centroid(self, idx):
        c = self.emb[idx].mean(axis=0)
        n = np.linalg.norm(c)
        return c / n if n else c

    def _terms(self, top=5):
        """Distinguishing words per group (TF-IDF over one document per group)."""
        ids = self.cluster_ids()
        if not ids:
            return {}
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            docs = [' '.join(self.texts[i] for i in np.where(self.assign == c)[0]) for c in ids]
            vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
            matrix = vec.fit_transform(docs)
            vocab = np.array(vec.get_feature_names_out())
            return {c: [str(vocab[j]) for j in np.argsort(matrix[r].toarray()[0])[::-1][:top] if matrix[r, j] > 0]
                    for r, c in enumerate(ids)}
        except ValueError:          # empty vocabulary
            return {c: [] for c in ids}

    def _example_rows(self, idx, n, offset=0, order='typical'):
        sims = self.emb[idx] @ self._centroid(idx)
        ranked = idx[np.argsort(-sims if order == 'typical' else sims)]
        return [{'number': int(i) + 1, 'text': self.texts[i][:300]} for i in ranked[offset:offset + n]]

    # -- reading -------------------------------------------------------------------------------

    def overview(self):
        with self.lock:
            ids = self.cluster_ids()
            total = len(self.texts)
            cents = {c: self._centroid(self._members(c)) for c in ids}
            terms = self._terms()
            clusters = []
            for c in ids:
                idx = self._members(c)
                nearest = None
                others = [(float(cents[c] @ cents[o]), o) for o in ids if o != c]
                if others:
                    sim, o = max(others)
                    nearest = {'cluster': o, 'similarity': round(sim, 2)}
                m = self.meta.get(c, {})
                clusters.append({
                    'id': c, 'size': int(len(idx)), 'percent': round(100 * len(idx) / max(1, total), 1),
                    'label': m.get('label', ''), 'summary': m.get('summary', ''), 'suggested_action': m.get('action', ''),
                    'labeled': bool(m.get('label')),
                    'cohesion': round(float((self.emb[idx] @ cents[c]).mean()), 2),
                    'nearest_cluster': nearest, 'top_terms': terms.get(c, []),
                    'examples': self._example_rows(idx, 3),
                })
            clusters.sort(key=lambda r: -r['size'])
            return {'session_id': self.id, 'responses': total, 'groups': len(ids),
                    'non_answers': int((self.assign == NONANSWER_ID).sum()), 'finished': self.finished,
                    'clusters': clusters, 'hints': self._hints(clusters, ids, cents)}

    def _hints(self, clusters, ids, cents):
        pairs = sorted(((float(cents[a] @ cents[b]), a, b) for i, a in enumerate(ids) for b in ids[i + 1:]), reverse=True)
        out = {
            'most_similar_pairs': [{'clusters': [a, b], 'similarity': round(s, 2)} for s, a, b in pairs[:3]],
            'least_cohesive': [{'cluster': c['id'], 'cohesion': c['cohesion']}
                               for c in sorted(clusters, key=lambda c: c['cohesion'])[:2] if c['size'] >= 6],
            'unlabeled': [c['id'] for c in clusters if not c['labeled']],
            'note': 'Hints are statistics, not decisions: read the examples before merging or splitting.',
        }
        return out

    def examples(self, cid, n=8, offset=0, order='typical'):
        with self.lock:
            if order not in ('typical', 'edge'):
                raise SessionError("order must be 'typical' or 'edge'")
            n, offset = max(1, min(int(n), 25)), max(0, int(offset))
            idx = self._members(int(cid))
            return {'cluster': int(cid), 'size': int(len(idx)), 'order': order, 'offset': offset,
                    'examples': self._example_rows(idx, n, offset, order)}

    # -- editing -------------------------------------------------------------------------------

    def merge(self, ids, label=None):
        with self.lock:
            self._guard_open()
            ids = sorted(set(int(i) for i in ids))
            if len(ids) < 2:
                raise SessionError("merge needs at least two different group ids")
            for i in ids:
                self._members(i)
            target = ids[0]
            for i in ids[1:]:
                self.assign[self.assign == i] = target
                self.meta.pop(i, None)
            self.meta.pop(target, None)         # the old label no longer describes the merged group
            if label:
                self.meta[target] = {'label': _clean_text(label, LIMITS['label']), 'summary': '', 'action': ''}
            self.log('merge', groups=ids, into=target)
            return {'merged_into': target, 'size': int((self.assign == target).sum()), 'needs_label': not label}

    def split(self, cid, into=2):
        with self.lock:
            self._guard_open()
            cid, into = int(cid), int(into)
            idx = self._members(cid)
            if not 2 <= into <= 5:
                raise SessionError("into must be between 2 and 5")
            if len(idx) < 2 * into:
                raise SessionError(f"Group {cid} has {len(idx)} responses; too few to split into {into}.")
            parts = clustering.kmeans_labels(self.emb[idx], into)
            next_id = max(self.cluster_ids()) + 1
            created = []
            for p in range(1, into):
                self.assign[idx[parts == p]] = next_id
                created.append(next_id)
                next_id += 1
            self.meta.pop(cid, None)
            self.log('split', group=cid, into=into, new_groups=created)
            return {'kept': cid, 'new_groups': created,
                    'sizes': {g: int((self.assign == g).sum()) for g in [cid] + created}, 'needs_label': True}

    def move(self, numbers, to):
        """Reassign individual responses (by their 1-based row number in the overview) to another group."""
        with self.lock:
            self._guard_open()
            to = int(to)
            self._members(to)
            numbers = sorted({int(n) for n in numbers})
            if not numbers or len(numbers) > 200:
                raise SessionError("Give between 1 and 200 response numbers")
            bad = [n for n in numbers if not 1 <= n <= len(self.texts)]
            if bad:
                raise SessionError(f"Unknown response numbers: {bad[:10]}")
            idx = np.array(numbers) - 1
            if (self.assign[idx] == NONANSWER_ID).any():
                raise SessionError("Non-answers cannot be moved.")
            sources = sorted({int(c) for c in self.assign[idx]} - {to})
            self.assign[idx] = to
            for c in sources:                       # a group that lost every member disappears; others keep their labels
                if not (self.assign == c).any():
                    self.meta.pop(c, None)
            self.log('move', responses=numbers, to=to, from_groups=sources)
            return {'moved': len(numbers), 'to': to, 'from_groups': sources,
                    'sizes': {g: int((self.assign == g).sum()) for g in [to] + sources}}

    def set_label(self, cid, label, summary='', action=''):
        with self.lock:
            self._guard_open()
            self._members(int(cid))
            label = _clean_text(label, LIMITS['label'])
            if not label:
                raise SessionError("label must not be empty")
            self.meta[int(cid)] = {'label': label, 'summary': _clean_text(summary, LIMITS['summary']),
                                   'action': _clean_text(action, LIMITS['action'])}
            self.log('label', group=int(cid), label=label)
            return {'cluster': int(cid), **self.meta[int(cid)]}

    def finish(self):
        with self.lock:
            missing = [c for c in self.cluster_ids() if not self.meta.get(c, {}).get('label')]
            if missing:
                raise SessionError(f"Groups still need labels: {missing}. Use set_cluster_label for each.")
            self.finished = True
            self.log('finish')
            return self.overview()

    def _guard_open(self):
        if self.finished:
            raise SessionError("This session is finished and can no longer be edited.")

    # -- output --------------------------------------------------------------------------------

    def rows(self):
        """Report rows in the same shape the CLI writes (see texturr.build_report)."""
        with self.lock:
            terms = self._terms()
            out = []
            for c in self.cluster_ids():
                idx = self._members(c)
                m = self.meta.get(c, {})
                reps = self._example_rows(idx, 3)
                out.append({'Cluster': c, 'Size': int(len(idx)), 'Label': m.get('label', ''), 'Summary': m.get('summary', ''),
                            'Suggested Action': m.get('action', ''), 'Keyphrases': ', '.join(terms.get(c, [])),
                            'Representative Responses': ' | '.join(r['text'] for r in reps),
                            'Responses': ', '.join(str(int(i) + 1) for i in idx)})
            skipped = np.where(self.assign == NONANSWER_ID)[0]
            if len(skipped):
                out.append({'Cluster': NONANSWER_ID, 'Size': int(len(skipped)), 'Label': 'Non-answers',
                            'Summary': 'Blank or boilerplate responses.', 'Suggested Action': '', 'Keyphrases': '',
                            'Representative Responses': ' | '.join(self.texts[i][:300] for i in skipped[:3]),
                            'Responses': ', '.join(str(int(i) + 1) for i in skipped)})
            return out

    def annotation(self):
        """Cluster and theme per response, for writing back into the uploaded file."""
        with self.lock:
            if not self.source:
                raise SessionError("This session was not created from an uploaded file.")
            terms = self._terms()
            labels = {NONANSWER_ID: 'Non-answer'}
            for c in self.cluster_ids():
                labels[c] = self.meta.get(c, {}).get('label') or ', '.join(terms.get(c, [])[:3])
            return {'header': self.source['header'], 'positions': self.source['positions'],
                    'clusters': self.assignments(), 'labels': labels}

    def assignments(self):
        return [int(c) for c in self.assign]


class SessionStore:
    """In-memory only: nothing is written to disk. Old sessions expire and the oldest are evicted."""

    def __init__(self, max_sessions=20, ttl_seconds=3600, clock=time.time):
        self.max_sessions, self.ttl, self._clock = max_sessions, ttl_seconds, clock
        self._items = {}
        self._lock = threading.Lock()

    def _expire(self):
        now = self._clock()
        for sid in [k for k, s in self._items.items() if now - s.touched > self.ttl]:
            del self._items[sid]

    def add(self, session):
        with self._lock:
            self._expire()
            while len(self._items) >= self.max_sessions:
                del self._items[min(self._items, key=lambda k: self._items[k].touched)]
            self._items[session.id] = session
            return session

    def get(self, sid):
        with self._lock:
            self._expire()
            s = self._items.get(sid)
        if s is None:
            raise SessionError("Unknown or expired session_id.")
        s.touched = self._clock()
        return s

    def delete(self, sid):
        with self._lock:
            return self._items.pop(sid, None) is not None
