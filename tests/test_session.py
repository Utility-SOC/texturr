import numpy as np
import pytest

import session as sess
from session import Session, SessionError, SessionStore

DIRS = {'slow': (10, 0), 'crash': (9, 2), 'design': (0, 10), 'support': (-8, -8)}


def make(texts_by_theme=None, clusters='auto'):
    rng = np.random.default_rng(0)
    texts, vecs = [], {}
    for theme, n in (texts_by_theme or {'slow': 6, 'crash': 6, 'design': 6, 'support': 6}).items():
        for i in range(n):
            t = f'{theme} comment {i}'
            texts.append(t)
            vecs[t] = np.array(DIRS[theme]) + rng.normal(0, 0.3, 2)
    texts.insert(3, 'N/A')
    vecs['N/A'] = np.array([1.0, 1.0])
    return Session.from_texts(texts, lambda ts: np.array([vecs[t] for t in ts]), clusters), texts


def test_from_texts_sets_aside_nonanswers():
    s, texts = make(clusters=4)
    o = s.overview()
    assert o['groups'] == 4 and o['non_answers'] == 1 and o['responses'] == len(texts)
    assert s.assign[3] == -1


def test_overview_has_hints_and_examples_and_is_json_ready():
    import json
    s, _ = make(clusters=4)
    o = s.overview()
    json.dumps(o)
    assert all(len(c['examples']) == 3 and 0 < c['cohesion'] <= 1 for c in o['clusters'])
    top = o['hints']['most_similar_pairs'][0]
    near = {frozenset(p['clusters']) for p in o['hints']['most_similar_pairs']}
    assert top['similarity'] > 0.9                       # 'slow' and 'crash' point the same way
    assert sorted(o['hints']['unlabeled']) == s.cluster_ids()


def test_merge_split_label_flow_and_history():
    s, _ = make(clusters=4)
    ids = s.cluster_ids()
    slow_crash = [c for c in ids if 'slow' in s.texts[int(np.where(s.assign == c)[0][0])] or 'crash' in s.texts[int(np.where(s.assign == c)[0][0])]]
    r = s.merge(slow_crash)
    assert r['merged_into'] == min(slow_crash) and r['size'] == 12 and r['needs_label']
    assert len(s.cluster_ids()) == 3
    sp = s.split(r['merged_into'], 2)
    assert len(sp['new_groups']) == 1 and sum(sp['sizes'].values()) == 12
    for c in s.cluster_ids():
        s.set_label(c, f'Theme {c}', 'sum', 'act')
    assert [h['op'] for h in s.history] == ['created', 'merge', 'split'] + ['label'] * 4


def test_labels_are_sanitized_and_bounded():
    s, _ = make(clusters=4)
    c = s.cluster_ids()[0]
    out = s.set_label(c, '  Bad\x00 \n label ' + 'x' * 500, 'a\tb\x1b[31m', 'z' * 900)
    assert '\x00' not in out['label'] and '\n' not in out['label'] and len(out['label']) <= 80
    assert '\x1b' not in out['summary'] and len(out['action']) <= 300
    with pytest.raises(SessionError, match='empty'):
        s.set_label(c, '   \x00 ')


def test_bad_requests_give_helpful_errors():
    s, _ = make(clusters=4)
    with pytest.raises(SessionError, match='at least two'):
        s.merge([s.cluster_ids()[0]])
    with pytest.raises(SessionError, match='No group 99'):
        s.merge([99, s.cluster_ids()[0]])
    with pytest.raises(SessionError, match='non-answers'):
        s.set_label(-1, 'x')
    with pytest.raises(SessionError, match='between 2 and 5'):
        s.split(s.cluster_ids()[0], 9)
    with pytest.raises(SessionError, match='order'):
        s.examples(s.cluster_ids()[0], order='random')


def test_split_refuses_tiny_groups():
    s, _ = make({'slow': 6, 'crash': 6, 'design': 6, 'support': 3}, clusters=4)
    tiny = [c for c in s.cluster_ids() if (s.assign == c).sum() == 3][0]
    with pytest.raises(SessionError, match='too few'):
        s.split(tiny, 2)


def test_finish_requires_labels_then_locks_edits():
    s, _ = make(clusters=4)
    with pytest.raises(SessionError, match='still need labels'):
        s.finish()
    for c in s.cluster_ids():
        s.set_label(c, f'L{c}')
    assert s.finish()['finished']
    for call in (lambda: s.merge(s.cluster_ids()[:2]), lambda: s.set_label(s.cluster_ids()[0], 'again'),
                 lambda: s.split(s.cluster_ids()[0])):
        with pytest.raises(SessionError, match='finished'):
            call()


def test_examples_paging_and_order():
    s, _ = make(clusters=4)
    c = s.cluster_ids()[0]
    first = s.examples(c, n=2)['examples']
    nxt = s.examples(c, n=2, offset=2)['examples']
    assert not {e['number'] for e in first} & {e['number'] for e in nxt}
    assert s.examples(c, n=999)['examples'].__len__() <= 25
    typical = s.examples(c, n=1)['examples'][0]['number']
    edge = s.examples(c, n=1, order='edge')['examples'][0]['number']
    assert typical != edge


def test_rows_match_cli_report_shape_and_number_responses_by_original_row():
    s, texts = make(clusters=4)
    for c in s.cluster_ids():
        s.set_label(c, f'L{c}')
    rows = s.rows()
    assert {'Cluster', 'Size', 'Label', 'Summary', 'Suggested Action', 'Keyphrases', 'Representative Responses', 'Responses'} <= set(rows[0])
    na = [r for r in rows if r['Cluster'] == -1][0]
    assert na['Responses'] == '4'                         # 'N/A' was inserted as the 4th row
    assert sum(r['Size'] for r in rows) == len(texts)


def test_store_expiry_eviction_and_delete():
    t = [0.0]
    store = SessionStore(max_sessions=2, ttl_seconds=100, clock=lambda: t[0])
    mk = lambda: Session(['a'], np.ones((1, 2)), [0], clock=lambda: t[0])
    a, b = store.add(mk()), store.add(mk())
    t[0] = 10; store.get(b.id)                            # b is fresher
    c = store.add(mk())                                   # evicts the stalest (a)
    with pytest.raises(SessionError, match='Unknown'):
        store.get(a.id)
    assert store.get(b.id) and store.get(c.id)
    t[0] = 500
    with pytest.raises(SessionError, match='expired'):
        store.get(b.id)
    assert store.delete(c.id) is False                    # already expired


def test_move_responses_between_groups():
    s, texts = make(clusters=4)
    ids = s.cluster_ids()
    src, dst = ids[0], ids[1]
    n = int(np.where(s.assign == src)[0][0]) + 1
    s.set_label(src, 'A'); s.set_label(dst, 'B')
    out = s.move([n], dst)
    assert out['moved'] == 1 and s.assign[n - 1] == dst and out['from_groups'] == [src]
    assert s.meta[src]['label'] == 'A' and s.meta[dst]['label'] == 'B'      # labels of other groups are kept
    assert s.history[-1]['op'] == 'move'


def test_move_drops_emptied_group_and_validates():
    s, texts = make({'slow': 6, 'crash': 6, 'design': 6, 'support': 1}, clusters=4)
    lone = [c for c in s.cluster_ids() if (s.assign == c).sum() == 1][0]
    other = [c for c in s.cluster_ids() if c != lone][0]
    s.set_label(lone, 'Lone')
    n = int(np.where(s.assign == lone)[0][0]) + 1
    s.move([n], other)
    assert lone not in s.cluster_ids() and lone not in s.meta                # emptied group disappears
    with pytest.raises(SessionError, match='No group'):
        s.move([1], lone)
    with pytest.raises(SessionError, match='Unknown response'):
        s.move([9999], other)
    with pytest.raises(SessionError, match='between 1 and 200'):
        s.move([], other)
    with pytest.raises(SessionError, match='Non-answers'):
        s.move([4], other)                                                    # row 4 is the 'N/A' inserted by make()
