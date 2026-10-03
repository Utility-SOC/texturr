import hashlib
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import models

NOW = datetime(2026, 10, 3, tzinfo=timezone.utc)


def rec(repo, downloads=1000, lic='apache-2.0', total=8e9, updated='2026-09-01T00:00:00.000Z',
        tags=('conversational',), sha='a' * 40, pipeline='text-generation'):
    return {'id': repo, 'author': repo.split('/')[0], 'downloads': downloads, 'likes': 1, 'sha': sha,
            'lastModified': updated, 'pipeline_tag': pipeline, 'gguf': {'total': total} if total else {},
            'tags': ['gguf', f'license:{lic}', *tags]}


def test_vet_rules():
    ok = lambda **k: models.vet(rec('ibm-granite/granite-8b-GGUF', **k), now=NOW)
    assert ok() is not None
    assert ok(lic='gemma') is None                      # non-permissive license
    assert ok(lic='llama3.1') is None
    assert ok(total=70e9) is None                       # over the size budget
    assert ok(pipeline='automatic-speech-recognition') is None   # not a text model
    assert ok(pipeline=None) is not None                # repos with no pipeline tag are fine
    derived = rec('ibm-granite/x-GGUF', tags=('conversational', 'base_model:finetune:Qwen/Qwen3-1.7B'))
    assert models.vet(derived, now=NOW) is None          # derived from a non-allowlisted publisher
    own = rec('ibm-granite/x-GGUF', tags=('conversational', 'base_model:quantized:ibm-granite/x'))
    assert models.vet(own, now=NOW) is not None
    assert ok(total=None) is None                       # unknown size is rejected
    assert ok(updated='2024-01-01T00:00:00.000Z') is None   # stale
    assert models.vet(rec('x/model-base-GGUF'), now=NOW) is None
    assert models.vet(rec('x/model-GGUF', tags=()), now=NOW) is None   # not chat/instruct
    assert models.vet(rec('x/model-instruct-GGUF', tags=()), now=NOW) is not None


def test_recommend_ranks_caps_per_org_and_ignores_unlisted_orgs():
    data = {
        'ibm-granite': [rec('ibm-granite/a', 900), rec('ibm-granite/b', 800), rec('ibm-granite/c', 700)],
        'microsoft': [rec('microsoft/phi', 500)],
        'mistralai': [rec('mistralai/m', 950, lic='mit')],
    }
    get = lambda url: data[url.split('author=')[1].split('&')[0]]
    out = models.recommend(('ibm-granite', 'microsoft', 'mistralai'), per_org=2, top=5, get=get, now=NOW)
    assert [c.repo for c in out] == ['mistralai/m', 'ibm-granite/a', 'ibm-granite/b', 'microsoft/phi']


def test_default_orgs_exclude_qwen_and_deepseek():
    assert not {'Qwen', 'deepseek-ai'} & set(models.DEFAULT_ORGS)


def test_all_orgs_failing_raises():
    def get(url):
        raise models.ModelsError('down')
    with pytest.raises(models.ModelsError, match='down'):
        models.recommend(('a', 'b'), get=get, now=NOW)


def test_pick_file_refuses_shards_and_unverifiable():
    tree = [{'path': 'm-Q4_K_M-00001-of-00002.gguf', 'lfs': {'oid': 'x', 'size': 1}},
            {'path': 'm-Q8_0.gguf', 'lfs': {'oid': 'y', 'size': 2}}]
    with pytest.raises(models.ModelsError, match='Available'):
        models.pick_file('o/r', 's', 'Q4_K_M', get=lambda u: tree)
    assert models.pick_file('o/r', 's', 'Q8_0', get=lambda u: tree)['sha256'] == 'y'
    with pytest.raises(models.ModelsError, match='unverifiable'):
        models.pick_file('o/r', 's', 'Q8_0', get=lambda u: [{'path': 'm-Q8_0.gguf'}])


class Blob(BaseHTTPRequestHandler):
    payload = b'not really a model' * 1000

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(self.payload)


@pytest.fixture
def blob_url():
    srv = HTTPServer(('127.0.0.1', 0), Blob)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{srv.server_port}/f.gguf'
    srv.shutdown()


def test_download_verifies_hash(blob_url, tmp_path):
    good = hashlib.sha256(Blob.payload).hexdigest()
    dest = str(tmp_path / 'd' / 'f.gguf')
    models.download_verified(blob_url, dest, good, report=None)
    assert open(dest, 'rb').read() == Blob.payload


def test_download_rejects_and_deletes_on_mismatch(blob_url, tmp_path):
    dest = str(tmp_path / 'f.gguf')
    with pytest.raises(models.ModelsError, match='mismatch'):
        models.download_verified(blob_url, dest, '0' * 64, report=None)
    assert not list(tmp_path.iterdir())


def test_pull_requires_consent_and_pins_revision(monkeypatch, tmp_path, blob_url):
    monkeypatch.setenv('TEXTURR_HOME', str(tmp_path))
    good = hashlib.sha256(Blob.payload).hexdigest()
    tree = [{'path': 'm-Q4_K_M.gguf', 'lfs': {'oid': good, 'size': len(Blob.payload)}}]
    c = models.Candidate('o/r-GGUF', 'o', 'apache-2.0', 8.0, 1, 1, '2026-09-01', 'f' * 40)
    calls = []
    fetch = lambda url, dest, sha, size: calls.append(url) or models.download_verified(blob_url, dest, sha, size, None)
    with pytest.raises(models.ModelsError, match='cancelled'):
        models.pull(c, ask=lambda _: 'n', get=lambda u: tree, fetch=fetch)
    assert calls == []                                   # nothing downloaded without a yes
    path = models.pull(c, yes=True, get=lambda u: tree, fetch=fetch)
    assert 'f' * 40 in calls[0] and '/resolve/' in calls[0]   # pinned to the commit, not 'main'
    assert open(path + '.provenance.json').read().count(good) == 1
