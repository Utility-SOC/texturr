import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

import llm
import texturr


class FakeLLM(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({'data': [{'id': 'tiny-model'}]}).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        FakeLLM.seen.append((self.path, dict(self.headers), payload))
        reply = {'label': 'Slow loading', 'summary': 'Pages are slow.', 'action': 'Profile load time.'}
        if self.path.endswith('/messages'):
            body = {'content': [{'type': 'text', 'text': json.dumps(reply)}]}
        else:
            body = {'choices': [{'message': {'content': 'Sure:\n' + json.dumps(reply)}}]}
        out = json.dumps(body).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(out)


@pytest.fixture
def server():
    FakeLLM.seen.clear()
    srv = HTTPServer(('127.0.0.1', 0), FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{srv.server_port}/v1'
    srv.shutdown()


def test_remote_provider_blocked_without_flag():
    cfg = llm.resolve_config('openai', environ={'OPENAI_API_KEY': 'k'})
    with pytest.raises(llm.LLMError, match='--allow-remote'):
        llm.check_policy(cfg, allow_remote=False, offline=False)
    llm.check_policy(cfg, allow_remote=True, offline=False)


def test_offline_forbids_remote_even_with_allow_remote():
    cfg = llm.resolve_config('anthropic', environ={'ANTHROPIC_API_KEY': 'k'})
    with pytest.raises(llm.LLMError, match='--offline'):
        llm.check_policy(cfg, allow_remote=True, offline=True)


def test_loopback_is_local_and_lan_is_not():
    assert llm.is_local_url('http://localhost:11434/v1')
    assert llm.is_local_url('http://127.0.0.1:8080/v1')
    assert not llm.is_local_url('http://192.168.1.5:8080/v1')
    assert not llm.is_local_url('https://api.openai.com/v1')


def test_missing_key_names_env_var_and_never_reads_cli():
    with pytest.raises(llm.LLMError, match='GEMINI_API_KEY'):
        llm.resolve_config('gemini', environ={})


def test_remote_needs_model_when_no_default():
    with pytest.raises(llm.LLMError, match='--model'):
        llm.resolve_config('openrouter', environ={'OPENROUTER_API_KEY': 'k'})


def test_auto_picks_first_responding_local_server():
    def probe(url):
        if '1234' in url:
            return ['qwen']
        raise llm.LLMError('down')
    cfg = llm.resolve_config('auto', probe=probe)
    assert cfg.preset.name == 'lmstudio' and cfg.model == 'qwen' and cfg.is_local


def test_auto_returns_none_when_nothing_local():
    def probe(url):
        raise llm.LLMError('down')
    assert llm.resolve_config('auto', probe=probe) is None


def test_openai_compatible_roundtrip(server):
    cfg = llm.resolve_config('openai-compatible', base_url=server, model='m')
    out = llm.label_cluster(cfg, ['too slow', 'slow load'], ['slow'])
    assert out['label'] == 'Slow loading'
    path, headers, payload = FakeLLM.seen[0]
    assert path.endswith('/chat/completions') and 'Authorization' not in headers
    assert '<answers>' in payload['messages'][1]['content']


def test_anthropic_wire_format(server):
    cfg = llm.LLMConfig(llm.PRESETS['anthropic'], server, 'claude-haiku-4-5-20251001', 'sekret')
    assert llm.label_cluster(cfg, ['x'], [])['label'] == 'Slow loading'
    path, headers, payload = FakeLLM.seen[0]
    assert path.endswith('/messages') and {k.lower(): v for k, v in headers.items()}['x-api-key'] == 'sekret'
    assert payload['system'] and payload['messages'][0]['role'] == 'user'


def test_parse_label_rejects_garbage():
    assert llm.parse_label('no json here') is None
    assert llm.parse_label('{"summary": "x"}') is None


def test_csv_safe_neutralizes_formulas():
    assert texturr.csv_safe('=HYPERLINK("http://x")').startswith("'=")
    assert texturr.csv_safe('@SUM(A1)').startswith("'@")
    assert texturr.csv_safe('fine') == 'fine'


def test_report_with_and_without_llm(server):
    data = ['too slow', 'slow load', 'love it', 'great app']
    clusters = [0, 0, 1, 1]
    emb = np.array([[0, 0], [0, 1], [5, 5], [5, 6]], dtype=float)
    rows = texturr.build_report(data, clusters, {0: ['slow']}, emb)
    assert rows[0]['Label'] == '' and rows[0]['Size'] == 2 and rows[0]['Keyphrases'] == 'slow'
    cfg = llm.resolve_config('openai-compatible', base_url=server, model='m')
    rows = texturr.build_report(data, clusters, {}, emb, cfg)
    assert rows[1]['Label'] == 'Slow loading' and rows[1]['Responses'] == '3, 4'
