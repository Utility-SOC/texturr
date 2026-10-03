"""A scripted, OpenAI-compatible chat server for testing the n8n wiring. It is NOT a model.

It follows a fixed policy so a workflow can be run end to end without any LLM: look at the
overview, read a group's examples, merge the two most similar groups, label every group in
one parallel batch of tool calls, then finish. That exercises the agent loop, MCP tool
discovery, argument passing, parallel tool calls and streaming. It says nothing about how a
real model would judge the groups.

    python n8n/testing/fake_llm.py --port 11500
"""

import argparse
import json
import re
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = []


def tool_name(tools, suffix):
    for t in tools:
        name = t.get('function', {}).get('name', '')
        if name == suffix or name.endswith(suffix):
            return name
    return None


def unwrap(body):
    """Dig texturr's JSON out of however n8n wraps an MCP result, e.g.
    [{"response": [{"type": "text", "text": "{...}"}]}]."""
    try:
        data = json.loads(body) if isinstance(body, str) else body
    except ValueError:
        return None
    for _ in range(8):
        if isinstance(data, list) and data:
            data = data[0]
        elif isinstance(data, dict) and 'response' in data:
            data = data['response']
        elif isinstance(data, dict) and data.get('type') == 'text' and 'text' in data:
            try:
                data = json.loads(data['text'])
            except ValueError:
                return data['text']
        else:
            return data
    return data


def plan(messages, tools):
    """Return ('tool_calls', [(name, args), ...]) or ('text', str) for the next assistant turn."""
    text = ' '.join(m['content'] if isinstance(m.get('content'), str) else json.dumps(m.get('content')) for m in messages if m.get('role') == 'user')
    sid = (re.search(r'Session id:\s*([\w-]+)', text) or [None, ''])[1]
    results = []
    calls = {}
    for m in messages:
        if m.get('role') == 'assistant':
            for c in m.get('tool_calls') or []:
                calls[c['id']] = c['function']['name']
        if m.get('role') == 'tool':
            body = m['content'] if isinstance(m['content'], str) else json.dumps(m['content'])
            results.append((calls.get(m.get('tool_call_id'), ''), body))
    done = [n for n, _ in results]
    has = lambda suffix: any(n.endswith(suffix) for n in done)
    last_overview = None
    for n, body in results:
        if n.endswith('get_overview'):
            parsed = unwrap(body)
            if isinstance(parsed, dict) and 'clusters' in parsed:
                last_overview = parsed
    if not results:
        return 'tool_calls', [(tool_name(tools, 'get_overview'), {'session_id': sid})]
    if not has('get_cluster_examples') and last_overview:
        return 'tool_calls', [(tool_name(tools, 'get_cluster_examples'), {'session_id': sid, 'cluster_id': last_overview['clusters'][0]['id'], 'n': 3})]
    if not has('merge_clusters') and last_overview and last_overview['hints']['most_similar_pairs']:
        pair = last_overview['hints']['most_similar_pairs'][0]['clusters']
        return 'tool_calls', [(tool_name(tools, 'merge_clusters'), {'session_id': sid, 'cluster_ids': pair})]
    if has('merge_clusters') and sum(1 for n in done if n.endswith('get_overview')) < 2:
        return 'tool_calls', [(tool_name(tools, 'get_overview'), {'session_id': sid})]
    if not has('set_cluster_label') and last_overview:
        return 'tool_calls', [(tool_name(tools, 'set_cluster_label'),
                               {'session_id': sid, 'cluster_id': c['id'], 'label': f"Theme {c['id']}",
                                'summary': f"{c['size']} responses about {', '.join(c['top_terms'][:2]) or 'misc'}",
                                'suggested_action': 'Review with the team.'}) for c in last_overview['clusters']]
    if not has('finish_analysis'):
        return 'tool_calls', [(tool_name(tools, 'finish_analysis'), {'session_id': sid})]
    return 'text', 'Analysis finished. All groups were reviewed, merged where similar, and labeled.'


def completion(model, kind, payload):
    msg = {'role': 'assistant', 'content': payload if kind == 'text' else None}
    if kind == 'tool_calls':
        msg['tool_calls'] = [{'id': 'call_' + uuid.uuid4().hex[:8], 'type': 'function',
                              'function': {'name': n, 'arguments': json.dumps(a)}} for n, a in payload]
    return {'id': 'chatcmpl-' + uuid.uuid4().hex[:8], 'object': 'chat.completion', 'model': model,
            'choices': [{'index': 0, 'message': msg, 'finish_reason': 'stop' if kind == 'text' else 'tool_calls'}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}


def stream_chunks(model, full):
    base = {'id': full['id'], 'object': 'chat.completion.chunk', 'model': model}
    msg = full['choices'][0]['message']
    yield {**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]}
    if msg.get('tool_calls'):
        for i, c in enumerate(msg['tool_calls']):
            yield {**base, 'choices': [{'index': 0, 'finish_reason': None, 'delta': {'tool_calls': [
                {'index': i, 'id': c['id'], 'type': 'function', 'function': {'name': c['function']['name'], 'arguments': ''}}]}}]}
            yield {**base, 'choices': [{'index': 0, 'finish_reason': None, 'delta': {'tool_calls': [
                {'index': i, 'function': {'arguments': c['function']['arguments']}}]}}]}
    else:
        yield {**base, 'choices': [{'index': 0, 'delta': {'content': msg['content']}, 'finish_reason': None}]}
    yield {**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': full['choices'][0]['finish_reason']}]}


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip('/').endswith('/models'):
            return self._send(200, json.dumps({'object': 'list', 'data': [{'id': 'fake-agent', 'object': 'model'}]}).encode())
        self._send(404, b'{}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
        if not self.path.rstrip('/').endswith('/chat/completions'):
            return self._send(404, b'{}')
        kind, payload = plan(body.get('messages', []), body.get('tools', []))
        LOG.append((kind, [n for n, _ in payload] if kind == 'tool_calls' else 'text'))
        print('fake-llm:', LOG[-1], file=sys.stderr, flush=True)
        full = completion(body.get('model', 'fake'), kind, payload)
        if body.get('stream'):
            data = ''.join(f"data: {json.dumps(c)}\n\n" for c in stream_chunks(body.get('model', 'fake'), full)) + 'data: [DONE]\n\n'
            return self._send(200, data.encode(), 'text/event-stream')
        self._send(200, json.dumps(full).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=11500)
    args = ap.parse_args()
    print(f'fake LLM on http://{args.host}:{args.port}/v1', file=sys.stderr, flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
