import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'n8n'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'n8n', 'testing'))
import build_workflows as bw  # noqa: E402
import fake_llm  # noqa: E402
import server  # noqa: E402

BASE = 'http://texturr:8765'
BUILDERS = {'files': bw.files_workflow, 'form': bw.form_workflow}


def build(kind, chat='ollama', base=BASE):
    return BUILDERS[kind](base, chat, {})


@pytest.mark.parametrize('kind', BUILDERS)
@pytest.mark.parametrize('chat', bw.CHAT_MODELS)
def test_connections_reference_real_nodes_and_ids_are_unique(kind, chat):
    wf = build(kind, chat)
    names = [n['name'] for n in wf['nodes']]
    assert len(names) == len(set(names)) and len({n['id'] for n in wf['nodes']}) == len(names)
    for src, kinds in wf['connections'].items():
        assert src in names
        for kind_, branches in kinds.items():
            for branch in branches:
                for link in branch:
                    assert link['node'] in names and link['type'] == kind_
    ai = wf['connections']
    assert ai['Chat model']['ai_languageModel'][0][0]['node'] == 'AI Agent'
    assert ai['texturr tools']['ai_tool'][0][0]['node'] == 'AI Agent'


@pytest.mark.parametrize('kind', BUILDERS)
def test_every_node_is_reachable_from_the_trigger(kind):
    wf = build(kind)
    trigger = wf['nodes'][0]['name']
    seen, todo = {trigger}, [trigger]
    while todo:
        for branches in wf['connections'].get(todo.pop(), {}).get('main', []):
            for link in branches:
                if link['node'] not in seen:
                    seen.add(link['node']); todo.append(link['node'])
    main_nodes = {n['name'] for n in wf['nodes']} - {'Chat model', 'texturr tools'}
    assert seen == main_nodes


def test_no_secrets_in_generated_json_only_credential_references():
    for kind in BUILDERS:
        for chat in bw.CHAT_MODELS:
            text = json.dumps(build(kind, chat))
            assert not re.search(r'sk-[A-Za-z0-9]{10,}|Bearer [A-Za-z0-9_-]{10,}|apiKey', text)
    wf = build('files')
    creds = {n['name']: n.get('credentials') for n in wf['nodes'] if n.get('credentials')}
    assert creds['Create session'] == bw.TOKEN_CRED and creds['texturr tools'] == bw.TOKEN_CRED


def test_service_urls_follow_the_base_url():
    wf = build('files', base='http://127.0.0.1:9999')
    mcp = [n for n in wf['nodes'] if n['name'] == 'texturr tools'][0]
    assert mcp['parameters']['endpointUrl'] == 'http://127.0.0.1:9999/mcp'
    assert mcp['parameters']['serverTransport'] == 'httpStreamable'
    urls = [n['parameters']['url'] for n in wf['nodes'] if n['type'].endswith('httpRequest')]
    assert urls and all('127.0.0.1:9999' in u for u in urls)


def test_expressions_are_balanced():
    def walk(x):
        if isinstance(x, str):
            yield x
        elif isinstance(x, dict):
            for v in x.values():
                yield from walk(v)
        elif isinstance(x, list):
            for v in x:
                yield from walk(v)
    for kind in BUILDERS:
        for s in walk(build(kind)):
            assert s.count('{{') == s.count('}}'), s
            if '{{' in s:
                assert s.startswith('='), f'expression without leading = : {s[:60]}'


def test_system_prompt_mentions_every_server_tool():
    for tool in server.TOOLS:
        assert tool['name'] in bw.SYSTEM_PROMPT, tool['name']
    assert 'untrusted' in bw.SYSTEM_PROMPT and 'em dash' in bw.SYSTEM_PROMPT


def test_committed_workflows_match_the_builder():
    here = os.path.join(os.path.dirname(__file__), '..', 'n8n', 'workflows')
    for fname, kind in (('texturr-agentic-files.json', 'files'), ('texturr-agentic-form.json', 'form')):
        committed = json.load(open(os.path.join(here, fname)))
        assert committed == build(kind), f'{fname} is stale: run python n8n/build_workflows.py'


def test_chat_model_override():
    wf = bw.files_workflow(BASE, 'openai', {'model': {'__rl': True, 'mode': 'id', 'value': 'granite'}})
    model = [n for n in wf['nodes'] if n['name'] == 'Chat model'][0]
    assert model['parameters']['model']['value'] == 'granite'
    assert model['credentials'] == {'openAiApi': {'id': 'openai-credential', 'name': 'OpenAI-compatible'}}


# --- the scripted LLM used for integration testing ---------------------------------------------------

TOOLS = [{'type': 'function', 'function': {'name': 'texturr_tools_' + t['name']}} for t in server.TOOLS]


def overview(ids, pair=None):
    return {'clusters': [{'id': i, 'size': 4, 'top_terms': ['x']} for i in ids],
            'hints': {'most_similar_pairs': [{'clusters': pair}] if pair else []}}


def assistant(call_id, name):
    return {'role': 'assistant', 'tool_calls': [{'id': call_id, 'function': {'name': 'texturr_tools_' + name}}]}


def tool_msg(call_id, payload, wrap=True):
    body = json.dumps(payload)
    return {'role': 'tool', 'tool_call_id': call_id,
            'content': json.dumps([{'response': [{'type': 'text', 'text': body}]}]) if wrap else body}


def test_fake_llm_policy_walks_the_full_loop():
    msgs = [{'role': 'user', 'content': 'Session id: abc123'}]
    kind, calls = fake_llm.plan(msgs, TOOLS)
    assert calls == [('texturr_tools_get_overview', {'session_id': 'abc123'})]
    msgs += [assistant('1', 'get_overview'), tool_msg('1', overview([0, 1, 2], [0, 2]))]
    assert fake_llm.plan(msgs, TOOLS)[1][0][0] == 'texturr_tools_get_cluster_examples'
    msgs += [assistant('2', 'get_cluster_examples'), tool_msg('2', {'examples': []})]
    kind, calls = fake_llm.plan(msgs, TOOLS)
    assert calls[0] == ('texturr_tools_merge_clusters', {'session_id': 'abc123', 'cluster_ids': [0, 2]})
    msgs += [assistant('3', 'merge_clusters'), tool_msg('3', {'merged_into': 0})]
    assert fake_llm.plan(msgs, TOOLS)[1][0][0] == 'texturr_tools_get_overview'
    msgs += [assistant('4', 'get_overview'), tool_msg('4', overview([0, 1]))]
    kind, calls = fake_llm.plan(msgs, TOOLS)
    assert [c[0] for c in calls] == ['texturr_tools_set_cluster_label'] * 2          # one parallel batch, one per group
    msgs += [assistant('5', 'set_cluster_label'), tool_msg('5', {})]
    assert fake_llm.plan(msgs, TOOLS)[1][0][0] == 'texturr_tools_finish_analysis'
    msgs += [assistant('6', 'finish_analysis'), tool_msg('6', {'finished': True})]
    assert fake_llm.plan(msgs, TOOLS)[0] == 'text'


def test_fake_llm_streams_valid_sse_for_tool_calls():
    full = fake_llm.completion('m', 'tool_calls', [('texturr_tools_get_overview', {'session_id': 's'})])
    chunks = list(fake_llm.stream_chunks('m', full))
    args = ''.join(c['choices'][0]['delta']['tool_calls'][0]['function'].get('arguments', '')
                   for c in chunks if c['choices'][0]['delta'].get('tool_calls'))
    assert json.loads(args) == {'session_id': 's'} and chunks[-1]['choices'][0]['finish_reason'] == 'tool_calls'
