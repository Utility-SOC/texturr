"""Vetted local-model recommendations from the live Hugging Face Hub, plus a pinned, verified download.

No maintained Hugging Face leaderboard covers summarization (the Open LLM Leaderboard
was frozen in March 2025), so the list is built from live Hub data with explicit vetting
rules, ranked by recent downloads. That measures adoption, not quality; the vetting
rules are what make a candidate defensible for institutional use:

  * published by an allowlisted organization (first-party GGUF, not a third-party re-quant),
    and derived only from base models by allowlisted organizations
  * permissive OSI license (Apache-2.0 or MIT by default)
  * a text-generation model of known size within a budget, recently maintained, chat/instruct tuned
  * downloads are pinned to a commit SHA and checked against the Hub's SHA-256
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

HUB = 'https://huggingface.co'

# First-party publishers with a record of permissively licensed releases. Chinese-origin
# publishers (Qwen, DeepSeek, ...) are deliberately not in the default list because many
# government procurement policies restrict them; add them with --orgs if your policy allows.
DEFAULT_ORGS = ('ibm-granite', 'microsoft', 'mistralai', 'allenai', 'HuggingFaceTB', 'openai', 'nvidia')
DEFAULT_LICENSES = ('apache-2.0', 'mit')
SHARD = re.compile(r'-(\d{5})-of-(\d{5})\.gguf$', re.I)


class ModelsError(Exception):
    pass


@dataclass
class Candidate:
    repo: str
    org: str
    license: str
    params_b: float
    downloads: int
    likes: int
    updated: str
    sha: str


def _get_json(url):
    headers = {}
    token = os.environ.get('HF_TOKEN')
    if token and url.startswith(HUB):
        headers['Authorization'] = f'Bearer {token}'
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as resp:
            return json.loads(resp.read().decode())
    except OSError as e:
        raise ModelsError(f"Could not reach {url}: {e}")


def _parse_time(text):
    return datetime.fromisoformat(text.replace('Z', '+00:00'))


def vet(raw, licenses=DEFAULT_LICENSES, max_params_b=14.0, max_age_days=540, now=None, orgs=DEFAULT_ORGS):
    """Return a Candidate if a Hub model record passes every rule, else None."""
    now = now or datetime.now(timezone.utc)
    tags = raw.get('tags') or []
    repo = raw.get('id', '')
    allowed = {o.lower() for o in orgs}
    for t in tags:       # provenance: a derivative of a non-allowlisted publisher's model is rejected too
        m = re.match(r'base_model:(?:\w+:)?([^/]+)/', t)
        if m and m.group(1).lower() not in allowed:
            return None
    found = [t.split(':', 1)[1] for t in tags if t.startswith('license:')]
    if not found or found[0] not in licenses:
        return None
    if raw.get('pipeline_tag') not in (None, 'text-generation'):
        return None          # excludes speech, vision, embedding and other non-chat models
    total = (raw.get('gguf') or {}).get('total')
    if not total or total / 1e9 > max_params_b:
        return None          # unknown size is rejected, not guessed
    if re.search(r'(^|[-_/])base($|[-_])', repo, re.I) or not ('conversational' in tags or re.search(r'instruct|chat', repo, re.I)):
        return None
    updated = raw.get('lastModified', '')
    try:
        if now - _parse_time(updated) > timedelta(days=max_age_days):
            return None
    except ValueError:
        return None
    if not raw.get('sha'):
        return None
    return Candidate(repo, raw.get('author') or repo.split('/')[0], found[0], round(total / 1e9, 1),
                     int(raw.get('downloads') or 0), int(raw.get('likes') or 0), updated[:10], raw['sha'])


def fetch_org(org, limit=25, get=_get_json):
    url = (f"{HUB}/api/models?filter=gguf&author={org}&sort=downloads&direction=-1&limit={limit}"
           "&expand[]=downloads&expand[]=likes&expand[]=gguf&expand[]=tags&expand[]=lastModified"
           "&expand[]=sha&expand[]=author&expand[]=pipeline_tag")
    return get(url)


def recommend(orgs=DEFAULT_ORGS, licenses=DEFAULT_LICENSES, max_params_b=14.0, max_age_days=540,
              top=5, per_org=2, get=_get_json, now=None):
    """Top vetted candidates by 30-day downloads, at most `per_org` from any one publisher."""
    pool, failures = [], []
    for org in orgs:
        try:
            for raw in fetch_org(org, get=get):
                c = vet(raw, licenses, max_params_b, max_age_days, now, orgs)
                if c:
                    pool.append(c)
        except ModelsError as e:
            failures.append(str(e))
    if not pool and failures:
        raise ModelsError(failures[0])
    pool.sort(key=lambda c: c.downloads, reverse=True)
    out, count = [], {}
    for c in pool:
        if count.get(c.org, 0) < per_org:
            out.append(c)
            count[c.org] = count.get(c.org, 0) + 1
        if len(out) == top:
            break
    return out


def pick_files(repo, sha, quant='Q4_K_M', get=_get_json):
    """GGUF file(s) of the requested quantization at an exact commit.

    Returns a list of {'path','sha256','size'}: one entry normally, or every shard of a split
    model in order (llama.cpp loads the whole set when given the first shard).
    """
    tree = get(f"{HUB}/api/models/{repo}/tree/{sha}")
    ggufs = [f for f in tree if f.get('path', '').lower().endswith('.gguf')]
    hits = [f for f in ggufs if quant.lower() in f['path'].lower()]
    single = [f for f in hits if not SHARD.search(f['path'])]
    if single:
        chosen = [min(single, key=lambda f: len(f['path']))]
    else:
        groups = {}
        for f in hits:
            m = SHARD.search(f['path'])
            if m:
                groups.setdefault((f['path'][:m.start()], int(m.group(2))), []).append((int(m.group(1)), f))
        complete = [sorted(v) for (_, total), v in groups.items() if sorted(i for i, _ in v) == list(range(1, total + 1))]
        if not complete:
            names = ', '.join(sorted(f['path'] for f in ggufs)) or 'none'
            raise ModelsError(f"No complete {quant} GGUF in {repo}@{sha[:8]}. Available: {names}")
        chosen = [f for _, f in complete[0]]
    out = []
    for f in chosen:
        lfs = f.get('lfs') or {}
        if not lfs.get('oid'):
            raise ModelsError(f"{f['path']} has no SHA-256 on the Hub; refusing to download an unverifiable file.")
        out.append({'path': f['path'], 'sha256': lfs['oid'], 'size': lfs.get('size') or f.get('size', 0)})
    return out


def download_verified(url, dest, sha256, size=0, report=sys.stderr):
    """Stream to dest, verify SHA-256, and only then move into place. Deletes the file on mismatch."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + '.part'
    digest, done = hashlib.sha256(), 0
    with urllib.request.urlopen(urllib.request.Request(url), timeout=60) as resp, open(part, 'wb') as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if size and report:
                report.write(f"\r  {done / 1e9:.2f} / {size / 1e9:.2f} GB")
    if report:
        report.write("\n")
    if digest.hexdigest() != sha256:
        os.remove(part)
        raise ModelsError("SHA-256 mismatch: the download does not match the Hub's recorded hash; it was deleted.")
    os.replace(part, dest)
    return dest


def cache_dir():
    return os.path.join(os.environ.get('TEXTURR_HOME', os.path.expanduser('~/.cache/texturr')), 'models')


def pull(c, quant='Q4_K_M', yes=False, ask=input, get=_get_json, fetch=download_verified):
    """Consent, download at the pinned commit, verify, and record provenance. Returns the path to load
    (the first shard when the model is split)."""
    files = pick_files(c.repo, c.sha, quant, get)
    folder = os.path.join(cache_dir(), c.repo.replace('/', '__'), c.sha)
    total = sum(f['size'] for f in files)
    print(f"\n  Model:     {c.repo}  ({c.params_b}B parameters, {quant})\n"
          f"  Publisher: {c.org}   License: {c.license}\n"
          f"  Revision:  {c.sha}  (pinned)\n"
          f"  Source:    {HUB}/{c.repo}/resolve/{c.sha}/\n"
          f"  Files:     {', '.join(f['path'] for f in files)}\n"
          f"  Size:      {total / 1e9:.1f} GB   Destination: {folder}\n"
          f"  Integrity: SHA-256 of each file will be verified against the Hub's record")
    if not yes and ask(f"Download {total / 1e9:.1f} GB from huggingface.co? [y/N] ").strip().lower() != 'y':
        raise ModelsError("Download cancelled.")
    record = []
    for f in files:
        url = f"{HUB}/{c.repo}/resolve/{c.sha}/{f['path']}"
        dest = os.path.join(folder, os.path.basename(f['path']))
        if not (os.path.exists(dest) and _sha256_file(dest) == f['sha256']):
            fetch(url, dest, f['sha256'], f['size'])
        record.append({'file': f['path'], 'sha256': f['sha256'], 'source': url})
    first = os.path.join(folder, os.path.basename(files[0]['path']))
    with open(first + '.provenance.json', 'w') as out:
        json.dump({'repo': c.repo, 'revision': c.sha, 'license': c.license, 'files': record,
                   'downloaded_at': datetime.now(timezone.utc).isoformat()}, out, indent=2)
    return first


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def serve_hint(path):
    return (f"Start a local server (data stays on this machine), then re-run texturr:\n"
            f"  llama-server -m '{path}' --jinja --host 127.0.0.1 --port 8080\n"
            f"texturr finds it automatically with --llm auto (or --llm llamacpp).")


def serve_command(path, port=8080, which=shutil.which):
    """argv for a loopback-only llama.cpp server. Raises if llama-server is not installed."""
    exe = which('llama-server')
    if not exe:
        raise ModelsError("llama-server was not found on PATH. Install llama.cpp (https://github.com/ggml-org/llama.cpp), "
                          "or use Ollama/LM Studio with the downloaded file.")
    if not os.path.isfile(path):
        raise ModelsError(f"No such model file: {path}")
    return [exe, '-m', path, '--jinja', '--host', '127.0.0.1', '--port', str(port)]


def format_table(cands):
    lines = [f"{'#':>2}  {'model':<46} {'size':>6}  {'license':<11} {'30d downloads':>13}  updated"]
    for i, c in enumerate(cands, 1):
        lines.append(f"{i:>2}  {c.repo:<46} {c.params_b:>5}B  {c.license:<11} {c.downloads:>13,}  {c.updated}")
    return '\n'.join(lines)


def build_parser():
    p = argparse.ArgumentParser(prog='texturr.py models',
                                description='List vetted local models from the Hugging Face Hub and optionally download one.')
    p.add_argument('--orgs', nargs='+', default=list(DEFAULT_ORGS), help='Allowed publishers')
    p.add_argument('--licenses', nargs='+', default=list(DEFAULT_LICENSES), help='Allowed license tags')
    p.add_argument('--max-params-b', type=float, default=14.0, help='Largest model to consider, in billions of parameters')
    p.add_argument('--max-age-days', type=int, default=540, help='Skip repos not updated within this many days')
    p.add_argument('--top', type=int, default=5)
    p.add_argument('--pull', type=int, metavar='N', help='Download item N from the list')
    p.add_argument('--quant', default='Q4_K_M', help='GGUF quantization to download (default Q4_K_M)')
    p.add_argument('--yes', action='store_true', help='Skip the download confirmation')
    p.add_argument('--serve', nargs='?', const='', metavar='FILE',
                   help='Start a loopback-only llama-server for FILE, or for the model just pulled with --pull')
    p.add_argument('--port', type=int, default=8080, help='Port for --serve (default 8080)')
    return p


def cli(argv, get=_get_json):
    args = build_parser().parse_args(argv)
    if os.environ.get('HF_HUB_OFFLINE') == '1':
        print("Offline mode is set; the live model list needs the network.", file=sys.stderr)
        return 2
    try:
        if args.serve is not None and not args.pull:
            if not args.serve:
                raise ModelsError("--serve needs a model file, or use it with --pull N.")
            cmd = serve_command(args.serve, args.port)
            os.execv(cmd[0], cmd)
        cands = recommend(args.orgs, args.licenses, args.max_params_b, args.max_age_days, args.top, get=get)
        if not cands:
            print("No models passed the vetting rules; loosen --licenses, --orgs or --max-params-b.", file=sys.stderr)
            return 1
        print(format_table(cands))
        print("\nRanked by 30-day downloads among vetted models (adoption, not a quality benchmark).")
        if args.pull:
            if not 1 <= args.pull <= len(cands):
                print(f"--pull must be between 1 and {len(cands)}.", file=sys.stderr)
                return 2
            path = pull(cands[args.pull - 1], args.quant, args.yes, get=get)
            if args.serve is not None:
                cmd = serve_command(path, args.port)
                print("Starting:", ' '.join(cmd), "\n(Ctrl+C to stop; run texturr from another terminal.)")
                os.execv(cmd[0], cmd)
            print(serve_hint(path))
        else:
            print("Download one with: python3 texturr.py models --pull N")
        return 0
    except ModelsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


def first_run_offer(get=_get_json, ask=input):
    """Interactive first-run help when no local LLM server exists. Never downloads without consent."""
    print("\nNo local LLM server found. A local model lets texturr label clusters without any data leaving this machine.")
    try:
        cands = recommend(get=get)
    except ModelsError as e:
        print(f"(Could not fetch the vetted model list: {e})")
        return None
    if not cands:
        return None
    print(format_table(cands))
    print("Ranked by 30-day downloads among vetted publishers/licenses (adoption, not a quality benchmark).")
    choice = ask("Download one now? Enter its number, or press Enter to skip: ").strip()
    if not choice.isdigit() or not 1 <= int(choice) <= len(cands):
        print("Skipping. Run `python3 texturr.py models` any time.")
        return None
    try:
        path = pull(cands[int(choice) - 1], ask=ask, get=get)
    except ModelsError as e:
        print(f"ERROR: {e}")
        return None
    print(serve_hint(path))
    return path
