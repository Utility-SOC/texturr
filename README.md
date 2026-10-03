# texturr

Summarize and group free-form answers in spreadsheets, **local-first**.

texturr reads a column (or row) of free-text answers from an Excel or CSV file, groups
similar answers with sentence embeddings, has an LLM name and summarize each group, and
writes a CSV, an optional shareable HTML report, and an optional copy of your
spreadsheet with the group written beside every answer.

It is built for sensitive data. By default everything runs on your machine: embeddings
use a local model and labeling uses a local LLM server. Sending text to a hosted
provider is possible but off unless you pass `--allow-remote`, and `--offline` forbids
it outright.

## Install

```bash
pip install -r requirements.txt        # or: pip install .   (adds a `texturr` command)
```

For LLM labels you need a local model server. texturr can recommend and download a
vetted model for you (see [Picking a local model](#picking-a-local-model-models)).
With no server found it still works and fills in keyphrases only. No provider SDKs are needed.

## Quick start

```bash
# Try the bundled example (24 customer comments)
python3 texturr.py examples/feedback.csv --column Comments --llm none --html report.html

# Interactive: lists the columns (headers + a sample answer) and asks which to group
python3 texturr.py survey.xlsx

# Scripted, several columns at once, with a report and an annotated copy of the sheet
python3 texturr.py survey.xlsx --sheet "Q3 Results" --column Likes Dislikes \
    --html report.html --annotated survey_grouped.xlsx
```

## Options

| Option | Meaning |
| --- | --- |
| `filename` | `.xlsx`, `.csv` or `.tsv`/`.tab` file. |
| `--column` | One or more columns to group: letters (`C`) or header names (`Comments`, case-insensitive). Each column is analyzed separately. Prompts if omitted (comma-separate several). |
| `--row N` | Group the answers in row N instead of a column. |
| `--sheet` | Sheet name or 1-based number (Excel only). Prompts if omitted and the workbook has several. |
| `--header-row N` | Row holding column headers (default `1`; `0` means no header row). Headers name the columns and are excluded from the answers. |
| `--clusters` | `auto` (default) picks the group count by silhouette score (2 to 12), or give a number. |
| `--keep-nonanswers` | Cluster "N/A", "none", "idk" and similar too. By default they are set aside as one group (cluster `-1`). |
| `--output` | Summary CSV path (default `summary_output.csv`). |
| `--html PATH` | Also write a self-contained HTML report (no scripts, no external assets, works air-gapped). |
| `--annotated PATH` | Also write a copy of the input with `<column> Cluster` and `<column> Theme` columns added beside the data (column analysis only). |
| `--llm` | Labeling provider: `auto` (default, first local server found), `none`, or one of `ollama`, `llamacpp`, `lmstudio`, `anthropic`, `openai`, `gemini`, `mistral`, `groq`, `openrouter`, `openai-compatible`. |
| `--model` | Model name. Local servers default to the model they have loaded; hosted providers have a default you can override. |
| `--base-url` | Override the provider URL (required for `openai-compatible`, e.g. vLLM). |
| `--api-key-env` | Name of the environment variable holding the key (default per provider). |
| `--allow-remote` | Permit sending sampled cluster text to a remote provider. |
| `--offline` | Air-gapped mode: no network except a loopback LLM server; remote providers are refused. |
| `--embedding-model` | Sentence-transformers model name or local directory (default `all-MiniLM-L6-v2`). |
| `--sample-size` | Responses per cluster shown to the LLM (default 8). |

## Output

**Summary CSV**, one row per group: `Column`, `Cluster`, `Size`, `Label` (LLM theme
name), `Summary`, `Suggested Action`, `Keyphrases`, `Representative Responses` (the
three closest to the group's center) and `Responses` (1-based positions of the answers
in the group). `Label`, `Summary` and `Suggested Action` are empty when no LLM is used. Blank or boilerplate
answers ("N/A", "none", "idk", symbols only) form their own group with cluster `-1`, labeled
"Non-answers", instead of distorting the real groups. Only genuinely empty cells are treated as
missing; the literal text "N/A" is kept and counted (pandas would otherwise drop it silently).

**HTML report:** one card per group, largest first, with a size bar, summary, suggested
action, keyphrases and representative quotes. Every value is HTML-escaped, the page
carries a `default-src 'none'` content security policy, and it follows the system light
or dark theme.

**Annotated copy:** your original workbook (formatting preserved) or CSV with two new
columns per analyzed column. Without an LLM the theme falls back to the top keyphrases.
New headers go on the header row; with `--header-row 0` they are omitted rather than
overwrite your data.

## Agentic mode: an AI agent that refines the grouping (n8n)

Automatic grouping tends to split one theme into several groups, and no statistic fixes that (see
[Evaluation](#evaluation)). So texturr can run as a local service that an AI agent drives:

```bash
cp .env.example .env && docker compose up -d --build && ./n8n/bootstrap.sh
```

The agent, built as an n8n workflow, reads the groups, merges those that mean the same thing,
splits mixed ones, moves misplaced answers, labels every group from what it read, and returns your
spreadsheet with a `Cluster` and `Theme` column beside each answer. Two workflows ship: an upload
form for people and a file-based one for batch use. Everything runs on your machine with a local
model by default. Full setup, security notes and what was and was not tested: **[n8n/README.md](n8n/README.md)**.

The service (`python3 texturr.py serve`) is useful on its own: REST endpoints to create a session from
a list of answers or an uploaded file and download the report (`json`, `csv`, `html`, or an
annotated copy of the upload), plus an MCP endpoint (`/mcp`) exposing `get_overview`,
`get_cluster_examples`, `merge_clusters`, `split_cluster`, `move_responses`, `set_cluster_label` and
`finish_analysis`, so any MCP client can use it. It binds loopback by default, requires a bearer
token, keeps sessions in memory only (nothing on disk), expires them, caps request sizes, rejects
zip bombs, and never logs request bodies or response text. Every merge, split, move and label is kept in the session
history, which is returned with the JSON report as an audit trail.

## Picking a local model (`models`)

No local model yet? texturr builds a live shortlist from the Hugging Face Hub:

```bash
python3 texturr.py models                 # show the top 5 vetted models
python3 texturr.py models --pull 1        # download #1 (asks first)
python3 texturr.py models --pull 1 --serve   # ...then start a loopback-only llama-server
python3 texturr.py models --serve path/to/model.gguf
```

The first interactive run with no local server found offers the same list. Nothing is
downloaded without an explicit yes.

**Why not a leaderboard?** No maintained Hugging Face leaderboard covers summarization
(the Open LLM Leaderboard was frozen in March 2025), so the list comes from live Hub
data (downloads, licenses, sizes, commit SHAs), ranked by 30-day downloads. That
measures adoption, not quality. The vetting rules are what make a candidate defensible,
and they are all applied in code you can read (`models.py`):

- **Publisher allowlist**, first-party GGUF repos only (no third-party re-quants):
  `ibm-granite`, `microsoft`, `mistralai`, `allenai`, `HuggingFaceTB`, `openai`, `nvidia`.
  Override with `--orgs`. Chinese-origin publishers (Qwen, DeepSeek) are left off by
  default because many government procurement policies restrict them; add them if your
  policy allows.
- **Provenance:** a model must also be derived only from base models by allowlisted
  publishers, so a fine-tune of someone else's model cannot ride in on an allowed name.
- **Permissive license only:** Apache-2.0 or MIT (`--licenses`). Custom community
  licenses (Llama, Gemma) are excluded.
- **Text chat models of known size,** at most 14B parameters (`--max-params-b`), updated
  within 18 months (`--max-age-days`), base-only models excluded, and at most two per
  publisher so one vendor cannot fill the list.
- **Pinned and verified download:** each file is fetched at an exact commit SHA (never
  `main`), its SHA-256 is checked against the hash the Hub records, a mismatch deletes
  the file, and a `.provenance.json` (repo, revision, hashes, license, source URLs,
  timestamp) is written next to it. Split (sharded) models are downloaded and verified
  shard by shard. Files go under `~/.cache/texturr/models` (set `TEXTURR_HOME` to move them).
- **Serving stays on this machine:** `--serve` runs `llama-server` bound to `127.0.0.1`
  only. It requires llama.cpp to be installed; texturr does not install it.

Then run texturr normally; `--llm auto` finds the server on port 8080.

The Hub is contacted only for the list and the download, and never when `--offline` is
set. In an air-gapped environment, download on a connected machine and carry the file
and its `.provenance.json` across.

## Privacy model

- **Local by default.** Providers on `localhost` are local; everything else is treated
  as remote, including LAN addresses.
- **Remote is opt-in.** A remote provider fails with an explanation unless
  `--allow-remote` is passed, and texturr logs the host and how many responses per
  cluster will be sent. Only a sample of each cluster goes out (`--sample-size`, each
  response truncated to 400 characters), never the whole sheet.
- **API keys come from environment variables only**, never the command line, so they
  stay out of shell history and process listings. They are never logged.
- **Air-gapped use:** pre-download the embedding model, point `--embedding-model` at its
  directory, and pass `--offline`. That blocks Hugging Face downloads and refuses
  remote providers.
- **Responses are fenced as untrusted data.** They are wrapped in `<answers>` tags in the
  prompt with an instruction never to follow anything inside them, and model replies are
  parsed strictly (a reply that is not the expected JSON is discarded).
- **Outputs contain verbatim responses.** Handle the CSV, HTML and annotated copy at the
  same classification as the input. Cells starting with `=`, `+`, `-`, `@` are neutralized
  so opening a result in Excel cannot run formulas, including theme text written into
  the annotated workbook, which is stored as text.

```bash
# Local only (default behavior)
python3 texturr.py survey.xlsx --column Comments --llm ollama --model llama3.1

# Air-gapped
python3 texturr.py survey.xlsx --column Comments --offline --embedding-model ./models/all-MiniLM-L6-v2

# Hosted, deliberately
export ANTHROPIC_API_KEY=...
python3 texturr.py survey.xlsx --column Comments --llm anthropic --allow-remote
```

## Evaluation

`eval/run_eval.py` scores the grouping step on a public labeled dataset, Banking77
(customer-service queries labeled with 77 intents, a reasonable stand-in for survey
comments). Each trial samples 8 intents x 30 queries, hides the labels, groups the
queries, and compares the grouping with the true intents. Adjusted Rand index (ARI) and
normalized mutual information (NMI) are 1.0 for a perfect match and about 0 for random.

10 trials, embedding model `all-MiniLM-L6-v2` (full output in `eval/RESULTS.md`):

| Method | ARI (mean ± sd) | NMI (mean ± sd) |
|---|---|---|
| TF-IDF + KMeans (true k) | 0.44 ± 0.09 | 0.64 ± 0.07 |
| texturr, fixed k=5 (old default) | 0.56 ± 0.06 | 0.78 ± 0.05 |
| texturr, auto k | 0.84 ± 0.14 | 0.90 ± 0.08 |
| texturr, true k (upper bound) | 0.87 ± 0.12 | 0.91 ± 0.08 |

Embeddings clearly beat a word-count baseline, and automatic group count lands close to
knowing the answer in advance. Limits to know about:

- Auto k picked the true count (8) in 5 of 10 trials and was off in the rest (4 to 11).
  On very small, noisy sets it tends to over-split: the 24-row bundled example has 4
  obvious themes and auto picks 7. I tried a "prefer fewer groups" rule; it fixed that
  example but lowered accuracy on the benchmark (ARI 0.83 to 0.74), so it is not used.
  Pass `--clusters N` when you know roughly how many themes to expect.
- This table evaluates grouping only. Labels are evaluated separately below.
- Banking77 queries are short and clean; real survey text is messier.

Reproduce: `pip install datasets && python eval/run_eval.py --trials 10 --intents 8`.

**Choosing the group count: what did not work.** I tried to fix over-splitting with statistics,
on held-out trials at four set sizes (4x6, 5x12, 8x30 and 12x60 responses): Davies-Bouldin, agglomerative
clustering with a distance threshold, and merging groups whose centers are similar after
silhouette. Plain silhouette won or tied at every size. Merging by center similarity changed
nothing at any threshold that did not also hurt accuracy, and the other two were worse (for example
agglomerative missed the true count by 16 to 25 groups on the larger sets). What remains is
semantic over-splitting that only reading the responses can fix, which is the case for the
[agentic mode](#agentic-mode-an-ai-agent-that-refines-the-grouping-n8n).

**Label quality.** `eval/label_eval.py` takes groups whose true intent is known and checks whether the
generated label is closer, by embedding, to its own intent than to any other intent in the trial
(top-1), using the true groups so label quality is not confounded with grouping quality. Today's
numbers are for the no-LLM baseline; the same command scores any model
(`--provider ollama --model llama3.1`).

| Labels (10 trials x 8 intents) | top-1 | similarity to own intent |
|---|---|---|
| keyphrases only (no LLM) | 0.84 ± 0.10 | 0.61 ± 0.06 |
| shuffled labels (floor) | 0.20 ± 0.17 | 0.37 ± 0.16 |
| the true intent text (ceiling) | 1.00 | 1.00 |

**An LLM has not yet been scored.** The harness is validated (it separates correct from shuffled
labels) but no real model has run through it, so the claim that LLM labels beat keyphrases is
unproven here. The metric also measures identification, not fluency or usefulness of the summary
and suggested action, which still need a human read.

## Tests

```bash
python3 -m pytest
```

The suite (83 tests) uses stub local HTTP servers, so it needs no model, no Hugging Face
account and no network. It covers the provider policy, model vetting and downloads, the session
logic and its guardrails, the REST and MCP server (auth, limits, uploads, error handling), the
generated n8n workflows (valid connections, no secrets, in sync with the server's tool list) and
the scripted test LLM. One end-to-end test runs the real embedding and keyphrase stack and skips
itself when `sentence-transformers` is not installed. GitHub Actions runs the suite on Python 3.9 to 3.12.

## Changelog

- **Agentic mode and limit fixes**: `texturr.py serve` (REST + MCP, in-memory sessions, token auth);
  an n8n AI Agent workflow (upload form and file-based) that merges, splits, moves and labels groups;
  Docker image and compose stack with execution-data retention disabled; `move_responses`;
  non-answers ("N/A", "none") set aside as their own group; fixed pandas silently dropping the text
  "N/A"/"None"/"NA" when reading files; label-quality harness with a measured baseline; documented
  negative results on group-count selection. **Change:** with default settings, non-answers now appear as cluster `-1`.

- **Features batch**: automatic group count (silhouette); several columns in one run;
  CSV/TSV input; `--html` shareable report; `--annotated` copy of the input with
  cluster and theme columns; sharded GGUF downloads; `models --serve` (loopback-only
  llama-server); `pyproject.toml` with a `texturr` command; CI; an evaluation on
  Banking77 with published results; a bundled example. **Changes:** `--clusters`
  now defaults to `auto` (was 5); the summary CSV gains a leading `Column` column;
  the embedding model is loaded once and shared with KeyBERT.
- **Vetted model shortlist and verified download**: `texturr.py models` lists the top 5
  vetted local models from the live Hub, and `--pull N` downloads one pinned to a
  commit SHA with SHA-256 verification and a provenance record.
- **Local-first LLM labeling**: `llm.py` supports local servers and hosted providers
  (Anthropic, OpenAI, Gemini, Mistral, Groq, OpenRouter) with keys read from the
  environment. Remote use requires `--allow-remote`; `--offline` forbids it. Replaced
  the per-response BART summarizer with one LLM label, summary and suggested action per
  cluster, plus representative responses.
- **Column picker**: the grouping column is chosen by letter or header name, via
  `--column` or an interactive list. Added `--sheet`, `--row`, `--header-row`,
  `--clusters`, `--output`. Row 1 is now a header by default (`--header-row 0` for the
  old behavior).

## Status and plans

Working and tested end to end with real components: the pipeline (embeddings, grouping, keyphrases,
HTML, annotated copy), the grouping evaluation, the service with real embeddings, and both n8n
workflows running in real n8n 2.41.6 with the real AI Agent and MCP nodes.

**Not yet verified, and the honest list of what is unproven:**
- **Any real LLM.** The labeling path is tested against stub servers, and the n8n agent against a
  scripted stand-in that follows a fixed plan. That proves the wiring, not that a model makes good
  merge decisions or labels. This is the main gap; the harnesses to measure it exist.
- **A real model download.** The downloader is tested against a stub server and the Hub's live metadata.
- Hosted-provider default model names come from memory and should be checked against provider docs.
- Auto k is still approximate (see Evaluation); the agent is the intended fix, but that is untested with a real model.

Not built: Ollama import of downloaded files, outlier detection beyond non-answers, installing
llama.cpp for you, and multi-column analysis in the service (the CLI supports it). This README is
updated with every change.
