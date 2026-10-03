# texturr

Summarize and group free-form answers within spreadsheets.

texturr reads one column (or row) of free-text answers from an Excel file,
groups similar answers together with sentence embeddings, has an LLM name and
summarize each group, and writes a CSV.

**Built for sensitive data: local-first.** By default everything runs on your
machine: embeddings use a local model and labeling uses a local LLM server
(Ollama, llama.cpp, LM Studio). Sending text to a hosted provider is possible but
off unless you pass `--allow-remote`, and `--offline` forbids it outright.

## Install

```bash
pip install -r requirements.txt
```

For LLM labels, run any local server, for example `ollama serve` with a model
pulled (`ollama pull llama3.1`). With no server found, texturr still works and
fills in keyphrases only. No provider SDKs are needed.

## Usage

```bash
# Interactive: lists the columns (with header names and a sample answer) and asks which to group
python3 texturr.py survey.xlsx

# Scripted: pick the sheet and the column by letter or by header name
python3 texturr.py survey.xlsx --sheet "Q3 Results" --column Comments --clusters 8 --output themes.csv
```

| Option | Meaning |
| --- | --- |
| `--column` | Column to group: a letter (`C`) or a header name (`Comments`, case-insensitive). Prompts if omitted. |
| `--row N` | Group the answers in row N instead of a column. |
| `--sheet` | Sheet name or 1-based number. Prompts if omitted and the workbook has several. |
| `--header-row N` | Row holding column headers (default `1`; `0` means no header row). Header text is used for naming columns and is excluded from the answers. |
| `--clusters K` | Number of groups (default `5`). |
| `--output` | Output CSV path (default `summary_output.csv`). |
| `--llm` | Labeling provider: `auto` (default, first local server found), `none`, or one of `ollama`, `llamacpp`, `lmstudio`, `anthropic`, `openai`, `gemini`, `mistral`, `groq`, `openrouter`, `openai-compatible`. |
| `--model` | Model name. Local servers default to the model they have loaded; hosted providers have a default you can override. |
| `--base-url` | Override the provider URL (required for `openai-compatible`, e.g. vLLM). |
| `--api-key-env` | Name of the environment variable holding the key (default per provider). |
| `--allow-remote` | Permit sending sampled cluster text to a remote provider. |
| `--offline` | Air-gapped mode: no network except a loopback LLM server; remote providers are refused. |
| `--embedding-model` | Sentence-transformers model name or local directory (default `all-MiniLM-L6-v2`). |
| `--sample-size` | Responses per cluster shown to the LLM (default 8). |

## Picking a local model (`models`)

No local model yet? texturr builds a live shortlist from the Hugging Face Hub:

```bash
python3 texturr.py models            # show the top 5 vetted models
python3 texturr.py models --pull 1   # download #1 (asks first)
```

The first interactive run with no local server found offers the same list. Nothing
is downloaded without an explicit yes.

**Why not a leaderboard?** No maintained Hugging Face leaderboard covers
summarization (the Open LLM Leaderboard was frozen in March 2025), so the list
comes from live Hub data (downloads, licenses, sizes, commit SHAs), ranked by
30-day downloads. That measures adoption, not quality. The vetting rules are what
make a candidate defensible, and they are all applied in code you can read
(`models.py`):

- **Publisher allowlist**, first-party GGUF repos only (no third-party re-quants):
  `ibm-granite`, `microsoft`, `mistralai`, `allenai`, `HuggingFaceTB`, `openai`, `nvidia`.
  Override with `--orgs`. Chinese-origin publishers (Qwen, DeepSeek) are left off
  by default because many government procurement policies restrict them; add them
  if your policy allows.
- **Provenance:** a model must also be derived only from base models by allowlisted
  publishers, so a fine-tune of someone else's model cannot ride in on an allowed name.
- **Permissive license only:** Apache-2.0 or MIT (`--licenses`). Custom community
  licenses (Llama, Gemma) are excluded.
- **Text chat models of known size,** at most 14B parameters (`--max-params-b`),
  updated within 18 months (`--max-age-days`), base-only models excluded, and at
  most two per publisher so one vendor cannot fill the list.
- **Pinned and verified download:** the file is fetched at an exact commit SHA
  (never `main`), its SHA-256 is checked against the hash the Hub records, a
  mismatch deletes the file, and a `.provenance.json` (repo, revision, hash,
  license, source URL, timestamp) is written next to it. Files go under
  `~/.cache/texturr/models` (set `TEXTURR_HOME` to move them). Sharded GGUFs are
  refused for now.

After a download, run the model with llama.cpp and texturr finds it automatically:

```bash
llama-server -m <path printed by texturr> --jinja --port 8080
python3 texturr.py survey.xlsx --column Comments
```

The Hub is contacted only for the list and the download, and never when `--offline`
is set. In an air-gapped environment, download on a connected machine and carry the
file and its `.provenance.json` across.

## Privacy model

- **Local by default.** Providers on `localhost` are local; everything else is
  treated as remote, including LAN addresses.
- **Remote is opt-in.** A remote provider fails with an explanation unless
  `--allow-remote` is passed, and texturr logs the host and how many responses
  per cluster will be sent. Only a sample of each cluster goes out
  (`--sample-size`, each response truncated to 400 characters), never the whole sheet.
- **API keys come from environment variables only**, never the command line, so
  they stay out of shell history and process listings. They are never logged.
- **Air-gapped use:** pre-download the embedding model, point `--embedding-model`
  at its directory, and pass `--offline`. That blocks Hugging Face downloads and
  refuses remote providers.
- **Responses are fenced as untrusted data.** They are wrapped in `<answers>` tags
  in the prompt with an instruction never to follow anything inside them, and model
  replies are parsed strictly (a reply that is not the expected JSON is discarded).
- **The output CSV contains verbatim responses.** Handle it at the same
  classification as the input. Cells starting with `=`, `+`, `-`, `@` are
  prefixed with `'` so opening the CSV in Excel cannot run formulas.

```bash
# Local only (default behavior)
python3 texturr.py survey.xlsx --column Comments --llm ollama --model llama3.1

# Air-gapped
python3 texturr.py survey.xlsx --column Comments --offline --embedding-model ./models/all-MiniLM-L6-v2

# Hosted, deliberately
export ANTHROPIC_API_KEY=...
python3 texturr.py survey.xlsx --column Comments --llm anthropic --allow-remote
```

In the interactive prompt, type a column letter, a header name, or `row N`.

## Output

One row per group: `Cluster`, `Size`, `Label` (LLM theme name), `Summary`,
`Suggested Action`, `Keyphrases`, `Representative Responses` (the three closest to
the group's center), and `Responses` (1-based positions of the answers in the group).
`Label`, `Summary` and `Suggested Action` are empty when no LLM is used.

## Tests

```bash
python3 -m pytest
```

The tests use stub local HTTP servers, so they need no model, no Hugging Face
account and no network.

## Changelog

- **Vetted model shortlist and verified download**: `python3 texturr.py models`
  lists the top 5 vetted local models from the live Hub, and `--pull N` (also
  offered on first run) downloads one pinned to a commit SHA with SHA-256
  verification and a provenance record. Vetting rules: publisher allowlist, base-model
  provenance, permissive license, size and recency limits. Replaces the idea of a
  leaderboard-driven list, since the HF leaderboard is frozen and not about summarization.

- **Local-first LLM labeling**: new `llm.py` supports local servers (Ollama,
  llama.cpp, LM Studio, any OpenAI-compatible endpoint) and hosted providers
  (Anthropic, OpenAI, Gemini, Mistral, Groq, OpenRouter) with keys read from the
  environment. Remote use requires `--allow-remote`; `--offline` forbids it.
  Replaced the per-response BART summarizer (which barely shortened one-line
  answers) with one LLM label, summary and suggested action per cluster, plus
  representative responses. Output columns changed (`Actionable Summary` is
  gone). KeyBERT now uses `--embedding-model` instead of a second hidden download.
  Output cells are neutralized against spreadsheet formula injection. Added tests.

- **Column picker**: the grouping column is now chosen by letter or header name,
  via `--column` or an interactive list that shows headers and sample answers.
  Added `--sheet`, `--row`, `--header-row`, `--clusters`, `--output` so runs can
  be scripted. Heavy ML libraries now load only when needed. **Behavior change:**
  row 1 is treated as a header by default; use `--header-row 0` for the old behavior.

## Status and plans

This is an early prototype. Planned: automatic choice of cluster count, an
evaluation on a public labeled dataset, CI, and a shareable HTML report. The
LLM path has been tested against stub servers but not yet against a real model, and
the downloader has been checked against the live Hub's metadata but has not yet
downloaded a real model.
This README is updated with every change.
