# texturr

Summarize and group free-form answers within spreadsheets.

texturr reads one column (or row) of free-text answers from an Excel file,
groups similar answers together with sentence embeddings, and writes a CSV
with each group's keyphrases, a summary, and the response numbers it contains.

## Install

```bash
pip install pandas openpyxl tqdm scikit-learn sentence-transformers transformers keybert
```

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

In the interactive prompt, type a column letter, a header name, or `row N`.

## Output

One row per group: `Cluster`, `Keyphrases`, `Actionable Summary`, `Responses`
(1-based positions of the answers in that group).

## Changelog

- **Column picker**: the grouping column is now chosen by letter or header name,
  via `--column` or an interactive list that shows headers and sample answers.
  Added `--sheet`, `--row`, `--header-row`, `--clusters`, `--output` so runs can
  be scripted. Heavy ML libraries now load only when needed. **Behavior change:**
  row 1 is treated as a header by default; use `--header-row 0` for the old behavior.

## Status and plans

This is an early prototype. Planned: LLM-written theme labels with representative
quotes, automatic choice of cluster count, an evaluation on a public labeled
dataset, tests and CI, and a shareable HTML report. This README is updated with
every change.
