#!/usr/bin/env python3
"""texturr: group and summarize free-form answers in a spreadsheet, local-first."""

import argparse
import logging
import os
import string
import sys

import numpy as np
import pandas as pd

import clustering
import llm
import models
import report

CSV_SUFFIXES = ('.csv', '.tsv', '.tab')


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def parse_arguments(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Group and summarize free-form answers in a spreadsheet.')
    parser.add_argument('filename', type=str, help='Excel (.xlsx) or CSV/TSV file')
    parser.add_argument('--sheet', help='Sheet name or 1-based number (prompts if omitted and there are several)')
    parser.add_argument('--column', nargs='+',
                        help='Column(s) to group: letters (C) or header names ("Comments"). Several are analyzed '
                             'separately. Prompts if omitted.')
    parser.add_argument('--row', type=int, help='Group the answers in this 1-based row instead of a column')
    parser.add_argument('--header-row', type=int, default=1,
                        help='1-based row holding column headers (default 1, 0 = no header row)')
    parser.add_argument('--clusters', default='auto',
                        help="Number of groups: 'auto' (default; best silhouette score from 2 to 12) or a number")
    parser.add_argument('--output', default='summary_output.csv', help='Summary CSV (default summary_output.csv)')
    parser.add_argument('--html', metavar='PATH', help='Also write a self-contained HTML report')
    parser.add_argument('--annotated', metavar='PATH',
                        help='Also write a copy of the input with cluster and theme columns added beside the data')
    parser.add_argument('--llm', default='auto',
                        help='Labeling model provider: auto (default; first local server found), none, '
                             + ', '.join(llm.PRESETS) + '. Local servers keep data on this machine.')
    parser.add_argument('--model', help='Model name for the provider (local servers default to what is loaded)')
    parser.add_argument('--base-url', help='Override the provider URL (needed for openai-compatible)')
    parser.add_argument('--api-key-env', help='Environment variable holding the API key (default per provider)')
    parser.add_argument('--allow-remote', action='store_true',
                        help='Permit sending sampled cluster text to a remote provider. Off by default.')
    parser.add_argument('--offline', action='store_true',
                        help='Air-gapped mode: never touch the network except a local LLM server; '
                             'the embedding model must already be on disk')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2',
                        help='Sentence-transformers model name or local directory (default all-MiniLM-L6-v2)')
    parser.add_argument('--sample-size', type=int, default=8,
                        help='Responses per cluster shown to the LLM (default 8)')
    args = parser.parse_args(argv)
    try:
        args.clusters = clustering.parse_clusters(args.clusters)
    except ValueError as e:
        parser.error(str(e))
    return args


def is_table_file(filename):
    return filename.lower().endswith(CSV_SUFFIXES)


def read_frame(filename, sheet_name=None):
    """Read a sheet (or a CSV/TSV file) as raw cells, with no header interpretation."""
    if is_table_file(filename):
        sep = '\t' if filename.lower().endswith(('.tsv', '.tab')) else ','
        return pd.read_csv(filename, header=None, sep=sep, dtype=str, skip_blank_lines=False)
    return pd.read_excel(filename, sheet_name=sheet_name if sheet_name is not None else 0, header=None)


def list_sheets(filename, requested=None):
    """List all sheets in the Excel workbook and allow user to select one. CSV/TSV files have none."""
    if is_table_file(filename):
        return None
    try:
        xl = pd.ExcelFile(filename)
        sheet_names = xl.sheet_names
        if not sheet_names:
            logging.error("No sheets found in the Excel file.")
            sys.exit(1)
        if requested is not None:
            if requested in sheet_names:
                return requested
            if requested.isdigit() and 1 <= int(requested) <= len(sheet_names):
                return sheet_names[int(requested) - 1]
            logging.error(f"Sheet '{requested}' not found. Available: {', '.join(sheet_names)}")
            sys.exit(1)
        elif len(sheet_names) == 1:
            logging.info(f"Only one sheet found: '{sheet_names[0]}'. Loading it automatically.")
            return sheet_names[0]
        else:
            print("Available sheets:")
            for idx, sheet in enumerate(sheet_names):
                print(f"{idx + 1}. {sheet}")
            while True:
                try:
                    selection = int(input("Enter the number corresponding to the sheet you want to load: "))
                    if 1 <= selection <= len(sheet_names):
                        selected_sheet = sheet_names[selection - 1]
                        logging.info(f"Selected sheet: '{selected_sheet}'")
                        return selected_sheet
                    else:
                        print(f"Please enter a number between 1 and {len(sheet_names)}.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
    except Exception as e:
        logging.error(f"Error reading Excel file: {e}")
        sys.exit(1)


def column_labels(df, header_row):
    """Map column letter -> header text (or the letter itself when there is no header)."""
    labels = {}
    for i in range(df.shape[1]):
        letter = index_to_column_letter(i)
        text = ''
        if header_row and header_row <= len(df) and pd.notna(df.iat[header_row - 1, i]):
            text = str(df.iat[header_row - 1, i]).strip()
        labels[letter] = text or letter
    return labels


def resolve_column(value, labels):
    """Resolve a letter or header name to a column letter, or None if no match."""
    if value.upper() in labels and value.isalpha():
        return value.upper()
    matches = [k for k, v in labels.items() if v.lower() == value.lower()]
    return matches[0] if len(matches) == 1 else None


def show_columns(df, labels, header_row):
    print("Columns:")
    for letter, label in labels.items():
        col = df.iloc[header_row:, column_letter_to_index(letter)].dropna()
        sample = str(col.iloc[0])[:40] if len(col) else ''
        print(f"  {letter:>3}  {label[:30]:<30}  {len(col):>5} answers  e.g. {sample!r}")


def get_row_or_column(df, header_row=1, column=None, row=None):
    """Return ('column', [letters]) or ('row', number), from flags or an interactive prompt."""
    labels = column_labels(df, header_row)
    if row is not None:
        if not 1 <= row <= len(df):
            logging.error(f"Row {row} is out of range (1-{len(df)}).")
            sys.exit(1)
        return 'row', row
    if column:
        letters = []
        for value in column:
            letter = resolve_column(value, labels)
            if letter is None:
                logging.error(f"Column '{value}' not found. Available: {get_available_columns(df, labels)}")
                sys.exit(1)
            if letter not in letters:
                letters.append(letter)
        logging.info("Selected column(s): " + ', '.join(f"{c} ('{labels[c]}')" for c in letters))
        return 'column', letters
    show_columns(df, labels, header_row)
    while True:
        try:
            selection = input("Column(s) to group (letters or header names, comma-separated), or 'row N': ").strip()
        except EOFError:
            print("\nNo input detected. Exiting the program.")
            sys.exit(1)
        if selection.lower().startswith('row '):
            num = selection[4:].strip()
            if num.isdigit() and 1 <= int(num) <= len(df):
                return 'row', int(num)
            print(f"Please enter a row number between 1 and {len(df)}.")
            continue
        parts = [p.strip() for p in selection.split(',') if p.strip()]
        letters = [resolve_column(p, labels) for p in parts]
        if parts and all(letters):
            return 'column', list(dict.fromkeys(letters))
        print("No such column. Try letters, header names, or 'row N'.")


def get_available_columns(df, labels=None):
    if labels:
        return ', '.join(f"{k} ({v})" if v != k else k for k, v in labels.items())
    return ', '.join(index_to_column_letter(i) for i in range(df.shape[1]))


def column_letter_to_index(letter):
    letter = letter.upper()
    result = 0
    for char in letter:
        if char in string.ascii_uppercase:
            result *= 26
            result += ord(char) - ord('A') + 1
        else:
            raise ValueError(f"Invalid column letter: {letter}")
    return result - 1


def index_to_column_letter(index):
    index += 1
    result = ''
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def extract_answers(df, selection_type, selection_value, header_row=1):
    """Return (texts, positions): the non-empty answers and their 0-based row (column mode) or column index."""
    if selection_type == 'column':
        col_index = column_letter_to_index(selection_value)
        if col_index >= df.shape[1]:
            raise ValueError(f"Column '{selection_value}' does not exist in the spreadsheet.")
        series = df.iloc[header_row:, col_index].dropna().astype(str)
        series = series[series.str.strip() != '']
    elif selection_type == 'row':
        row_index = selection_value - 1
        if not 0 <= row_index < len(df):
            raise ValueError(f"Row '{selection_value}' does not exist in the spreadsheet.")
        series = df.iloc[row_index, :].dropna().astype(str)
        series = series[series.str.strip() != '']
    else:
        raise ValueError('Selection type must be either "row" or "column".')
    return series.tolist(), [int(i) for i in series.index]


_models = {}


def _sentence_model(model_name):
    """Load the sentence-transformers model once and share it with KeyBERT."""
    if model_name not in _models:
        from sentence_transformers import SentenceTransformer
        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


def compute_embeddings(data, model_name='all-MiniLM-L6-v2'):
    try:
        return _sentence_model(model_name).encode(data, show_progress_bar=len(data) > 200)
    except Exception as e:
        logging.error(f"Error computing embeddings: {e}")
        sys.exit(1)


def extract_keyphrases(data, clusters, model_name='all-MiniLM-L6-v2'):
    try:
        from keybert import KeyBERT
        kw_model = KeyBERT(model=_sentence_model(model_name))
        cluster_keyphrases = {}
        for cluster_id in set(clusters):
            cluster_texts = [data[i] for i in range(len(data)) if clusters[i] == cluster_id]
            keyphrases = kw_model.extract_keywords(' '.join(cluster_texts), keyphrase_ngram_range=(1, 2),
                                                   stop_words='english', top_n=5)
            cluster_keyphrases[cluster_id] = [kw for kw, _ in keyphrases]
        return cluster_keyphrases
    except Exception as e:
        logging.error(f"Error extracting keyphrases: {e}")
        return {}


def representative_indices(embeddings, clusters, cluster_id, n):
    """Indices of the n responses closest to the cluster centroid (the most typical ones)."""
    idx = np.where(np.asarray(clusters) == cluster_id)[0]
    vecs = np.asarray(embeddings)[idx]
    dist = np.linalg.norm(vecs - vecs.mean(axis=0), axis=1)
    return [int(idx[i]) for i in np.argsort(dist)[:n]]


def csv_safe(value):
    """Neutralize spreadsheet formula injection: cells starting with = + - @ are executed by Excel."""
    text = str(value)
    return "'" + text if text[:1] in ('=', '+', '-', '@', '\t', '\r') else text


def build_report(data, clusters, keyphrases, embeddings, config=None, sample_size=8):
    """One row per cluster. With a provider config, rows also get an LLM label/summary/action."""
    rows = []
    for cluster_id in sorted(set(int(c) for c in clusters)):
        members = [i for i in range(len(data)) if clusters[i] == cluster_id]
        reps = representative_indices(embeddings, clusters, cluster_id, sample_size)
        phrases = keyphrases.get(cluster_id, [])
        label = {'label': '', 'summary': '', 'action': ''}
        if config is not None:
            try:
                label = llm.label_cluster(config, [data[i] for i in reps], phrases) or label
            except llm.LLMError as e:
                logging.warning(f"Cluster {cluster_id}: labeling failed ({e}); keeping keyphrases only.")
        rows.append({
            'Cluster': cluster_id,
            'Size': len(members),
            'Label': label['label'],
            'Summary': label['summary'],
            'Suggested Action': label['action'],
            'Keyphrases': ', '.join(phrases),
            'Representative Responses': ' | '.join(data[i] for i in reps[:3]),
            'Responses': ', '.join(str(i + 1) for i in members),
        })
    return rows


def save_results_to_csv(rows, output_filename='summary_output.csv'):
    try:
        df = pd.DataFrame(rows)
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].map(csv_safe)
        df.to_csv(output_filename, index=False)
        logging.info(f"Results saved to {output_filename}")
    except Exception as e:
        logging.error(f"Error saving results to CSV: {e}")
        sys.exit(1)


def choose_provider(args):
    """Resolve the labeling provider and enforce the local-first policy. Returns a config or None."""
    if args.llm == 'none':
        return None
    try:
        config = llm.resolve_config(args.llm, args.model, args.base_url, args.api_key_env)
        if config is None:
            if args.llm == 'auto' and not args.offline and sys.stdin.isatty():
                if models.first_run_offer():
                    sys.exit(0)   # model downloaded; the user must start the server, then re-run
            logging.warning("No local LLM server found (tried ollama, llamacpp, lmstudio). "
                            "Continuing with keyphrases only. List vetted local models with: "
                            "python3 texturr.py models")
            return None
        llm.check_policy(config, args.allow_remote, args.offline)
    except llm.LLMError as e:
        logging.error(e)
        sys.exit(2)
    where = "local (data stays on this machine)" if config.is_local else f"REMOTE ({config.host})"
    logging.info(f"Labeling with {config.preset.name} / {config.model}: {where}")
    if not config.is_local:
        logging.warning(f"Up to {args.sample_size} responses per cluster will be sent to {config.host}.")
    return config


def analyze(data, args, config):
    """Embed, cluster and describe one list of answers. Returns (rows, clusters)."""
    logging.info("Computing embeddings...")
    embeddings = compute_embeddings(data, args.embedding_model)
    clusters, k, scores = clustering.cluster(embeddings, args.clusters)
    if args.clusters == 'auto':
        logging.info(f"Chose {k} groups automatically" + (f" (silhouette {scores[k]:.2f})" if scores else ""))
    keyphrases = extract_keyphrases(data, clusters, args.embedding_model)
    return build_report(data, clusters, keyphrases, embeddings, config, args.sample_size), clusters


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'models':
        sys.exit(models.cli(sys.argv[2:]))
    setup_logging()
    args = parse_arguments()
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    config = choose_provider(args)

    sheet_name = list_sheets(args.filename, args.sheet)
    df = read_frame(args.filename, sheet_name)
    selection_type, selection_value = get_row_or_column(df, args.header_row, args.column, args.row)
    targets = selection_value if selection_type == 'column' else [None]
    labels = column_labels(df, args.header_row)

    all_rows, results, annotations = [], [], []
    for target in targets:
        try:
            data, positions = extract_answers(df, selection_type, target if target else selection_value, args.header_row)
        except ValueError as e:
            logging.error(f"Error loading data: {e}")
            sys.exit(1)
        name = labels[target] if target else f"Row {selection_value}"
        logging.info(f"{name}: {len(data)} answers")
        if not data:
            logging.warning(f"{name}: no answers found; skipping.")
            continue
        rows, clusters = analyze(data, args, config)
        rows = [{'Column': name, **r} for r in rows]
        all_rows += rows
        results.append((name, rows, len(data)))
        annotations.append({'header': name, 'positions': positions, 'clusters': clusters,
                            'labels': {r['Cluster']: r['Label'] or ', '.join(r['Keyphrases'].split(', ')[:3]) for r in rows}})

    if not all_rows:
        logging.error("Nothing to analyze.")
        sys.exit(1)
    save_results_to_csv(all_rows, args.output)
    if args.html:
        report.write_html(args.html, args.filename, results, 'labeled by ' + config.model if config else 'keyphrases only')
        logging.info(f"HTML report saved to {args.html}")
    if args.annotated:
        if selection_type != 'column':
            logging.warning("--annotated applies to column analysis only; skipped.")
        elif is_table_file(args.filename):
            report.annotate_csv(args.filename, annotations, args.annotated, csv_safe, args.header_row)
            logging.info(f"Annotated copy saved to {args.annotated}")
        else:
            report.annotate_workbook(args.filename, sheet_name, annotations, args.annotated, args.header_row)
            logging.info(f"Annotated copy saved to {args.annotated}")


if __name__ == '__main__':
    main()
