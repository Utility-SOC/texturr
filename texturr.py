#!/usr/bin/env python3

import argparse
import pandas as pd
import logging
import sys
import string
from tqdm import tqdm
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from keybert import KeyBERT

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Perform textual analysis on a column or row in an Excel spreadsheet.')
    parser.add_argument('filename', type=str, help='Excel file name (xlsx format)')
    args = parser.parse_args()
    return args

def list_sheets(filename):
    """List all sheets in the Excel workbook and allow user to select one."""
    try:
        xl = pd.ExcelFile(filename)
        sheet_names = xl.sheet_names
        if not sheet_names:
            logging.error("No sheets found in the Excel file.")
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

def get_row_or_column(df):
    """Prompt the user to enter a row number or column letter."""
    while True:
        try:
            selection = input("Enter the row number or column letter of interest: ").strip()
            if selection.isdigit():
                row_num = int(selection)
                if 1 <= row_num <= len(df):
                    logging.info(f"Selected row: {row_num}")
                    return 'row', row_num
                else:
                    print(f"Please enter a row number between 1 and {len(df)}.")
            elif selection.isalpha():
                column_letter = selection.upper()
                col_index = column_letter_to_index(column_letter)
                if col_index >= df.shape[1]:
                    print(f"Column '{column_letter}' does not exist. Available columns: {get_available_columns(df)}")
                    continue
                logging.info(f"Selected column: '{column_letter}'")
                return 'column', column_letter
            else:
                print("Invalid input. Please enter a row number or column letter.")
        except EOFError:
            print("\nNo input detected. Exiting the program.")
            sys.exit(1)

def get_available_columns(df):
    num_cols = df.shape[1]
    letters = [index_to_column_letter(i) for i in range(num_cols)]
    return ', '.join(letters)

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

def load_data(filename, sheet_name, selection_type, selection_value):
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
        if selection_type == 'column':
            col_index = column_letter_to_index(selection_value)
            if col_index >= df.shape[1]:
                raise ValueError(f"Column '{selection_value}' does not exist in the spreadsheet.")
            data_series = df.iloc[:, col_index].dropna().astype(str)
        elif selection_type == 'row':
            row_index = selection_value - 1
            if not 0 <= row_index < len(df):
                raise ValueError(f"Row '{selection_value}' does not exist in the spreadsheet.")
            data_series = df.iloc[row_index, :].dropna().astype(str)
        else:
            raise ValueError('Selection type must be either "row" or "column".')
        logging.info(f"Loaded {len(data_series)} items from the spreadsheet.")
        return data_series.tolist()
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

def compute_embeddings(data):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(data, show_progress_bar=True)
        return embeddings
    except Exception as e:
        logging.error(f"Error computing embeddings: {e}")
        sys.exit(1)

def cluster_embeddings(embeddings, n_clusters):
    try:
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
        with tqdm(total=1, desc="Clustering embeddings") as pbar:
            cluster_assignment = clustering_model.fit_predict(embeddings)
            pbar.update(1)
        return cluster_assignment
    except Exception as e:
        logging.error(f"Error clustering embeddings: {e}")
        sys.exit(1)

def extract_keyphrases(data, clusters):
    try:
        kw_model = KeyBERT()
        cluster_keyphrases = {}

        for cluster_id in set(clusters):
            cluster_texts = [data[i] for i in range(len(data)) if clusters[i] == cluster_id]
            combined_text = ' '.join(cluster_texts)
            keyphrases = kw_model.extract_keywords(combined_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
            cluster_keyphrases[cluster_id] = [kw for kw, _ in keyphrases]

        return cluster_keyphrases
    except Exception as e:
        logging.error(f"Error extracting keyphrases: {e}")
        return {}

def generate_actionable_summary(cluster_texts):
    try:
        summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
        actionable_summaries = []

        for response in cluster_texts:
            input_length = len(response.split())
            if input_length < 15:
                actionable_summaries.append(response)
                continue

            dynamic_max_length = max(10, input_length - 5)
            dynamic_min_length = max(5, input_length - 10)
            summary = summarizer(response, max_length=dynamic_max_length, min_length=dynamic_min_length, do_sample=False)[0]['summary_text']
            actionable_summaries.append(summary)

        return list(set(actionable_summaries))
    except Exception as e:
        logging.error(f"Error generating actionable summaries: {e}")
        return ["Error generating summary"]

def save_results_to_csv(data, clusters, keyphrases, output_filename='summary_output.csv'):
    try:
        cluster_mapping = {}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_mapping:
                cluster_mapping[cluster_id] = []
            cluster_mapping[cluster_id].append(data[idx])

        summary_mapping = []
        for cluster_id, responses in cluster_mapping.items():
            summaries = generate_actionable_summary(responses)
            keyphrase_summary = ', '.join(keyphrases.get(cluster_id, []))
            summary_mapping.append({
                'Cluster': cluster_id,
                'Keyphrases': keyphrase_summary,
                'Actionable Summary': "; ".join(summaries),
                'Responses': ", ".join([str(i + 1) for i in range(len(data)) if clusters[i] == cluster_id])
            })

        df = pd.DataFrame(summary_mapping)
        df.to_csv(output_filename, index=False)
        logging.info(f"Results saved to {output_filename}")
    except Exception as e:
        logging.error(f"Error saving results to CSV: {e}")
        sys.exit(1)

def main():
    setup_logging()
    args = parse_arguments()
    filename = args.filename

    sheet_name = list_sheets(filename)
    df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
    selection_type, selection_value = get_row_or_column(df)
    data = load_data(filename, sheet_name, selection_type, selection_value)

    logging.info("Computing embeddings...")
    embeddings = compute_embeddings(data)

    n_clusters = min(len(data), 5)
    logging.info("Clustering embeddings...")
    clusters = cluster_embeddings(embeddings, n_clusters)

    logging.info("Extracting keyphrases for each cluster...")
    keyphrases = extract_keyphrases(data, clusters)

    save_results_to_csv(data, clusters, keyphrases)

if __name__ == '__main__':
    main()
