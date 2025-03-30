import json
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
import os
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import logging

def check_datasets_file():
    try:
        os.mkdir('datasets')
    except FileExistsError:
        pass
    except PermissionError:
        print("Permission denied: Unable to create datasets.")
    except Exception as e:
        print(f"An error has occurred: {e}")

def json_to_dataframe(dir):
    data = [] # List to append all JSON objects to

    with open(dir, 'r') as f:
        for line in f:
            try:
                json_data = json.loads(line)
                data.append(json_data)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line.strip()}. Error: {e}")

    if data: 
        df = pd.DataFrame(data) # Returning DataFrame with parsed json data
    else:
        df = pd.DataFrame() # Returning an empty DataFrame if no valid data
    return df

def load_and_transform_json(dir, **kwargs): # dir: directory of json data, optional kwargs: drop: returns a dataframe with the column removed (takes in column titles as a list), subject_match_rows: returns a dataframe with only the matching subjects (takes in subject names as a list)
    df = json_to_dataframe(dir) # Calling json_to_dataframe to load json data into a DataFrame

    for key, value in kwargs.items():
        try:
            match key:
                case "drop": # Used to drop columns taking in the column titles
                    for title in value:
                        df = df.drop(value, axis=1)

                case "subject_match": # Used to only pass through rows that contain a keyword
                    subject_df = pd.DataFrame()
                    for subject in value:
                        mask_df = df['subject_name'].str.contains(subject, case=False, na=False) 
                        subject_df = pd.concat(objs=(df[mask_df], subject_df))
                    return subject_df

        except Exception as e:
            print(f"Error transforming database with key: {key}, value: {value}. Error: {e}")
    
    return df


def df_to_parquet(df_data, file_name):
    check_datasets_file()
    parquet_table = pa.Table.from_pandas(df_data)
    pq.write_table(parquet_table, f'datasets/{file_name}.parquet')

def transform_dataset_to_parquet(dir, **kwargs):
    new_data = load_and_transform_json(dir, **kwargs)
    file_name = os.path.splitext(os.path.basename(dir))[0]
    df_to_parquet(new_data, file_name)


# Only used for debugging
def _df_to_csv(df_data, file_name): 
    check_datasets_file()
    csv_data = df_data.to_csv(f'datasets/{file_name}.csv', sep=',')


# Vector Embedding
class EmbeddingGenerator:
    def __init__(self, model_name="dmis-lab/biobert-v1.1"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model=  AutoModel.from_pretrained(model_name).to(self.device)
        logging.info(f"Loaded {model_name} on {self.device}")

    def generate_embeddings(self, texts, batch_size=32):
        # Generating vector embeddings for a list of texts in batches (default of 32)
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts [i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)

        return np.concatenate(embeddings)
            
    def add_embeddings_to_df(self, df, df_column):
        # Adding embeddings to a column in a dataframe
        try: 
            texts = df[df_column].tolist()
            embeddings = self.generate_embeddings(texts)
            df[f"{df_column}_embedding"] = list(embeddings)
            return df
        except Exception as e:
            print(f"An error as occurred: {e}")
            return pd.DataFrame()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Data Utilities')

    # Main file argument
    parser.add_argument('filename', help='Path to JSON data file or directory')
    
    # Subparser for different operations
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Read command
    read_parser = subparsers.add_parser('read', help='Convert to DataFrame without transformation')

    # Transform command with options
    transform_parser = subparsers.add_parser('transform', help='Transform the data')
    transform_parser.add_argument('--drop', nargs="+", help='Columns to drop (space separated)')
    transform_parser.add_argument('--subject-match', nargs='+', help='Subjects to filter by (space separated)')

    args = parser.parse_args()

    if args.command == 'read':
        df = json_to_dataframe(args.filename)
        print(df.head())
    
    elif args.command == 'transform':
        kwargs = {}
        if args.drop:
            kwargs['drop'] = args.drop
        if args.subject_match:
            kwargs['subject_match'] = args.subject_match
        
        df = load_and_transform_json(args.filename, **kwargs)
        print(df.head())