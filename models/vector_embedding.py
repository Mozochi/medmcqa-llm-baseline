from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import logging
import numpy as np
import pandas as pd

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
        
    def embed(self, text):
        if isinstance(text, str):
            text = [text]
        try:
            embeddings = self.generate_embeddings(text)
            return embeddings[0]
        except Exception as e:
            print(f"An error as occurred: {e}")
            return 0
        