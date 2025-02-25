import os 
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import nltk
from nltk.tokenize import sent_tokenize
import glob
from tqdm import tqdm

nltk.download('punkt')

def load_resouces():
    resources =[]

    #load all text files in the data folder
    for filepath in glob.glob('data/*.txt'):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text  = f.read()
                filename = os.path.basename(filepath)
                resource_name = os.path.splitext(filename)[0].replace('_', ' ').title()
                resources.append({
                    "id": len(resources),
                    "title": resource_name,
                    "content": text,
                    "source": filepath
                })
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
    return pd.DataFrame(resources)

class EmbeddingModel:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_embeddings(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512,return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            model_output = self.model(**inputs)
        embeddings = self.mean_pooling(model_output, inputs["attention_mask"])
        return embeddings.cpu().numpy()
    
def chunk_resources(df, chunk_size=3):
    chunks = []
    for _,row in df.iterrows():
        sentences = sent_tokenize(row['content'])
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i+chunk_size])
            if len(chunk.split())> 10:
                chunks.append({
                    "id": len(chunks),
                    "parent_id": row['id'],
                    "title": row['title'],
                    "content": chunk,
                    "source": row['source']
                })

    return pd.DataFrame(chunks)

def create_vector_db(df_chunks, embedding_model):
    embeddings=[]
    for i in range(0, len(df_chunks), 32):
        texts = df_chunks['content'].iloc[i:i+32].tolist()
        batch_embeddings = embedding_model.get_embeddings(texts)
        embeddings.extend(batch_embeddings)
    embeddings_matrix = np.vstack(embeddings)

    vector_db = {
        'df_chunks': df_chunks,
        'embeddings_matrix': embeddings_matrix
    }
    return vector_db