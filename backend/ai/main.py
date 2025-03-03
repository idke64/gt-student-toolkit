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
import logging

nltk.download('punkt_tab')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_resources():
    resources =[]
    logging.info("Loading resources from data folder...")

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
            logging.error(f"Error loading file {filepath}: {e}")
    return pd.DataFrame(resources)

class EmbeddingModel:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        logging.info(f"Initializing embedding model: {model_name}")
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
        logging.info(f"Generating embeddings for {len(texts)} texts...")
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512,return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            model_output = self.model(**inputs)
        embeddings = self.mean_pooling(model_output, inputs["attention_mask"])
        return embeddings.cpu().numpy()
    
def chunk_resources(df, chunk_size=3):
    logging.info(f"Chunking resources with chunk size {chunk_size}...")
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
    logging.info("Creating vector database...")
    embeddings=[]
    for i in tqdm(range(0, len(df_chunks), 32)):
        texts = df_chunks['content'].iloc[i:i+32].tolist()
        batch_embeddings = embedding_model.get_embeddings(texts)
        embeddings.extend(batch_embeddings)
    embeddings_matrix = np.vstack(embeddings)

    vector_db = {
        'df_chunks': df_chunks,
        'embeddings_matrix': embeddings_matrix
    }
    return vector_db


class ResourceRetriever:
    def __init__(self, vector_db, embedding_model, top_k=5):
        logging.info("Initializing ResourceRetriever...")
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.top_k = top_k

    def retrieve(self, query):
        query_embedding = self.embedding_model.get_embeddings([query])
        similarity_scores = cosine_similarity(query_embedding, self.vector_db['embeddings_matrix'])[0]
        top_indices = np.argsort(similarity_scores)[-self.top_k:][::1]
        
        results= []

        for idx in top_indices:
            chunk  = self.vector_db['df_chunks'].iloc[idx]
            results.append({
                "title": chunk['title'],
                "content": chunk['content'],
                "similarity": similarity_scores[idx],
                "source": chunk['source']
            })

        return results
    

class QADataset(Dataset):
    def __init__(self, queries, responses, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.input = []

        for query, response in zip(queries, responses):

            text = f"Question: {query}  Context: {response}"
            encodings = tokenizer(text, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
            
            self.input.append({
                'input_ids': encodings['input_ids'][0],
                'attention_mask': encodings['attention_mask'][0],
                'labels': encodings['input_ids'][0].clone()
            })
    
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        return self.input[idx]    
    
def prepare_training_data(resources_df, num_examples=100):

    queries = []
    responses = []
    

    for i in range(min(num_examples, len(resources_df))):
        resource = resources_df.iloc[i]
        title = resource['title']
        content_preview = ' '.join(resource['content'].split()[:30]) + "..."
        
      
        query_templates = [
            f"Can you tell me about {title}?",
            f"What information do we have on {title}?",
            f"I need resources about {title}",
            f"Where can I find information about {title}?"
        ]
        
        response_template = f"Based on our school resources, here's information about {title}: {content_preview} You can find more details in the document titled '{title}'."
        
        for template in query_templates:
            queries.append(template)
            responses.append(response_template)
    
    return queries, responses


def fine_tune_model(queries, responses, model_name="gpt2", output_dir="./fine_tuned_model"):

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    dataset = QADataset(queries, responses, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-5,
    )
    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

class ResourceRAG:
    def __init__(self, retriever, model_path, max_new_tokens=150):
        self.retriever = retriever
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.max_new_tokens = max_new_tokens
    
    def answer(self, query):
        
        retrieved_results = self.retriever.retrieve(query)
        
        
        context = "Based on the following resources:\n\n"
        for i, result in enumerate(retrieved_results, 1):
            context += f"{i}. {result['title']}: {result['content']}\n\n"
        
        
        prompt = f"{context}\nQuery: {query}\nResponse:"
        
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        output = self.model.generate(
            input_ids,
            temperature=0.9,
            top_p=0.9,
        )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        
        response = response.split("Response:")[1].strip()
        
        return {
            'response': response,
            'sources': [{'title': r['title'], 'source': r['source']} for r in retrieved_results]
        }
    
def save_vector_db(vector_db, path="./vector_db.pkl"):
    with open(path, 'wb') as f:
        pickle.dump(vector_db, f)

def load_vector_db(path="./vector_db.pkl"):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def main():
    # Configuration
    resources_dir = "./school_resources"  # Directory containing your text files
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
    
    # Step 1: Load resources
    logging.info("Loading resources...")
    resources_df = load_resources()
    logging.info(f"Loaded {len(resources_df)} resources")
    
    # Step 2: Initialize embedding model
    logging.info("Initializing embedding model...")
    embedding_model = EmbeddingModel(model_name)
    
    # Step 3: Chunk resources
    logging.info("Chunking resources...")
    chunked_resources = chunk_resources(resources_df)
    logging.info(f"Created {len(chunked_resources)} chunks")
    
    # Step 4: Create vector database
    logging.info("Creating vector database...")
    vector_db = create_vector_db(chunked_resources, embedding_model)
    
    # Save the vector database
    save_vector_db(vector_db, "./vector_db.pkl")
    
    # Step 5: Create retriever
    retriever = ResourceRetriever(vector_db, embedding_model, top_k=3)
    
    # Step 6: Prepare training data and fine-tune model
    logging.info("Preparing training data...")
    queries, responses = prepare_training_data(resources_df, num_examples=50)
    
    logging.info("Fine-tuning model...")
    fine_tune_model(queries, responses, output_dir="./school_resources_model")
    
    # Step 7: Initialize RAG system
    logging.info("Initializing RAG system...")
    rag = ResourceRAG(retriever, "./school_resources_model")
    
    # Example query
    query = "Can you tell me about Threads?"
    logging.info(f"\nQuery: {query}")
    
    result = rag.answer(query)
    logging.info("\nResponse:" +  str(result['response']))
    logging.info("\nSources:")
    for source in result['sources']:
        logging.info(f"- {source['title']}")

if __name__ == "__main__":
    main()