import requests
import json
import os
import torch
import glob
import logging
import numpy as np
import pickle
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from langchain.text_splitter import CharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

resources=[]

def create_chunks():
    for filepath in glob.glob('data/*.txt'):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                text_splitter = CharacterTextSplitter(
                    separator="\n\n",
                    chunk_size=1024,
                    chunk_overlap=200,
                    length_function=len,  # Use len function, not len(text)
                    is_separator_regex=False
                )
                chunks = text_splitter.create_documents([text])  # Pass text as a list
                resources.append(
                    {
                        "id": len(resources),
                        "filename": os.path.basename(filepath),
                        "chunks": chunks
                    }
                )
                print(f"Created {len(chunks)} chunks from {filepath}")
        except Exception as e:
            logging.error(f"Error loading file {filepath}: {e}")
            print(f"Error loading file {filepath}: {e}")

def embed_text():
    print("Starting text embedding process...")
    
    model_name = "intfloat/e5-large-v2"
    model_kwargs = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    encode_kwargs = {
        "normalize_embeddings": True,
        "batch_size": 8  
    }

    print(f"Loading embedding model: {model_name}")
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    total_resources = len(resources)
    for i, resource in enumerate(resources):
        print(f"Embedding resource {i+1}/{total_resources}: {resource['filename']}")
        
        # Process chunks in batches for efficiency
        chunks = [chunk.page_content for chunk in resource['chunks']]
        total_chunks = len(chunks)
        
        if total_chunks > 0:
            print(f"  Processing {total_chunks} chunks...")
            
            # Using batch processing directly from the model
            embeddings = []
            batch_size = 16  # Adjust based on your GPU/CPU capacity
            
            for j in range(0, total_chunks, batch_size):
                batch_end = min(j + batch_size, total_chunks)
                print(f"  Embedding batch {j+1}-{batch_end} of {total_chunks}")
                batch = chunks[j:batch_end]
                batch_embeddings = hf.embed_documents(batch)
                embeddings.extend(batch_embeddings)
                
            resource['embeddings'] = embeddings
            print(f"  Completed embedding {len(embeddings)} chunks")
        else:
            print(f"  No chunks to embed in {resource['filename']}")
    
    print("Text embedding completed!")
    return resources


def save_embeddings(resources, filepath="embeddings.pkl"):
    """Save the embeddings to disk"""
    print(f"Saving embeddings to {filepath}...")
    with open(filepath, "wb") as f:
        pickle.dump(resources, f)
    print("Embeddings saved successfully!")

def load_embeddings(filepath="embeddings.pkl"):
    """Load the embeddings from disk"""
    print(f"Loading embeddings from {filepath}...")
    with open(filepath, "rb") as f:
        return pickle.load(f)

def retrieve_relevant_chunks(query, resources, embedding_model, top_k=3):
    """Retrieve the most relevant chunks for a query"""
    print(f"Retrieving relevant chunks for query: {query}")
    
    # Embed the query
    query_embedding = embedding_model.embed_query(query)
    
    # Find the most similar chunks across all resources
    all_results = []
    
    for resource in resources:
        embeddings = resource['embeddings']
        chunks = resource['chunks']
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        # Pair chunks with their similarity scores
        chunk_similarities = list(zip(chunks, similarities))
        
        # Sort by similarity (descending)
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Add top results from this resource
        for chunk, score in chunk_similarities[:top_k]:
            all_results.append({
                'chunk': chunk,
                'score': score,
                'resource_name': resource['filename']
            })
    
    # Get overall top results
    all_results.sort(key=lambda x: x['score'], reverse=True)
    return all_results[:top_k]

def answer_question(query, resources, embedding_model, api_key=None, model="anthropic/claude-3-haiku"):
    """Answer a question using RAG with OpenRouter API"""
    print(f"Answering question: {query}")
    
    # Get API key from environment variable if not provided
    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable.")
    
    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(query, resources, embedding_model, top_k=5)
    
    # Prepare context from retrieved chunks
    context = ""
    for i, result in enumerate(relevant_chunks):
        context += f"Document {i+1} ({result['resource_name']}): {result['chunk'].page_content}\n\n"
    
    print(f"Found {len(relevant_chunks)} relevant chunks")
    
    # Format prompt for API request
    prompt = f"""You are a helpful AI assistant with expertise in education and academic programs. 
    
Based on these documents:

{context}

Answer this question in detail: {query}
"""
    
    print(f"Generating answer using {model}...")
    
    # Set up the API request
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant with expertise in education and academic programs."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024
    }
    
    try:
        # Make the API request
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse the response
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        
    except Exception as e:
        print(f"Error using OpenRouter API: {e}")
        answer = f"Error generating response: {str(e)}"
    
    return {
        "query": query,
        "answer": answer,
        "sources": [{"name": r["resource_name"], "score": r["score"]} for r in relevant_chunks],
        "model": model
    }

def main():
    # Create chunks and generate embeddings
    create_chunks()
    resources_with_embeddings = embed_text()
    
    # Save embeddings to avoid regenerating them
    save_embeddings(resources_with_embeddings)
    
    # Initialize the embedding model for retrieval
    model_name = "intfloat/e5-large-v2"
    model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # Interactive question answering
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        result = answer_question(query, resources_with_embeddings, embedding_model)
        print("\nAnswer:", result["answer"])
        print("\nSources:")
        for source in result["sources"]:
            print(f"- {source['name']} (relevance: {source['score']:.2f})")

if __name__ == "__main__":
    main()