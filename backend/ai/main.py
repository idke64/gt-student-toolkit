import os
import torch
import glob
import logging
import numpy as np
import pickle
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def answer_question(query, resources, embedding_model, llm_model_name="microsoft/phi-3-mini-4k-instruct"):
    """Answer a question using RAG with a fast, high-quality model"""
    print(f"Answering question: {query}")
    
    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(query, resources, embedding_model, top_k=5)
    
    # Prepare context from retrieved chunks
    context = ""
    for i, result in enumerate(relevant_chunks):
        context += f"Document {i+1} ({result['resource_name']}): {result['chunk'].page_content}\n\n"
    
    print(f"Found {len(relevant_chunks)} relevant chunks")
    
    # Load model and tokenizer with GPU optimizations
    print(f"Loading LLM: {llm_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        torch_dtype=torch.float16,  # Use half precision for speed
        device_map="auto"           # Efficiently map to GPU
    )
    
    # Format prompt for Phi-3-mini
    prompt = f"""<|system|>
You are a helpful AI assistant with expertise in education and academic programs. Provide comprehensive, 
detailed answers based only on the context provided.

<|user|>
Based on these documents:

{context}

Answer this question in detail: {query}

<|assistant|>"""
    
    print("Generating answer...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generation parameters optimized for speed while maintaining quality
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=512,
        temperature=0.7,         # Good balance of creativity and factuality
        top_p=0.9,               # Nucleus sampling for natural text
        do_sample=True,          
        repetition_penalty=1.15,  # Reduce repetition
        num_beams=1,             # Greedy decoding for speed
        pad_token_id=tokenizer.eos_token_id
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response if needed
    if "<|assistant|>" in answer:
        answer = answer.split("<|assistant|>")[1].strip()
    
    return {
        "query": query,
        "answer": answer,
        "sources": [{"name": r["resource_name"], "score": r["score"]} for r in relevant_chunks]
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