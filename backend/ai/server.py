import os
import asyncio
import json
import uuid
import time
import threading
import websockets
from concurrent.futures import ThreadPoolExecutor
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM

import llm

# logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

resources = None
embedding_model = None
tokenizer = None
model = None
model_lock = threading.Lock()  # For thread-safe model access

active_rooms = {}  # room_id -> set of websockets
room_locks = {}    # room_id -> lock for thread safety

def load_resources():
    """Load embeddings and initialize models"""
    global resources, embedding_model
    
    try:
        logger.info("Loading pre-computed embeddings")
        resources = llm.load_embeddings()
        logger.info(f"Loaded embeddings for {len(resources)} documents")
    except FileNotFoundError:
        logger.info("No pre-computed embeddings found, creating new ones")
        llm.create_chunks()
        resources = llm.embed_text()
        llm.save_embeddings(resources)
    
    logger.info("Initializing embedding model")
    model_name = "intfloat/e5-large-v2"
    model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    logger.info("Resources and embedding model initialized")

def get_llm():
    """Thread-safe access to the LLM model"""
    global tokenizer, model
    
    with model_lock:
        if tokenizer is None or model is None:
            logger.info("Loading LLM model")
            model_name = "microsoft/phi-3-mini-4k-instruct"
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                logger.info("LLM model loaded")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16
                )
                if torch.cuda.is_available():
                    model = model.to("cuda")
                logger.info("LLM model loaded (fallback method)")
        return tokenizer, model

def process_query(query, room_id):
    """Process a query in a separate thread"""
    logger.info(f"Processing query in room {room_id}: {query}")
    
    try:
        result = llm.answer_question(query, resources, embedding_model)
        
        for source in result["sources"]:
            source["score"] = float(source["score"])
            
        return result
    except Exception as e:
        logger.error(f"Error processing query in room {room_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "query": query,
            "answer": f"Error: {str(e)}",
            "sources": []
        }

async def handle_room(websocket, room_id, client_id):
    """Handle a client connection in a specific room"""
    try:
        if room_id not in active_rooms:
            active_rooms[room_id] = set()
            room_locks[room_id] = threading.RLock()
        
        with room_locks[room_id]:
            active_rooms[room_id].add(websocket)
        
        await websocket.send(json.dumps({
            "type": "system",
            "message": f"Welcome to room {room_id}! You are connected as {client_id}.",
            "room_id": room_id,
            "client_id": client_id,
            "timestamp": time.time()
        }))
        
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if "query" in data:
                    query = data["query"]
                    logger.info(f"Received query in room {room_id} from {client_id}: {query}")
                    
                    await websocket.send(json.dumps({
                        "type": "processing",
                        "message": "Processing your query...",
                        "query": query,
                        "room_id": room_id,
                        "client_id": client_id,
                        "timestamp": time.time()
                    }))
                    
                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor() as executor:
                        result = await loop.run_in_executor(
                            executor, 
                            process_query, 
                            query, 
                            room_id
                        )
                    
                    await broadcast_to_room(room_id, json.dumps({
                        "type": "answer",
                        "data": result,
                        "room_id": room_id,
                        "client_id": client_id,
                        "timestamp": time.time()
                    }))
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON from client {client_id} in room {room_id}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid message format. Expected JSON.",
                    "timestamp": time.time()
                }))
            except Exception as e:
                logger.error(f"Error handling message in room {room_id}: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}",
                    "timestamp": time.time()
                }))
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Connection closed for client {client_id} in room {room_id}: {e}")
    finally:
        if room_id in room_locks:
            with room_locks[room_id]:
                if room_id in active_rooms:
                    active_rooms[room_id].discard(websocket)
                    if not active_rooms[room_id]:
                        del active_rooms[room_id]
                        del room_locks[room_id]

async def broadcast_to_room(room_id, message):
    """Broadcast a message to all clients in a room except the excluded one"""
    if room_id not in active_rooms:
        return
    
    with room_locks[room_id]:
        room_clients = active_rooms[room_id].copy()
    
    for client in room_clients:
        try:
            await client.send(message)
        except websockets.exceptions.ConnectionClosed:
            pass

async def handler(websocket):
    """Initial handler for new connections"""
    try:
        message = await websocket.recv()
        data = json.loads(message)
        

        room_id = data.get("room_id", str(uuid.uuid4())[:8])
        
        client_id = data.get("client_id", str(uuid.uuid4())[:8])
        
        await handle_room(websocket, room_id, client_id)
    except json.JSONDecodeError:
        logger.error("Invalid initial message")
        await websocket.send(json.dumps({
            "type": "error",
            "message": "Invalid initial message. Expected JSON with room_id.",
            "timestamp": time.time()
        }))
    except Exception as e:
        logger.error(f"Error in handler: {e}")
        try:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Server error: {str(e)}",
                "timestamp": time.time()
            }))
        except:
            pass

async def main():
    load_resources()
    
    logger.info("Starting WebSocket server")
    async with websockets.serve(handler, "0.0.0.0", 8765):
        logger.info("WebSocket server started on ws://0.0.0.0:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")