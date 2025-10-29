# api.py
import sys
import logging
import requests
import json
import os
import time
from fastapi import FastAPI
import uvicorn

# --- RAG Specific Imports ---
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# --- Reranking Imports ---
from sentence_transformers import CrossEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ollama API endpoint for chat
MODEL_NAME = "mistral:latest" # Use a model that you know is fast on your system
EMBEDDING_MODEL_NAME = "nomic-embed-text:latest"

# --- RAG Setup ---
KNOWLEDGE_BASE_DIR = "knowledge_base"
CHROMA_DB_DIR = "chroma_db"

# Global variables
vectorstore = None
reranker_model = None

# A simple in-memory cache for demonstration
direct_query_cache = {}

def initialize_vectorstore():
    """
    Loads documents, chunks them, and creates/loads a Chroma vectorstore.
    This function runs ONCE when the API service starts.
    """
    global vectorstore
    global reranker_model

    start_time = time.time()
    logging.info("Task: Initializing vectorstore...")
    

    if not check_ollama_status(EMBEDDING_MODEL_NAME):
        logging.error(f"Embedding model '{EMBEDDING_MODEL_NAME}' not found. Please pull it and ensure Ollama is running.")
        sys.exit(1)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url="http://localhost:11434")

    # Check for and load existing ChromaDB
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        logging.info("Sub-task: Loading existing ChromaDB...")
        try:
            vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
            logging.info("ChromaDB loaded successfully.")
        except Exception as e:
            logging.warning(f"Failed to load existing ChromaDB: {e}. Re-creating.")
            vectorstore = None

    if vectorstore is None:
        logging.info("Sub-task: Creating new ChromaDB vectorstore...")
        documents = []
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            logging.error(f"Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found.")
            sys.exit(1)
        for filename in os.listdir(KNOWLEDGE_BASE_DIR):
            if filename.endswith(".txt"):
                filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
                try:
                    loader = TextLoader(filepath, encoding='utf-8')
                    documents.extend(loader.load())
                except Exception as e:
                    logging.error(f"Error loading {filepath}: {e}")

        if not documents:
            logging.warning("No text files found. RAG will not have any context.")
            return

        text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        try:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_DB_DIR
            )
            logging.info(f"ChromaDB created and persisted to {CHROMA_DB_DIR}.")
        except Exception as e:
            logging.error(f"Error creating ChromaDB: {e}. Please ensure '{EMBEDDING_MODEL_NAME}' is running.")
            sys.exit(1)

    try:
        reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        logging.info("Reranker model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading reranker model: {e}. Reranking will be skipped.")
        reranker_model = None

    end_time = time.time()
    logging.info(f"Task: Vectorstore initialization completed in {end_time - start_time:.2f} seconds.")


def generate_response_with_rag(input_text: str):
    """
    Generates a response using the optimized RAG pipeline.
    """
    if vectorstore is None:
        return "I'm sorry, I cannot access my knowledge base right now."

    start_time = time.time()
    logging.info("Task: Generating response with RAG...")

    # --- Step 1: Semantic Cache Check ---
    if input_text in direct_query_cache:
        logging.info("Sub-task: Cache hit. Returning response from cache.")
        return direct_query_cache[input_text]
    
    try:
        logging.info(f"User query: {input_text}")
        
        # --- Step 2: Vector DB Retrieval ---
        retrieval_start = time.time()
        retrieved_docs = vectorstore.similarity_search(input_text, k=10)
        retrieval_end = time.time()
        logging.info(f"Sub-task: Retrieved 10 documents in {retrieval_end - retrieval_start:.2f} seconds.")

        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

        # --- Step 3: Reranking ---
        if reranker_model:
            rerank_start = time.time()
            pairs = [[input_text, doc.page_content] for doc in retrieved_docs]
            scores = reranker_model.predict(pairs)
            
            ranked_docs = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
            top_ranked_docs = [doc for score, doc in ranked_docs[:4]]
            context_text = "\n\n---\n\n".join([doc.page_content for doc in top_ranked_docs])
            rerank_end = time.time()
            logging.info(f"Sub-task: Reranked documents in {rerank_end - rerank_start:.2f} seconds.")
        
        # --- Step 4: LLM Generation ---
        generation_start = time.time()
        
        rag_template = """
You are a human representative for iamneo.ai (formerly Examly). Your name is Neo. You are knowledgeable, helpful, and confident in your responses. Your primary role is to provide accurate and comprehensive information about iamneo.ai's offerings, including products, services, courses, training programs, and affiliations.

Answer the user's question with confidence, **strictly based on the following context**.
If the information is not present in the provided context, or if the question is outside of iamneo.ai's scope, respond politely by stating that you don't have that information. Do not invent or speculate.

Context:
{context}

Question: {question}
"""
        prompt = ChatPromptTemplate.from_template(rag_template)
        # Updated temperature for more natural, varied language
        llm = Ollama(model=MODEL_NAME, temperature=0.7, base_url="http://localhost:11434")

        formatted_prompt = prompt.format(context=context_text, question=input_text)
        response = llm.invoke(formatted_prompt)
        generated_text = response.strip()
        generation_end = time.time()
        logging.info(f"Sub-task: LLM generated response in {generation_end - generation_start:.2f} seconds.")

        logging.info(f"Generated response: {generated_text}")
        direct_query_cache[input_text] = generated_text

        return generated_text

    except requests.exceptions.ConnectionError:
        logging.error("Could not connect to Ollama. Is it running?")
        return "I apologize, but I couldn't connect to the Ollama service."
    except requests.exceptions.Timeout:
        logging.error("Request to Ollama timed out.")
        return "I apologize, but the request to the AI model timed out."
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
        return "I apologize, but I encountered an internal error. Please try again."

def check_ollama_status(model_name):
    """
    Check if Ollama is running and if the specified model is available.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            return model_name in model_names
        return False
    except requests.exceptions.ConnectionError:
        return False
    except Exception as e:
        logging.error(f"Error checking Ollama status: {str(e)}")
        return False

# --- FastAPI Setup ---
app = FastAPI()

# Event handler for startup
@app.on_event("startup")
async def startup_event():
    logging.info("Starting FastAPI application...")
    # Initialize the vector store only once when the server starts
    if not check_ollama_status(MODEL_NAME):
        logging.error("Ollama server or model not available on startup.")
        sys.exit(1)
    initialize_vectorstore()
    logging.info("FastAPI application started.")

@app.post("/chat")
def chat_endpoint(request_data: dict):
    input_text = request_data.get("message")
    if not input_text:
        return {"response": "No message provided."}
    
    response = generate_response_with_rag(input_text)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, log_level="info", reload=False)