# Wyrd Wiki RAG System

**Homework Done:** [ Chunking Strategy](https://www.notion.so/Wyrd-3193ff2345d3803ea380c01413db3231?source=copy_link)

## click abouve link to see which chunking strategy i chose and why

## About This Project
A highly optimized local Retrieval-Augmented Generation (RAG) system custom-built for Wyrd Media Labs. This RAG natively understands the Wyrd brand, enforcing strict protections against "Dead Language" and preventing persona-driven hallucinations. 

## Dependencies Used
- **FastAPI & Uvicorn**: For high-performance async API routing.
- **LangChain**: For vector operations and document parsing.
- **ChromaDB**: The local vector database for storing document embeddings.
- **Ollama**: To run the local models (both embeddings and LLM) quickly.
- **Sentence-Transformers**: For the cross-encoder reranking mechanism.
- **Requests**: To query Ollama's API.

## Models
- **LLM**: `mistral:7b-instruct-v0.3-q3_K_M`
- **Embedding Model**: `nomic-embed-text:v1.5`
- **Reranker Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Run Instructions

1. Ensure Ollama is running (`ollama serve`) and the models are pulled:
   ```bash
   ollama pull mistral:7b-instruct-v0.3-q3_K_M
   ollama pull nomic-embed-text:v1.5
   ```
2. Start the FastAPI backend server:
   ```bash
   python generate_response.py
   ```
3. In a new terminal, launch the CLI to chat with Sumit_AI:
   ```bash
   python cli.py
   ```

## Chunking Strategy Explained

This RAG implementation uses a customized **structural-semantic hybrid** chunking method:

1. **Phase 1: Structural Parsing** 
   We first split the Markdown documents by their physical headers (`#`, `##`, `###`) using Langchain's `MarkdownHeaderTextSplitter`. The hierarchy of the document is preserved within the metadata of the split.
   
2. **Phase 2: Semantic Grouping via Cosine Distance**
   We then pass each headed chunk through our semantic grouper. It splits the remaining text into raw sentences. Each sentence is embedded via `nomic-embed-text:v1.5` and we measure the cosine distance between neighboring concepts.
   - If the distance between two sentences exceeds a configured topic-drift threshold, a split is performed automatically. 
   - This ensures that topics are grouped by semantic coherence, preventing the "Temporal Context Blending" that plagues naive fixed-size token chunkers.


   # **Homework Assignment Reference:** [Homework for Chunking Strategy](https://www.notion.so/Wyrd-3193ff2345d3803ea380c01413db3231?source=copy_link)

