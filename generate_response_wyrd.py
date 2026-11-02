import sys
import requests
import os
from fastapi import FastAPI
import uvicorn

# RAG imports
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
import re
import math

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from sentence_transformers import CrossEncoder


# models
MODEL_NAME = "mistral:7b-instruct-v0.3-q3_K_M"
EMBEDDING_MODEL_NAME = "nomic-embed-text:v1.5"

# folders
KNOWLEDGE_BASE_DIR = "wyrd_wiki"
CHROMA_DB_DIR = "chroma_db"

# globals
vectorstore = None
reranker_model = None
llm = None

# cache
direct_query_cache = {}


def calculate_cosine_distance(vec1, vec2):
    # cosine calculation
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_vec1 = math.sqrt(sum(a * a for a in vec1))
    norm_vec2 = math.sqrt(sum(b * b for b in vec2))

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 1.0

    return 1.0 - (dot_product / (norm_vec1 * norm_vec2))


def parse_markdown_file(filepath):
    # read md and extract metadata
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    metadata = {}

    match = re.match(r'^---\n(.*?)\n---\n(.*)', content, re.DOTALL)

    if match:
        yaml_content = match.group(1)
        content = match.group(2)

        for line in yaml_content.split('\n'):
            if ':' in line:
                key, val = line.split(':', 1)
                metadata[key.strip()] = val.strip().strip('"\'')
                
    return content, metadata


def advanced_semantic_chunking(document, embeddings_model, distance_threshold=0.80):

    # add header metadata to text
    headers_text = " - ".join([f"{v}" for k, v in document.metadata.items() if k.startswith("Header")])

    if headers_text:
        text = f"[{headers_text}]\n{document.page_content}"
    else:
        text = document.page_content

    raw_sentences = re.split(r'(?<=[.?!])\s+(?=[A-Z])', text)

    sentences = []

    for s in raw_sentences:
        s = s.strip()

        if len(s) > 5:

            if len(s) > 1000:
                parts = [s[i:i+800] for i in range(0, len(s), 800)]
                sentences.extend(parts)
            else:
                sentences.append(s)

    if len(sentences) <= 1:
        return [document]

    sentence_embeddings = embeddings_model.embed_documents(sentences)

    chunks = []

    current_chunk_sentences = [sentences[0]]
    current_chunk_length = len(sentences[0])

    for i in range(len(sentences) - 1):

        dist = calculate_cosine_distance(
            sentence_embeddings[i],
            sentence_embeddings[i + 1]
        )

        next_sentence = sentences[i + 1]
        next_len = len(next_sentence)

        if dist > distance_threshold or current_chunk_length + next_len > 1500:

            chunk_text = " ".join(current_chunk_sentences)

            new_doc = Document(
                page_content=chunk_text,
                metadata=document.metadata.copy()
            )

            chunks.append(new_doc)

            current_chunk_sentences = [next_sentence]
            current_chunk_length = next_len

        else:

            current_chunk_sentences.append(next_sentence)
            current_chunk_length += next_len

    if current_chunk_sentences:

        chunk_text = " ".join(current_chunk_sentences)

        new_doc = Document(
            page_content=chunk_text,
            metadata=document.metadata.copy()
        )

        chunks.append(new_doc)

    return chunks


def initialize_vectorstore():

    global vectorstore
    global reranker_model
    global llm

    if not check_ollama_status(EMBEDDING_MODEL_NAME):
        sys.exit(1)

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        base_url="http://localhost:11434"
    )

    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):

        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings
            )
        except:
            vectorstore = None

    if vectorstore is None:

        documents = []

        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            sys.exit(1)

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )

        for root, _, files in os.walk(KNOWLEDGE_BASE_DIR):

            for filename in files:

                if filename.endswith(".md"):

                    filepath = os.path.join(root, filename)

                    try:

                        content, file_metadata = parse_markdown_file(filepath)

                        header_splits = markdown_splitter.split_text(content)

                        for split in header_splits:

                            split.metadata.update(file_metadata)

                            semantic_chunks = advanced_semantic_chunking(
                                split,
                                embeddings,
                                distance_threshold=0.55
                            )

                            documents.extend(semantic_chunks)

                    except:
                        pass

        if not documents:
            return

        try:

            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=CHROMA_DB_DIR
            )

        except:
            sys.exit(1)

    try:
        reranker_model = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            max_length=512
        )
    except:
        reranker_model = None

    llm = OllamaLLM(
        model=MODEL_NAME,
        temperature=0.1,
        base_url="http://localhost:11434"
    )


def generate_response_with_rag(input_text: str):

    if vectorstore is None:
        return "I cannot access my knowledge base."

    if input_text in direct_query_cache:
        return direct_query_cache[input_text]

    try:

        retrieved_docs = vectorstore.similarity_search(input_text, k=40)

        if reranker_model:

            pairs = [[input_text, doc.page_content] for doc in retrieved_docs]

            scores = reranker_model.predict(pairs)

            ranked_docs = sorted(
                zip(scores, retrieved_docs),
                key=lambda x: x[0],
                reverse=True
            )

            retrieved_docs = [doc for score, doc in ranked_docs[:6]]

        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        rag_template = """
You are Sumit_Ai a highly capable AI assistant for Wyrd Media Labs. Your primary role is to provide accurate, comprehensive, and "wyrd" answers based on the company's internal wiki and knowledge base. Embody the brand's persona - be sharp, witty, and unapologetically real.

Answer the user's question confidently, **strictly based on the following context**.

STRICT RULE: If the user asks about a specific 'Chapter' or 'Act', you MUST prioritize the text found directly under that specific heading. Do not blend stories from different Acts.
NEVER use the following terms: leverage, innovative, strategic, synergy, or end-to-end. If the context uses them, find a wyrdr way to say it or skip them.

If the information is not present in the provided context, or if the question is outside of the provided material's scope, you MUST ONLY output: "I do not have enough information to answer that." Do not add any persona, apologies, or extra text.

Context:
{context}

Question:
{question}
"""

        prompt = ChatPromptTemplate.from_template(rag_template)

        formatted_prompt = prompt.format(
            context=context_text,
            question=input_text
        )

        response = llm.invoke(formatted_prompt)

        generated_text = response.strip()

        direct_query_cache[input_text] = generated_text

        return generated_text

    except requests.exceptions.ConnectionError:
        return "Could not connect to Ollama."

    except requests.exceptions.Timeout:
        return "Model request timed out."

    except:
        return "Internal error."


def check_ollama_status(model_name):

    try:

        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )

        if response.status_code == 200:

            models = response.json()

            model_names = [
                model['name']
                for model in models.get('models', [])
            ]

            return model_name in model_names

        return False

    except:
        return False


app = FastAPI()


@app.on_event("startup")
async def startup_event():

    if not check_ollama_status(MODEL_NAME):
        sys.exit(1)

    initialize_vectorstore()


@app.post("/chat")
def chat_endpoint(request_data: dict):

    input_text = request_data.get("message")

    if not input_text:
        return {"response": "No message provided."}

    response = generate_response_with_rag(input_text)

    return {"response": response}


if __name__ == "__main__":

    uvicorn.run(
        "generate_response:app",
        host="0.0.0.0",
        port=9000,
        log_level="info",
        reload=False
    )