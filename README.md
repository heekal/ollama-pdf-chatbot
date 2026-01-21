# PDF RAG Chatbot (Local LLM)

## Overview
A local Retrieval-Augmented Generation (RAG) chatbot that answers questions from uploaded PDF documents using local LLM inference.

## Architecture
User Question  
↓  
Embedding Model (SentenceTransformers)  
↓  
FAISS Vector Search  
↓  
Relevant PDF Chunks  
↓  
LLM (Ollama – Local Inference)  
↓  
Answer  

## Tech Stack
- Python
- Streamlit
- SentenceTransformers
- FAISS
- Ollama (Mistral)

## Key Points
- Fully local inference (no cloud, no data leakage)
- RAG used to reduce hallucination
- Designed with scalable inference architecture in mind
