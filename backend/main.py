from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from rag.loader import load_pdf
from rag.chunker import chunk_text
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.llm import generate_answer

app = FastAPI(title="Document Chatbot Backend")

embedder = Embedder()
vectorstore = None

class ChatRequest(BaseModel):
  question: str

@app.post("/health")
async def health_check():
  return {"status": "Oke"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
  global vectorstore

  text = load_pdf(file.file)
  chunks = chunk_text(text)

  embeddings = embedder.embed(chunks)
  vectorstore = VectorStore(embeddings.shape[1])
  vectorstore.add(embeddings, chunks)

  return {"status": "PDF indexed", "chunks": len(chunks)}

@app.post("/chat")
async def chat(req: ChatRequest):
  if vectorstore is None:
    return {"error": "No document uploaded"}

  query_embedding = embedder.embed([req.question])
  contexts = vectorstore.search(query_embedding, k=3)

  context_text = "\n".join(contexts)
  answer = generate_answer(context_text, req.question)

  return {"answer": answer}
