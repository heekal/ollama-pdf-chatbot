import ollama

def generate_answer(context: str, question: str) -> str:
  prompt = f"""
    Answer the question using ONLY the context below.
    If the answer is not in the context, say "I don't know."

    Context:
    {context}

    Question:
    {question}
    """

  response = ollama.chat(
    model="mistral",
    messages=[{"role": "user", "content": prompt}]
  )

  return response["message"]["content"]
