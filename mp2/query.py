import chromadb
import ollama

CHROMA_DIR = "chroma_db"
COLLECTION = "cdram"
TOP_K = 5


def embed(text):
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def retrieve(question):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION)
    results = collection.query(
        query_embeddings=[embed(question)],
        n_results=TOP_K,
    )
    return results["documents"][0], results["metadatas"][0]


def generate(question, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""Responde à pergunta usando apenas o contexto abaixo.
Se a resposta não estiver no contexto, diz "Não sei, essa informação não está nos documentos."

Contexto:
{context}

Pergunta: {question}
Resposta:"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def rag_query(question):
    chunks, sources = retrieve(question)

    print("\n--- Chunks recuperados ---")
    for i, (chunk, meta) in enumerate(zip(chunks, sources)):
        print(f"[{i+1}] {meta['source']}: {chunk[:120]}...")

    print("\n--- Resposta ---")
    answer = generate(question, chunks)
    print(answer)
    return answer


if __name__ == "__main__":
    question = input("Pergunta: ")
    rag_query(question)
