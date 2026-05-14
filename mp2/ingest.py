import os
import chromadb
import ollama

CORPUS_DIR = "07_cdram-corpus"
CHROMA_DIR = "chroma_db"
COLLECTION = "cdram"
CHUNK_SIZE = 500   # palavras por chunk
OVERLAP    = 50    # palavras de sobreposição entre chunks


def load_documents(directory):
    docs = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".md"):
            path = os.path.join(directory, filename)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            docs.append({"filename": filename, "text": text})
    return docs


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


def embed(text):
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def ingest():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # apaga e recria para começar sempre do zero
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)

    docs = load_documents(CORPUS_DIR)
    print(f"Documentos carregados: {len(docs)}")

    chunk_id = 0
    for doc in docs:
        chunks = chunk_text(doc["text"])
        print(f"  {doc['filename']}: {len(chunks)} chunks")
        for chunk in chunks:
            collection.add(
                ids=[str(chunk_id)],
                embeddings=[embed(chunk)],
                documents=[chunk],
                metadatas=[{"source": doc["filename"]}],
            )
            chunk_id += 1

    print(f"\nPronto! {chunk_id} chunks indexados.")


if __name__ == "__main__":
    ingest()
