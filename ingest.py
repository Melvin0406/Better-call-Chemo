import os
import pickle

import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from rag import (
    DEFAULT_DATA_DIR,
    build_index,
    load_documents,
    resolve_config,
    split_documents,
)

INDEX_DIR = "index"
INDEX_FILE = os.path.join(INDEX_DIR, "sat.faiss")
CHUNKS_FILE = os.path.join(INDEX_DIR, "sat_chunks.pkl")


def main():
    load_dotenv()
    config = resolve_config({
        "chunk_size": os.getenv("CHUNK_SIZE"),
        "chunk_overlap": os.getenv("CHUNK_OVERLAP"),
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
    })

    print("Cargando documentos...")
    docs = load_documents(DEFAULT_DATA_DIR)
    doc_names = sorted(set(d.metadata["doc_name"] for d in docs))
    print(f"  {len(docs)} páginas de {len(doc_names)} documentos: {', '.join(doc_names)}")

    print("Dividiendo en fragmentos...")
    chunks = split_documents(docs, config["chunk_size"], config["chunk_overlap"])
    print(f"  {len(chunks)} chunks (size={config['chunk_size']}, overlap={config['chunk_overlap']})")

    print(f"Cargando modelo de embedding ({config['embedding_model']})...")
    embedding_model = SentenceTransformer(config["embedding_model"])

    print("Construyendo índice FAISS...")
    index = build_index(chunks, embedding_model)
    print(f"  {index.ntotal} vectors (dim={index.d})")

    print("Saving to disk...")
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"  {INDEX_FILE}")
    print(f"  {CHUNKS_FILE}")
    print("\nListo.")


if __name__ == "__main__":
    main()
