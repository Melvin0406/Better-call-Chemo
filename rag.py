# You might need the following imports. Feel free to change it if you opt for different libraries.

import os
import glob as globmod
from typing import Any
import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from openai import OpenAI

# Default configs
DEFAULT_DATA_DIR = "data"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 32
DEFAULT_TOP_K = 4
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RETRIEVE_K = 10
DEFAULT_FINAL_K = 4


def _parse_int_setting(name: str, value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc
    return parsed


def resolve_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolves runtime configuration with defaults and typed settings."""
    config = config or {}

    resolved = {
        "api_key": config.get("api_key", None),
        "base_url": config.get("base_url", None),
        "model": config.get("model") or DEFAULT_LLM_MODEL,
        "embedding_model": config.get("embedding_model") or DEFAULT_EMBEDDING_MODEL,
        "top_k": _parse_int_setting(
            "TOP_K",
            config.get("top_k") or DEFAULT_TOP_K,
        ),
        "chunk_size": _parse_int_setting(
            "CHUNK_SIZE",
            config.get("chunk_size") or DEFAULT_CHUNK_SIZE,
        ),
        "chunk_overlap": _parse_int_setting(
            "CHUNK_OVERLAP",
            config.get("chunk_overlap") or DEFAULT_CHUNK_OVERLAP,
        ),
        "rerank_model": config.get("rerank_model") or DEFAULT_RERANK_MODEL,

        "retrieve_k": _parse_int_setting(
            "RETRIEVE_K",
            config.get("retrieve_k") or DEFAULT_RETRIEVE_K,
        ),

        "final_k": _parse_int_setting(
            "FINAL_K",
            config.get("final_k") or DEFAULT_FINAL_K,
        ),
    }

    if resolved["top_k"] <= 0:
        raise ValueError("TOP_K must be > 0")
    if resolved["chunk_size"] <= 0:
        raise ValueError("CHUNK_SIZE must be > 0")
    if resolved["chunk_overlap"] < 0:
        raise ValueError("CHUNK_OVERLAP must be >= 0")
    if resolved["chunk_overlap"] >= resolved["chunk_size"]:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    if resolved["retrieve_k"] <= 0:
        raise ValueError("RETRIEVE_K must be > 0")
    if resolved["final_k"] <= 0:
        raise ValueError("FINAL_K must be > 0")
    if resolved["final_k"] > resolved["retrieve_k"]:
        raise ValueError("FINAL_K cannot be greater than RETRIEVE_K")

    return resolved


def load_documents(data_dir: str = DEFAULT_DATA_DIR) -> list[Document]:
    """Loads documents from the personal data folders.

    The collection contains one LangChain Document per `.txt` file in the
    emails, notes, SMS, and calendar folders. Each document stores the file text
    as `page_content` and includes metadata for the source file path and
    document type.
    """
    documents = []

    for dir in os.listdir(data_dir):
        for file in os.listdir(data_dir + "/" + dir):
            path = data_dir + "/" + dir + "/" + file

            doc_type = dir
            if dir == "notes": doc_type = "note"
            elif dir == "emails": doc_type = "email"
            metadata = {
                "source": path,
                "doc_type": doc_type,
            }

            with open(path) as f:
                raw_text = f.read()
            doc = Document(page_content=raw_text, metadata=metadata)
            documents.append(doc)

    return documents


def split_documents(
        docs: list[Document],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Splits documents into overlapping chunks.

    The resulting chunked Document objects use the configured chunk size and
    overlap while preserving the original document metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.split_documents(docs)


def build_index(
        chunks: list[Document],
        embedding_model: SentenceTransformer,
) -> faiss.IndexFlatIP:
    """Creates a FAISS inner-product index for embedded document chunks.

    The index contains normalized float32 embeddings generated from each
    chunk's text with the provided embedding model.
    """
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.encode(
        texts, normalize_embeddings=True
    ).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def retrieve(
        query: str,
        index: faiss.IndexFlatIP,
        model: SentenceTransformer,
        chunks: list[Document],
        k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Gets the most relevant chunks for a query.

    Results are ordered by similarity and include the chunk text, similarity
    score, and metadata for each matching chunk.
    """
    query_embedding = model.encode(
        [query], normalize_embeddings=True
    ).astype(np.float32)
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "text": chunks[idx].page_content,
            "score": float(score),
            "metadata": chunks[idx].metadata,
        })

    return results


def rerank(
        query: str,
        results: list[dict],
        reranker: CrossEncoder,
        top_n: int,
) -> list[dict]:
    """
    Reorders retrieved documents using a cross-encoder reranker.

    The cross-encoder evaluates each (query, document) pair with
    higher precision than cosine similarity.
    """

    pairs = [(query, r["text"]) for r in results]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, results),
        key=lambda x: x[0],
        reverse=True
    )

    reranked_results = []

    for score, result in ranked[:top_n]:
        result["rerank_score"] = float(score)
        reranked_results.append(result)

    return reranked_results


SYSTEM_PROMPT = """You are a technical assistant. Answer the user's question using ONLY the provided context. \
Follow these rules:
- If the context doesn't contain the answer, say "I don't have enough information to answer this question."
- Be concise and precise.
- Do not use prior knowledge outside of the context.
- At the end of the message, cite the files with information that was actually used to formulate the answer \
using the following format:
"References:
- path/to/file1.txt
- path/to/file2.txt
- ..." """


class Assistant:
    """Stateful RAG assistant.

    The assistant owns the pipeline components, resolved configuration, and
    conversation history. Questions are answered with retrieved document context
    and the configured chat model.
    """

    def __init__(
            self,
            index: faiss.IndexFlatIP,
            model: SentenceTransformer,
            chunks: list[Document],
            client: OpenAI,
            config: dict[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.model = model
        self.chunks = chunks
        self.client = client
        self.config = resolve_config(config)
        self.llm_model = self.config["model"]
        self.top_k = self.config["top_k"]
        self.history: list[dict[str, str]] = []
        self.retrieve_k = self.config["retrieve_k"]
        self.final_k = self.config["final_k"]
        self.reranker = CrossEncoder(self.config["rerank_model"])

    def ask(self, question: str, k: int | None = None) -> str:
        """Generates an answer from the retrieved context and conversation history.

        The current question is combined with relevant document chunks, previous
        conversation messages, and the system prompt. The assistant response is
        appended to history alongside the user message.
        """

        question, doc_filter = self.parse_filters(question)

        questions = self.expand_query(question) # Query expansion
        all_results = []

        for q in questions:
            overfetch_k = (k or self.retrieve_k) * 3
            results = retrieve(q, self.index, self.model, self.chunks, k or overfetch_k)
            all_results.extend(results)

        all_results = self.filter_results(all_results, doc_filter) # Optional document type filtering

        # Sort by score and source
        seen_sources = set()
        unique_results = []
        for r in sorted(all_results, key=lambda x: x["score"], reverse=True):
            source = r["metadata"]["source"]
            if source not in seen_sources:
                seen_sources.add(source)
                unique_results.append(r)

        # Reranking
        reranked_results = rerank(
            question,
            unique_results,
            self.reranker,
            self.final_k
        )

        context = "\n\n".join(
            f"[source: {r['metadata']['source']} | type: {r['metadata']['doc_type']}]\n{r['text']}"
            for r in reranked_results
        )
        user_message = f"Context:\n{context}\n\nQuestion: {question}"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
        )
        answer = response.choices[0].message.content

        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": answer})

        return answer

    def clear_history(self) -> None:
        """Empties the conversation history."""
        self.history.clear()

    def expand_query(self, question: str) -> list[str]:
        response = self.client.chat.completions.create(
        model=self.llm_model,
        messages=[{"role": "user", "content": f"Generate 3 alternative \
        phrasings of this question for semantic search. Return only the \
        questions, one per line.\n\nQuestion: {question}"}],
        )
        variants = response.choices[0].message.content.strip().split("\n")
        return [question] + variants

    def parse_filters(self, question: str) -> tuple[str, str | None]:

        filter_map = {
            "/notes": "note",
            "/email": "email",
            "/sms": "sms",
            "/calendar": "calendar",
        }

        detected_filter = None

        for tag, doc_type in filter_map.items():
            if tag in question:
                detected_filter = doc_type
                question = question.replace(tag, "").strip()

        return question, detected_filter

    def filter_results(
            self,
            results: list[dict],
            doc_type: str | None,
    ) -> list[dict]:

        if doc_type is None:
            return results

        return [
            r for r in results
            if r["metadata"]["doc_type"] == doc_type
        ]

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "Assistant":
        """Initializes the components required by the assistant and instantiates it

        The pipeline includes resolved configuration, loaded documents, chunked
        documents, an embedding model, a FAISS index, and an OpenAI-compatible
        client.
        """
        resolved_config = resolve_config(config)

        print("Loading documents...")
        docs = load_documents()
        print(f"  Loaded {len(docs)} documents")

        print("Splitting into chunks...")
        chunks = split_documents(
            docs,
            chunk_size=resolved_config["chunk_size"],
            chunk_overlap=resolved_config["chunk_overlap"],
        )
        print(f"  Created {len(chunks)} chunks")

        embedding_model = SentenceTransformer(resolved_config["embedding_model"])

        print("Building FAISS index...")
        index = build_index(chunks, embedding_model)
        print(f"  Indexed {index.ntotal} vectors (dim={index.d})")

        client_kwargs = {}
        if resolved_config["api_key"]:
            client_kwargs["api_key"] = resolved_config["api_key"]
        if resolved_config["base_url"]:
            client_kwargs["base_url"] = resolved_config["base_url"]
        client = OpenAI(**client_kwargs)

        print("Ready!\n")
        return cls(index, embedding_model, chunks, client, resolved_config)
