import os
import glob as globmod
import pickle
from typing import Any
import numpy as np
import faiss
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from openai import OpenAI

INDEX_DIR = "index"
INDEX_FILE = os.path.join(INDEX_DIR, "leyes.faiss")
CHUNKS_FILE = os.path.join(INDEX_DIR, "leyes_chunks.pkl")

# Default configs
DEFAULT_DATA_DIRS = ["data/sat", "data/leyes"] 
DEFAULT_DATA_DIR = "data/sat"
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_TOP_K = 4
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RETRIEVE_K = 10
DEFAULT_FINAL_K = 4

CATEGORY_MAP: dict[str, str] = {
    "CPEUM":     "constitucional",
    "CCF":       "civil",
    "CNPCF":     "civil",
    "CFPC":      "civil",
    "LFDA":      "civil",
    "LFPCA":     "civil",
    "CPF":       "penal",
    "CNPP":      "penal",
    "LFCDO":     "penal",
    "LFRA":      "penal",
    "LFT":       "laboral",
    "LSS":       "laboral",
    "LFTSE":     "laboral",
    "LIFNVT":    "laboral",
    "LISSSTE":   "laboral",
    "CCom":      "mercantil",
    "LGSM":      "mercantil",
    "LGTOC":     "mercantil",
    "LIC":       "mercantil",
    "LFPDPPP":   "mercantil",
    "LFPA":      "administrativo",
    "LAmp":      "administrativo",
    "LFPRH":     "administrativo",
    "LGRA":      "administrativo",
    "LFPPI":     "propiedad_intelectual",
    "LFCE":      "propiedad_intelectual",
    "LFPC":      "consumidor",
    "LGS":       "salud",
    "LGDNNA":    "salud",
    "LGAMVLV":   "salud",
    "LGEEPA":    "ambiental",
    "LGDFS":     "ambiental",
    "CFF":       "fiscal",
    "LISR":      "fiscal",
    "LIVA":      "fiscal",
    "LIEPS":     "fiscal",
    "RMF_2026":  "fiscal",
}


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
        "data_dirs": config.get("data_dirs") or None,
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

def _infer_doc_type(doc_name: str, data_dir: str) -> str:
    """Infiere el doc_type a partir del nombre del archivo o directorio."""
    upper = doc_name.upper()
    if upper in CATEGORY_MAP:
        return CATEGORY_MAP[upper]
    for part in data_dir.replace("\\", "/").split("/"):
        if part in ("sat",):
            return "fiscal"
        if part in ("leyes",):
            return "ley_federal"
    return "ley_federal"

def load_documents(data_dir: str = DEFAULT_DATA_DIR) -> list[Document]:
    """Loads documents from PDF files in data_dir.

    Creates one LangChain Document per page. Each document stores the page text
    as `page_content` and includes the source file path, document name, page
    number, and doc_type in metadata.
    """
    dirs = [data_dir] if isinstance(data_dir, str) else list(data_dir)
    documents = []

    for directory in dirs:
        for pdf_path in globmod.glob(os.path.join(directory, "**", "*.pdf"), recursive=True):
            doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
            doc_type = _infer_doc_type(doc_name, pdf_path)
            pdf = fitz.open(pdf_path)
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text()
                if not text.strip():
                    continue
                documents.append(Document(
                    page_content=text,
                    metadata={"source": pdf_path, 
                              "doc_name": doc_name, 
                              "page": page_num, 
                              "doc_type": doc_type},
                ))
            pdf.close()

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


def load_index(
        index_dir: str = INDEX_DIR,
) -> tuple[faiss.IndexFlatIP, list[Document]]:
    """Loads a previously built FAISS index and chunk list from disk."""
    index_file = os.path.join(index_dir, "leyes.faiss")
    chunks_file = os.path.join(index_dir, "leyes_chunks.pkl")

    if not os.path.exists(index_file) or not os.path.exists(chunks_file):
        raise FileNotFoundError(
            f"Índice no encontrado en '{index_dir}'. Ejecuta 'python ingest.py' primero."
        )

    index = faiss.read_index(index_file)
    with open(chunks_file, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


SYSTEM_PROMPT = """Eres un asistente especializado en derecho mexicano. Respondes preguntas sobre leyes \
federales de México: derecho civil, penal, laboral, mercantil, administrativo, fiscal, constitucional, \
de salud, ambiental y más. Sigue estas reglas estrictamente:
 
1. Responde ÚNICAMENTE con base en el contexto proporcionado.
2. Responde siempre en español.
3. Si el contexto no contiene la respuesta, di: "No encontré información suficiente en los documentos \
para responder esta pregunta."
4. Cita el artículo o disposición específica cuando sea posible \
(ej. "según el Artículo 123 de la Constitución...", "conforme al Artículo 47 de la LFT...").
5. No uses conocimiento previo fuera del contexto proporcionado.
6. SIEMPRE advierte que tus respuestas son orientativas y no constituyen asesoría legal formal; \
recomienda consultar con un abogado para casos concretos.
7. Al final de tu respuesta, indica las fuentes consultadas con el siguiente formato exacto:
 
Fuentes:
- NOMBRE_DOCUMENTO, página X
- ...
"""


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

        question, doc_filter, type_filter = self.parse_filters(question)
        questions = self.expand_query(question) # Query expansion
        all_results = []

        for q in questions:
            overfetch_k = (k or self.retrieve_k) * 3
            results = retrieve(q, self.index, self.model, self.chunks, k or overfetch_k)
            all_results.extend(results)

        all_results = self.filter_results(all_results, doc_filter, type_filter) # Apply filters to retrieved results

        # Deduplicate by (doc_name, page) keeping highest-scoring chunk per page
        seen_pages = set()
        unique_results = []
        for r in sorted(all_results, key=lambda x: x["score"], reverse=True):
            key = (r["metadata"]["doc_name"], r["metadata"]["page"])
            if key not in seen_pages:
                seen_pages.add(key)
                unique_results.append(r)

        # Reranking
        reranked_results = rerank(
            question,
            unique_results,
            self.reranker,
            self.final_k
        )

        context = "\n\n".join(
            f"[documento: {r['metadata']['doc_name']} | página: {r['metadata']['page']}]\n{r['text']}"
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
        messages=[{"role": "user", "content": f"Genera 3 formas alternativas de formular esta pregunta \
        para búsqueda semántica en leyes federales mexicanas. Devuelve solo las preguntas, \
        una por línea, sin numeración.\n\nPregunta: {question}"}],
        )
        variants = response.choices[0].message.content.strip().split("\n")
        return [question] + variants

    DOC_FILTER_MAP: dict[str, str] = {
        "/cff":   "CFF",
        "/lisr":  "LISR",
        "/liva":  "LIVA",
        "/rmf":   "RMF_2026",
        "/cpeum": "CPEUM",
        "/ccf":   "CCF",
        "/lft":   "LFT",
        "/cpf":   "CPF",
        "/cnpp":  "CNPP",
        "/la":    "LAmp",
        "/lgsm":  "LGSM",
        "/lfpc":  "LFPC",
        "/lgs":   "LGS",
    }
 
    TYPE_FILTER_MAP: dict[str, str] = {
        "/civil":           "civil",
        "/penal":           "penal",
        "/laboral":         "laboral",
        "/mercantil":       "mercantil",
        "/administrativo":  "administrativo",
        "/fiscal":          "fiscal",
        "/constitucional":  "constitucional",
        "/salud":           "salud",
        "/ambiental":       "ambiental",
        "/ip":              "propiedad_intelectual",
        "/consumidor":      "consumidor",
    }
 
    def parse_filters(self, question: str) -> tuple[str, str | None, str | None]:
        """Extrae filtros de la pregunta y devuelve (pregunta_limpia, doc_filter, type_filter)."""
        doc_filter  = None
        type_filter = None
 
        for tag, doc_name in self.DOC_FILTER_MAP.items():
            if tag in question:
                doc_filter = doc_name
                question   = question.replace(tag, "").strip()
                break
 
        for tag, type_name in self.TYPE_FILTER_MAP.items():
            if tag in question:
                type_filter = type_name
                question    = question.replace(tag, "").strip()
                break
 
        return question, doc_filter, type_filter

    def filter_results(
            self,
            results: list[dict],
            doc_name: str | None,
            doc_type: str | None,
    ) -> list[dict]:

        if doc_name is not None:
            results = [r for r in results if r["metadata"]["doc_name"] == doc_name]
        if doc_type is not None:
            results = [r for r in results if r["metadata"]["doc_type"] == doc_type]
        return results

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "Assistant":
        """Initializes the assistant by loading a pre-built index from disk.

        Requires that ingest.py has been run beforehand to build and save the
        FAISS index and chunk list.
        """
        resolved_config = resolve_config(config)

        print("Cargando índice...")
        index, chunks = load_index()
        print(f"  {index.ntotal} vectores, {len(chunks)} chunks")

        embedding_model = SentenceTransformer(resolved_config["embedding_model"])

        client_kwargs = {}
        if resolved_config["api_key"]:
            client_kwargs["api_key"] = resolved_config["api_key"]
        if resolved_config["base_url"]:
            client_kwargs["base_url"] = resolved_config["base_url"]
        client = OpenAI(**client_kwargs)

        print("Listo.\n")
        return cls(index, embedding_model, chunks, client, resolved_config)
