from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Protocol

import numpy as np


DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 160
DEFAULT_TOP_K = 4
SAFETY_DISCLAIMER = (
    "This assistant summarizes medical papers for education only and does not provide "
    "a medical diagnosis. Please consult a qualified clinician for medical advice."
)

MEDICAL_TERMS = {
    "adenocarcinoma": "A type of cancer that starts in gland-like cells that make fluid or mucus.",
    "benign": "Not cancerous.",
    "biopsy": "A procedure where a small tissue sample is removed and examined.",
    "carcinoma": "Cancer that begins in cells lining organs or body surfaces.",
    "ct": "Computed tomography, an imaging scan that uses X-rays to create body images.",
    "lesion": "An area of abnormal tissue seen on imaging or examination.",
    "malignant": "Cancerous and able to invade or spread.",
    "metastasis": "Cancer spread from one part of the body to another.",
    "nodule": "A small rounded area or spot seen on imaging.",
    "opacity": "An area on an image that looks denser or less transparent than surrounding tissue.",
    "radiology": "The medical specialty that interprets imaging studies.",
    "sensitivity": "How well a test detects a condition when it is truly present.",
    "specificity": "How well a test rules out a condition when it is truly absent.",
    "tumor": "An abnormal growth of tissue that may be benign or malignant.",
}


@dataclass(frozen=True)
class PaperChunk:
    text: str
    source: str
    page: int | None
    chunk_id: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: PaperChunk
    score: float


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        ...


def save_uploaded_pdfs(uploaded_files: Iterable, output_dir: str | Path) -> list[Path]:
    """Save Streamlit UploadedFile-like objects and return local PDF paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for uploaded_file in uploaded_files:
        filename = Path(uploaded_file.name).name
        if not filename.lower().endswith(".pdf"):
            continue
        destination = output_path / filename
        destination.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(destination)

    return saved_paths


def uploaded_pdf_to_temp_path(uploaded_file) -> Path:
    """Persist one uploaded PDF to a temporary file for parsers that require a path."""
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def extract_pdf_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        import fitz  # PyMuPDF

        pages = []
        with fitz.open(path) as document:
            for page_index, page in enumerate(document, start=1):
                pages.append((page_index, page.get_text("text")))
        return pages
    except ModuleNotFoundError:
        pass

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [
            (page_index, page.extract_text() or "")
            for page_index, page in enumerate(reader.pages, start=1)
        ]
    except ModuleNotFoundError as error:
        raise ImportError(
            "Install PyMuPDF or pypdf to extract PDF text: "
            "`python3 -m pip install pymupdf` or `python3 -m pip install pypdf`."
        ) from error


def split_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_pdf_chunks(pdf_paths: Iterable[str | Path]) -> list[PaperChunk]:
    chunks: list[PaperChunk] = []
    chunk_id = 0
    for pdf_path in pdf_paths:
        path = Path(pdf_path)
        for page_number, page_text in extract_pdf_pages(path):
            for chunk_text in split_text(page_text):
                chunks.append(
                    PaperChunk(
                        text=chunk_text,
                        source=path.name,
                        page=page_number,
                        chunk_id=chunk_id,
                    )
                )
                chunk_id += 1
    return chunks


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as error:
            raise ImportError(
                "Install sentence-transformers to create local embeddings: "
                "`python3 -m pip install sentence-transformers`."
            ) from error
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.astype(np.float32)


class OpenAIEmbeddings:
    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as error:
            raise ImportError(
                "Install the OpenAI SDK to use OpenAI embeddings: `python3 -m pip install openai`."
            ) from error
        self.client = OpenAI()
        self.model_name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
        return normalize_vectors(vectors)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (vectors / norms).astype(np.float32)


class FaissChunkStore:
    def __init__(self, chunks: list[PaperChunk], embeddings: np.ndarray) -> None:
        try:
            import faiss
        except ModuleNotFoundError as error:
            raise ImportError("Install FAISS: `python3 -m pip install faiss-cpu`.") from error

        self.chunks = chunks
        self.embeddings = normalize_vectors(embeddings)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        query_embedding = normalize_vectors(query_embedding.reshape(1, -1))
        scores, indices = self.index.search(query_embedding, top_k)
        return [
            RetrievedChunk(chunk=self.chunks[int(index)], score=float(score))
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]


class InMemoryChunkStore:
    """Fallback cosine store for development when FAISS or ChromaDB is unavailable."""

    def __init__(self, chunks: list[PaperChunk], embeddings: np.ndarray) -> None:
        self.chunks = chunks
        self.embeddings = normalize_vectors(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        query_embedding = normalize_vectors(query_embedding.reshape(1, -1))[0]
        scores = self.embeddings @ query_embedding
        best_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(chunk=self.chunks[int(index)], score=float(scores[int(index)]))
            for index in best_indices
        ]


class MedicalPaperRAGAssistant:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        vector_backend: str = "faiss",
    ) -> None:
        self.embedding_provider = embedding_provider or SentenceTransformerEmbeddings()
        self.vector_backend = vector_backend
        self.chunks: list[PaperChunk] = []
        self.store = None

    def build_index(self, pdf_paths: Iterable[str | Path]) -> None:
        self.chunks = load_pdf_chunks(pdf_paths)
        if not self.chunks:
            raise ValueError("No text chunks were extracted from the uploaded papers.")

        embeddings = self.embedding_provider.embed([chunk.text for chunk in self.chunks])
        if self.vector_backend == "faiss":
            self.store = FaissChunkStore(self.chunks, embeddings)
        elif self.vector_backend == "memory":
            self.store = InMemoryChunkStore(self.chunks, embeddings)
        else:
            raise ValueError("Supported vector backends are: faiss, memory.")

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        if self.store is None:
            raise RuntimeError("Build the paper index before asking questions.")
        query_embedding = self.embedding_provider.embed([question])[0]
        return self.store.search(query_embedding, top_k=top_k)

    def answer(self, question: str, top_k: int = DEFAULT_TOP_K) -> str:
        retrieved = self.retrieve(question, top_k=top_k)
        return generate_simple_answer(question, retrieved)


def explain_medical_terms(text: str) -> list[tuple[str, str]]:
    lowered = text.lower()
    found_terms = []
    for term, explanation in MEDICAL_TERMS.items():
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            found_terms.append((term, explanation))
    return found_terms


def choose_explanatory_sentences(retrieved_chunks: list[RetrievedChunk], limit: int = 3) -> list[str]:
    sentences = []
    for retrieved in retrieved_chunks:
        chunk_sentences = re.split(r"(?<=[.!?])\s+", retrieved.chunk.text)
        for sentence in chunk_sentences:
            clean_sentence = sentence.strip()
            if 60 <= len(clean_sentence) <= 320:
                sentences.append(clean_sentence)
            if len(sentences) >= limit:
                return sentences
    return sentences


def format_sources(retrieved_chunks: list[RetrievedChunk]) -> str:
    seen = set()
    sources = []
    for retrieved in retrieved_chunks:
        page_label = f"page {retrieved.chunk.page}" if retrieved.chunk.page else "page unknown"
        key = (retrieved.chunk.source, page_label)
        if key in seen:
            continue
        seen.add(key)
        sources.append(f"{retrieved.chunk.source}, {page_label}")
    return "; ".join(sources) if sources else "No source reference available."


def generate_simple_answer(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    if not retrieved_chunks:
        return (
            "Simple explanation: I could not find relevant text in the uploaded papers. "
            "Key medical terms explained: none found. "
            f"Source paper/page reference: No source reference available. {SAFETY_DISCLAIMER}"
        )

    context = " ".join(retrieved.chunk.text for retrieved in retrieved_chunks)
    explanatory_sentences = choose_explanatory_sentences(retrieved_chunks)
    simple_explanation = " ".join(explanatory_sentences)
    if not simple_explanation:
        simple_explanation = (
            "The retrieved paper sections appear related to your question, but the wording "
            "is technical. The safest interpretation is to treat them as research context, "
            "not as patient-specific advice."
        )

    terms = explain_medical_terms(f"{question} {context}")
    if terms:
        terms_text = "; ".join(f"{term}: {explanation}" for term, explanation in terms[:8])
    else:
        terms_text = "No common glossary terms were detected in the retrieved sections."

    return (
        f"Simple explanation: {simple_explanation}\n\n"
        f"Key medical terms explained: {terms_text}\n\n"
        f"Source paper/page reference: {format_sources(retrieved_chunks)}\n\n"
        f"Safety disclaimer: {SAFETY_DISCLAIMER}"
    )


if __name__ == "__main__":
    print(
        "MedicalPaperRAGAssistant is ready. Build an index with PDF paths, then call "
        "`answer(question)` to retrieve paper-grounded explanations."
    )
