
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock

import faiss
import numpy as np
import ollama

from app.config import get_settings


EMBEDDING_MODEL = "nomic-embed-text"
CACHE_MAX_ENTRIES = 3

# Stores recently used document indexes.
# Oldest entries are removed when the cache reaches its limit.
_INDEX_CACHE = OrderedDict()
_CACHE_LOCK = RLock()


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict


def clear_index_cache() -> None:
    """Clear all cached document indexes."""

    with _CACHE_LOCK:
        _INDEX_CACHE.clear()


class DocumentRetriever:
    """Chunk document text and retrieve relevant passages using FAISS."""

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller than chunk_size."
            )

        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        settings = get_settings()
        self.client = ollama.Client(host=settings.ollama_base_url)

        self.chunks = []
        self.index = None
        self.dimension = None
        self.cache_hit = False

    def split_text(self, text: str) -> list[str]:
        """Split text into overlapping character-based chunks."""

        if not text or not text.strip():
            return []

        text = text.strip()
        step = self.chunk_size - self.chunk_overlap

        return [
            text[start:start + self.chunk_size]
            for start in range(0, len(text), step)
        ]

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Create normalized embedding vectors using Ollama."""

        response = self.client.embed(
            model=self.embedding_model,
            input=texts,
        )

        vectors = np.asarray(
            response["embeddings"],
            dtype=np.float32,
        )

        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("Embedding model returned an unexpected shape.")

        if not np.isfinite(vectors).all():
            raise ValueError("Embedding model returned invalid values.")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        if np.any(norms == 0):
            raise ValueError("Embedding model returned a zero vector.")

        vectors /= norms
        return vectors

    def _get_cache_key(self, text: str, filename: str) -> str:
        """Create a unique key for document content and index settings."""

        content_hash = sha256(text.encode("utf-8")).hexdigest()

        configuration = (
            f"{filename}|{self.embedding_model}|"
            f"{self.chunk_size}|{self.chunk_overlap}"
        )

        return sha256(
            f"{content_hash}|{configuration}".encode("utf-8")
        ).hexdigest()

    def build_index(
        self,
        text: str,
        filename: str = "document",
    ) -> int:
        """Build a FAISS index or reuse an existing cached index."""

        chunks = self.split_text(text)

        if not chunks:
            raise ValueError("Cannot index empty document text.")

        cache_key = self._get_cache_key(text.strip(), filename)

        with _CACHE_LOCK:

            # Reuse the existing index if this document is cached.
            if cache_key in _INDEX_CACHE:
                cached = _INDEX_CACHE.pop(cache_key)
                _INDEX_CACHE[cache_key] = cached

                self.index = cached["index"]
                self.dimension = cached["dimension"]
                self.chunks = cached["chunks"]
                self.cache_hit = True

                return len(self.chunks)

            # Generate embeddings only when the document is not cached.
            vectors = self._embed(chunks)

            dimension = vectors.shape[1]

            index = faiss.IndexFlatIP(dimension)
            index.add(vectors)

            indexed_chunks = [
                {
                    "text": chunk,
                    "metadata": {
                        "filename": filename,
                        "chunk_index": i,
                    },
                }
                for i, chunk in enumerate(chunks)
            ]

            self.index = index
            self.dimension = dimension
            self.chunks = indexed_chunks
            self.cache_hit = False

            # Store the completed index.
            _INDEX_CACHE[cache_key] = {
                "index": index,
                "dimension": dimension,
                "chunks": indexed_chunks,
            }

            # Remove the oldest cached document if necessary.
            while len(_INDEX_CACHE) > CACHE_MAX_ENTRIES:
                _INDEX_CACHE.popitem(last=False)

        return len(self.chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """Return the most semantically similar document chunks."""

        if self.index is None:
            raise RuntimeError("Build the index before searching.")

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_vector = self._embed([query.strip()])

        if query_vector.shape[1] != self.dimension:
            raise ValueError("Query embedding dimension does not match index.")

        count = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, count)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            item = self.chunks[int(index)]

            results.append(
                RetrievedChunk(
                    text=item["text"],
                    score=float(score),
                    metadata=item["metadata"].copy(),
                )
            )

        return results
