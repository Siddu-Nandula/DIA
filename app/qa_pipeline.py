
import logging
import time
from dataclasses import dataclass

from app.document_loader import DocumentResult
from app.llm_client import generate_response
from app.retrieval import DocumentRetriever


logger = logging.getLogger(__name__)


@dataclass
class QAResult:
    """Answer and the document passages used to produce it."""

    answer: str
    sources: list[dict]


def answer_question(
    document: DocumentResult,
    question: str,
    top_k: int = 3,
) -> QAResult:
    """Answer a question using retrieved document evidence and local Qwen."""

    total_start = time.perf_counter()

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if not document.text or not document.text.strip():
        raise ValueError("Document contains no searchable text.")

    # 1. Build or reuse the document vector index.
    retriever = DocumentRetriever()

    start = time.perf_counter()

    retriever.build_index(
        document.text,
        filename=document.filename,
    )

    # Safely support test mock retrievers without a cache_hit attribute.
    cache_hit = getattr(retriever, "cache_hit", False)

    logger.warning(
        "RETRIEVAL CACHE: %s",
        "HIT - reused existing FAISS index"
        if cache_hit
        else "MISS - generated new embeddings and FAISS index",
    )

    logger.info(
        "RETRIEVAL TIMING: index build/reuse = %.2f seconds",
        time.perf_counter() - start,
    )

    # 2. Retrieve passages relevant to the question.
    start = time.perf_counter()

    retrieved_chunks = retriever.search(
        question.strip(),
        top_k=top_k,
    )

    logger.info(
        "RETRIEVAL TIMING: search = %.2f seconds",
        time.perf_counter() - start,
    )

    if not retrieved_chunks:
        raise ValueError("No relevant document passages were retrieved.")

    # 3. Construct source-labelled evidence for the LLM.
    start = time.perf_counter()

    context_parts = []
    sources = []

    for source_number, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {source_number}]\n"
            f"Filename: {chunk.metadata['filename']}\n"
            f"Chunk: {chunk.metadata['chunk_index']}\n"
            f"Content:\n{chunk.text}"
        )

        sources.append(
            {
                "source": source_number,
                "filename": chunk.metadata["filename"],
                "chunk_index": chunk.metadata["chunk_index"],
                "score": chunk.score,
                "text": chunk.text,
            }
        )

    context = "\n\n".join(context_parts)

    # 4. Construct the evidence-grounded prompt.
    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the evidence provided below.

Rules:
- Do not invent facts, figures, or details.
- If the evidence does not contain the answer, say:
  "The provided document does not contain enough information to answer."
- Cite supporting evidence using labels such as [Source 1].
- Clearly distinguish facts from any interpretation.

DOCUMENT EVIDENCE:
{context}

USER QUESTION:
{question.strip()}

ANSWER:
""".strip()

    logger.info(
        "QA TIMING: prompt construction = %.2f seconds",
        time.perf_counter() - start,
    )

    # 5. Generate the answer using Qwen.
    start = time.perf_counter()

    answer = generate_response(prompt)

    logger.info(
        "LLM TIMING: answer generation = %.2f seconds",
        time.perf_counter() - start,
    )

    logger.warning(
        "QA TIMING: total pipeline = %.2f seconds",
        time.perf_counter() - total_start,
    )

    # 6. Return the answer and its source passages.
    return QAResult(
        answer=answer,
        sources=sources,
    )
