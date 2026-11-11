
"""FastAPI endpoints for the Document Intelligence Agent."""

import logging
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from openpyxl.utils.exceptions import InvalidFileException
from pdfminer.pdfparser import PDFSyntaxError
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.agent import run_agent
from app.config import get_settings
from app.document_loader import load_document
from app.llm_client import check_ollama_health


logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

ALLOWED_EXTENSIONS = {".pdf", ".csv", ".xlsx"}

ALLOWED_CHART_FILES = {
    "agent_chart.png",
    "agent_histogram.png",
    "agent_line_chart.png",
}

MAX_FILE_SIZE = get_settings().max_upload_mb * 1024 * 1024
CHUNK_SIZE = 1024 * 1024  # 1 MB


class QuestionResponse(BaseModel):
    filename: str
    question: str
    answer: str
    tool_used: str


app = FastAPI(
    title="Document Intelligence Agent API",
    description=(
        "Upload PDF, CSV, or Excel files and ask questions "
        "about their contents."
    ),
    version="1.0.0",
)


def clear_previous_charts() -> None:
    """Remove previous agent charts before a new chart request."""

    for filename in ALLOWED_CHART_FILES:
        chart_path = OUTPUTS_DIR / filename

        try:
            chart_path.unlink(missing_ok=True)
        except OSError:
            logger.exception(
                "Could not remove previous chart: %s",
                chart_path,
            )
            raise


@app.middleware("http")
async def log_requests(request, call_next):
    """Log API requests and their response status."""

    logger.info(
        "Request started: %s %s",
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)

        logger.info(
            "Request completed: %s %s - %s",
            request.method,
            request.url.path,
            response.status_code,
        )

        return response

    except Exception:
        logger.exception(
            "Unhandled error while processing %s %s",
            request.method,
            request.url.path,
        )
        raise


@app.get("/")
def home():
    """Return basic API status."""

    return {
        "status": "ok",
        "message": "Document Intelligence Agent API is running.",
    }


@app.get("/health")
def health_check():
    """Check API and Ollama readiness."""

    try:
        return check_ollama_health()

    except Exception:
        logger.exception("Ollama health check failed")

        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama is unavailable or the configured model "
                "is missing."
            ),
        ) from None


@app.get("/charts/{filename}")
def download_chart(filename: str):
    """Return a generated chart image securely."""

    if filename not in ALLOWED_CHART_FILES:
        raise HTTPException(
            status_code=404,
            detail="Chart not found.",
        )

    chart_path = OUTPUTS_DIR / filename

    if not chart_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Chart not found.",
        )

    return FileResponse(
        path=chart_path,
        media_type="image/png",
        filename=filename,
    )


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    """Upload a document and ask the agent a question."""

    # Normalize both Windows and Linux-style paths.
    filename = (file.filename or "").replace("\\", "/").split("/")[-1]
    extension = Path(filename).suffix.lower()

    if not filename or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload PDF, CSV, or XLSX.",
        )

    question = question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    temp_path = None

    try:
        # Stream the upload to disk in small chunks.
        total_size = 0

        with NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

            while True:
                chunk = await file.read(CHUNK_SIZE)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File is too large. Maximum size is "
                            f"{get_settings().max_upload_mb} MB."
                        ),
                    )

                temp_file.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        logger.info(
            "Upload received: filename=%s, size=%d bytes",
            filename,
            total_size,
        )

        # Load the document without blocking the async event loop.
        load_start = time.perf_counter()

        logger.info("Starting document loading: %s", filename)

        document = await run_in_threadpool(
            load_document,
            str(temp_path),
        )

        logger.info(
            "Document loading completed in %.2f seconds: %s",
            time.perf_counter() - load_start,
            filename,
        )

        # Detect chart-related requests.
        question_text = question.lower()

        chart_requested = any(
            word in question_text
            for word in (
                "chart",
                "plot",
                "graph",
                "visualize",
                "visualise",
                "histogram",
                "distribution",
                "trends",
                "over time",
                "time series",
                "line chart",
                "line graph",
                "monthly",
                "daily",
                "weekly",
                "yearly",
                "by date",
                "by month",
                "by day",
                "by year",
            )
        )

        if chart_requested:
            clear_previous_charts()

        # Run the agent in a thread pool.
        agent_start = time.perf_counter()

        logger.info("Starting agent execution: %s", filename)

        result = await run_in_threadpool(
            run_agent,
            document,
            question,
        )

        logger.info(
            "Agent execution completed in %.2f seconds. Tool: %s",
            time.perf_counter() - agent_start,
            result.tool_used,
        )

        return QuestionResponse(
            filename=filename,
            question=question,
            answer=result.answer,
            tool_used=result.tool_used,
        )

    except HTTPException:
        raise

    except (
        ValueError,
        UnicodeError,
        EOFError,
        BadZipFile,
        PDFSyntaxError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        InvalidFileException,
    ) as exc:
        logger.warning(
            "Invalid or unreadable document '%s': %s",
            filename,
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded document is invalid, corrupted, "
                "or could not be read. Please check the file."
            ),
        ) from exc

    except Exception:
        logger.exception(
            "Unexpected error while processing document: %s",
            filename,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The agent could not process this document. "
                "Check that Ollama is running and the model is installed."
            ),
        ) from None

    finally:
        await file.close()

        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.exception(
                    "Could not remove temporary upload: %s",
                    temp_path,
                )
