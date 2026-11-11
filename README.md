# Document Intelligence & Analytics Agent

An AI-powered document intelligence application that combines local LLMs, Retrieval-Augmented Generation (RAG), and data analytics to extract information, answer questions, analyze datasets, and generate visualizations from business documents.

## Key Features

- **Multi-format ingestion:** Supports PDF, CSV, and XLSX files.
- **PDF extraction:** Extracts document text and tables.
- **Document question answering:** Answers questions using document context.
- **Retrieval-Augmented Generation (RAG):** Uses FAISS for vector-based document retrieval.
- **Local LLM integration:** Uses Ollama with Qwen3 for local inference.
- **Embeddings:** Uses the `nomic-embed-text` embedding model.
- **Document summarization:** Generates summaries from uploaded documents.
- **Data analytics:** Uses Pandas for numerical analysis and dataset operations.
- **Chart generation:** Generates bar charts, histograms, and line charts using Matplotlib.
- **Agent tool routing:** Routes user requests to appropriate analytics and visualization tools.
- **REST API:** Provides backend endpoints through FastAPI.
- **Interactive interface:** Uses Streamlit for document interaction.
- **Automated testing:** Includes pytest-based tests for core functionality.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Local LLM | Ollama, Qwen3 |
| Embeddings | nomic-embed-text |
| Vector search | FAISS |
| Data processing | Pandas, NumPy |
| PDF extraction | pdfplumber |
| Visualization | Matplotlib |
| Backend | FastAPI |
| Frontend | Streamlit |
| Testing | pytest |

## Project Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd document-intelligence-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and configure the required settings.

### 5. Install local models

Make sure Ollama is installed and running.

```powershell
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

## Running the Application

### Start the FastAPI backend

```powershell
python -m uvicorn app.api:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

### Start the Streamlit interface

```powershell
python -m streamlit run frontend.py
```

## Running Tests

Execute the complete test suite:

```powershell
python -m pytest -q
```

## Supported Documents

- PDF
- CSV
- XLSX

## Project Architecture

The application combines document ingestion, retrieval, local language-model inference, analytics, and visualization tools.

```text
User
  |
  v
Streamlit Interface
  |
  v
FastAPI Backend
  |
  v
Document Intelligence Agent
  |
  +---- Document Processing
  |
  +---- FAISS Retrieval (RAG)
  |
  +---- Ollama / Qwen3
  |
  +---- Analytics Tools
  |
  +---- Chart Generation
  |
  v
Grounded Responses and Visualizations
```

## Project Status

Core development, application integration, and automated testing are complete.

The project is in its final documentation and deployment-readiness stage.

## Future Improvements

Potential future enhancements can be considered separately from the current completed scope.