
@echo off
cd /d "C:\Users\satya\OneDrive\Desktop\document-intelligence-agent"

start "Document Agent API" cmd /k ".venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak >nul

start "Document Agent UI" cmd /k ".venv\Scripts\python.exe -m streamlit run frontend.py"

exit
