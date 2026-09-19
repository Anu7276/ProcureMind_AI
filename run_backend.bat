@echo off
echo ===================================================
echo   BIS Standards Recommendation Engine - Backend
echo ===================================================
echo Starting FastAPI application on http://localhost:8000 ...
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
pause
