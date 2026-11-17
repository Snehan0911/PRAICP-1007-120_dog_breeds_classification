@echo off
title Dog Breeds Classification - Streamlit Web App
echo ========================================================
echo Dog Breeds Classification (PRAICP-1007-120)
echo Starting Streamlit Web Application on http://localhost:8501
echo ========================================================
.\.venv\Scripts\python -m streamlit run streamlit_app.py --server.port 8501
pause
