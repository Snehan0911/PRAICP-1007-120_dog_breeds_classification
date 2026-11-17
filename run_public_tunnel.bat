@echo off
title Dog Breeds Classification - Public Cloudflare Tunnel
echo ========================================================
echo Dog Breeds Classification (PRAICP-1007-120)
echo Exposing Local Streamlit Application to Public Internet
echo ========================================================
.\cloudflared.exe tunnel --url http://localhost:8501
pause
