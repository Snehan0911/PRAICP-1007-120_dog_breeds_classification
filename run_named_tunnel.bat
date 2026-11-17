@echo off
title Dog Breeds Classification - Named Public Tunnel
echo ======================================================================
echo Dog Breeds Classification (PRAICP-1007-120)
echo Exposing Web Application with Custom Name: dog-breeds-classification
echo Public URL: https://dog-breeds-classification.loca.lt
echo ======================================================================
echo Tunnel Password / Endpoint IP (if prompted by browser): 157.45.42.55
npx --yes localtunnel --port 5000 --subdomain dog-breeds-classification
pause
