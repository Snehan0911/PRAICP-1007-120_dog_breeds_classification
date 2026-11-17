@echo off
title Deploy to Hugging Face Spaces
echo ======================================================================
echo Dog Breeds Classification (PRAICP-1007-120)
echo Hugging Face Spaces Automated Deployment
echo ======================================================================
echo.
set /p HF_USERNAME="Enter your Hugging Face username [default: snehan0911]: "
if "%HF_USERNAME%"=="" set HF_USERNAME=snehan0911

set SPACE_NAME=dog-breeds-classification
set REPO_URL=https://huggingface.co/spaces/%HF_USERNAME%/%SPACE_NAME%

echo.
echo Preparing clean deployment package...
git checkout -B deploy

echo Configuring Hugging Face remote: %REPO_URL%
git remote remove space 2>nul
git remote add space %REPO_URL%

echo.
echo ======================================================================
echo Pushing codebase and deep learning model to Hugging Face...
echo Note: When Git asks for Username, enter: %HF_USERNAME%
echo When Git asks for Password, paste your Hugging Face Access Token.
echo (Get a free token here: https://huggingface.co/settings/tokens)
echo ======================================================================
echo.
git push -f space deploy:main

echo.
git checkout master
echo.
echo ======================================================================
echo Deployment push complete!
echo Your app will build and go live at:
echo %REPO_URL%
echo ======================================================================
pause
