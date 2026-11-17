@echo off
title Push Dog Breeds Classification to GitHub
echo ======================================================================
echo Dog Breeds Classification (PRAICP-1007-120)
echo Push Codebase to Your Public GitHub Repository
echo ======================================================================
echo.
echo Make sure you first created an empty repository at: https://github.com/new
echo with the name: dog-breeds-classification (Public)
echo.
set /p GH_USER="Enter your GitHub username: "
if "%GH_USER%"=="" (
    echo Error: GitHub username cannot be empty.
    pause
    exit /b 1
)

set REPO_URL=https://github.com/%GH_USER%/dog-breeds-classification.git

echo.
echo Setting default branch to main...
git branch -M main

echo Configuring remote origin: %REPO_URL%
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo.
echo ======================================================================
echo Pushing repository to GitHub...
echo (If prompted for password, paste your GitHub Personal Access Token)
echo ======================================================================
git push -u origin main

echo.
echo ======================================================================
echo Successfully pushed!
echo View your repository online at:
echo https://github.com/%GH_USER%/dog-breeds-classification
echo ======================================================================
pause
