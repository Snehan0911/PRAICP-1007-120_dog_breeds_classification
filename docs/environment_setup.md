# Dog Breeds Classification — Environment Setup Guide

## 1. Environment Architecture

This project runs inside an isolated Python virtual environment (`.venv`) powered by Python 3.13 (64-bit on Windows). Virtual environments prevent library conflicts between different projects and ensure identical behavior across development and deployment.

---

## 2. Activation Instructions (Windows PowerShell & CMD)

Whenever you open a new terminal in Antigravity IDE or VS Code, activate the environment using:

```powershell
# In Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Or in Command Prompt (cmd.exe):
.\.venv\Scripts\activate.bat
```

When activated, you will see `(.venv)` displayed at the start of your terminal prompt.

---

## 3. Core Libraries & Responsibilities

| Library | Version | Purpose in Dog Breeds Classification |
|---|---|---|
| **TensorFlow / Keras** | 2.21.0 / 3.14.0 | Core deep learning framework for building, training, and running CNN models. |
| **NumPy** | 2.5.5 | High-speed numerical tensor and matrix computations. |
| **Pandas** | 3.1.1 | Structured data management, CSV indexing, and dataset metrics. |
| **Pillow (PIL)** | 13.0.0 | Opening, decoding, validating, and converting raw image files. |
| **OpenCV (cv2)** | 5.0.0 | Computer vision routines and color-map blending for Grad-CAM explainability. |
| **Scikit-Learn** | 1.9.1 | Stratified train/val/test splitting and comprehensive evaluation metrics. |
| **Flask** | 3.1.3 | Web server framework rendering HTML templates for the interactive web portal. |
| **FastAPI / Uvicorn** | 0.163 / 0.54 | Asynchronous REST API server with automatic Swagger documentation. |
| **Streamlit** | 1.63.0 | Real-time interactive AI data dashboard. |
| **TensorBoard** | 2.21.0 | Real-time visualization of loss curves, accuracy, and graph computations. |
| **Plotly** | 7.1.0 | Interactive web-based data charts and probability distributions. |
| **Pytest** | 9.2.0 | Automated unit and integration testing suite. |
| **Ruff** | 0.17.5 | Extremely fast code linter and style formatter. |

---

## 4. Verification Command

To verify that all required libraries are operational, run:

```powershell
.\.venv\Scripts\python.exe -c "import tensorflow, flask, tensorboard, cv2, PIL, sklearn; print('All libraries loaded successfully!')"
```
