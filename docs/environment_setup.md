# Environment Setup & Engineering Standards
**Project Name:** Dog Breeds Classification  
**Project ID:** PRAICP-1007-120  

---

## 1. Environment Architecture

This project is built using **Python 3.13.3** running in an isolated virtual environment (`.venv`) on Windows 11.

### Virtual Environment Activation

To activate the virtual environment in Antigravity IDE or PowerShell:

```powershell
# In Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

If script execution policy restricts running scripts, execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

For standard Windows Command Prompt (`cmd.exe`):
```cmd
.venv\Scripts\activate.bat
```

---

## 2. Core Dependencies & Technical Rationale

| Package | Version | Purpose in Pipeline |
| :--- | :--- | :--- |
| `tensorflow` | `2.21.0` | Deep learning framework, computational graph execution, and Keras 3 layers |
| `tensorboard` | `2.21.0` | Real-time monitoring of training/validation loss, accuracy, and learning rates |
| `numpy` | `>=2.0.0` | High-performance multi-dimensional array operations and tensor conversions |
| `pandas` | `>=2.2.0` | Metadata structuring, `labels.csv` parsing, and stratified dataset splitting |
| `scipy` | `>=1.15.0` | Scientific computing utilities and statistical metrics |
| `pillow` | `>=11.0.0` | Byte-level image decoding, format validation, and thumbnail generation |
| `opencv-python` | `>=4.10.0` | Image processing, colorspace transformations, and Grad-CAM heatmap overlays |
| `scikit-learn` | `>=1.6.0` | Stratified train/val/test splitting, confusion matrices, and classification reports |
| `flask` | `>=3.1.0` | Lightweight web application framework serving the 7 application HTML views |
| `fastapi` | `>=0.115.0` | High-throughput asynchronous REST API serving JSON prediction endpoints |
| `uvicorn` | `>=0.32.0` | ASGI server running the FastAPI backend |
| `matplotlib` & `seaborn` | Latest | Static publication-grade visualizations, EDA plots, and confusion matrices |
| `plotly` | Latest | Interactive charts for model comparison and analytics dashboards |
| `pytest` | `>=8.3.0` | Automated test suite for data validation, preprocessing, and inference |
| `ruff` | `>=0.8.0` | Lightning-fast Python linter and code formatter adhering to PEP 8 |

---

## 3. Directory Layout & Standards

```text
Dog_Breeds_Classification/
│
├── dog_photos/                  # Sample & user uploads
│   ├── samples/                 # Verified test images from 5 distinct breeds
│   └── uploaded/                # Runtime temporary uploads from web users
│
├── log/                         # System and telemetry logs
│   ├── tensorboard/             # TensorBoard event logs (loss, accuracy curves)
│   ├── training/                # Training run history logs (JSON/CSV)
│   └── application/             # Application execution & error logs
│
├── model/                       # Serialized models and weights
│   ├── checkpoints/             # Best-epoch model weights (.keras / .weights.h5)
│   ├── saved_models/            # Final production export models
│   ├── metadata/                # Training run hyperparameter logs
│   └── class_mapping/           # class_indices.json (0-119 breed name map)
│
├── templates/                   # Frontend user interface
│   ├── base.html                # Master layout with navbar, theme, footer
│   ├── index.html               # Home page with drag-and-drop image upload
│   ├── prediction.html          # Prediction results with top-3 bars & Grad-CAM
│   ├── dashboard.html           # Live model metrics and dataset statistics
│   ├── analytics.html           # Class distribution and performance charts
│   ├── model_info.html          # Architecture breakdown and parameter specs
│   └── about.html               # Technical documentation & project backstory
│
├── dog-breed-120/               # Original Stanford Dogs dataset
│   └── Dogs class/              # 120 breed folders (n[0-9]{8}-<Breed>)
│
├── src/                         # Modular Python package
│   ├── data/                    # Ingestion, validation, preprocessing, pipeline
│   ├── models/                  # Custom CNN, MobileNetV2, EfficientNetB0
│   ├── training/                # Training loops, callbacks, trackers
│   ├── evaluation/              # Metrics, confusion matrix, error analysis
│   ├── inference/               # Predictor, validation, Grad-CAM
│   └── utils/                   # Config loader, logger, paths, seeding
│
├── outputs/                     # Generated figures and reports
│   └── eda/                     # Class distribution, image dimensions charts
│
├── docs/                        # Technical documentation
├── tests/                       # Automated pytest test suites
├── requirements.txt             # Pinned project dependencies
├── .gitignore                   # Version control rules
└── README.md                    # Project presentation
```

---

## 4. Hardware Constraints & Tuning Guidelines

- **Compute Mode:** Native Windows TensorFlow operates in CPU mode (12 logical cores, 7.7 GB available RAM).
- **Optimization Strategy:**
  1. Use asynchronous pipeline loading (`tf.data.Dataset.prefetch(buffer_size=tf.data.AUTOTUNE)`).
  2. Batch sizes set between **16 and 32** to avoid RAM memory pressure.
  3. Pre-trained transfer learning architectures (MobileNetV2) allow rapid convergence on CPU without requiring days of compute.
