# Training Guide – Dog Breeds Classification
**Project Name:** Dog Breeds Classification  
**Project ID:** PRAICP-1007-120  

This guide provides end-to-end instructions for configuring, training, monitoring, and checkpointing deep learning models for the 120-class dog breeds dataset.

---

## 1. Prerequisites & Environment Setup

Ensure the virtual environment is activated and dependencies are installed:

```powershell
# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\activate

# Verify TensorFlow environment
python -c "import tensorflow as tf; print('TensorFlow Version:', tf.__version__)"
```

---

## 2. Dataset Pipeline & Verification

Before initiating model training, ensure the dataset metadata and stratified splits exist:

```powershell
# 1. Byte-verify raw dataset and build labels.csv
python src/data/dataset_validator.py

# 2. Generate 70/15/15 stratified splits (Zero Data Leakage)
python src/data/data_split.py
```

This generates `data/splits/train.csv` (14,406 images), `val.csv` (3,087 images), and `test.csv` (3,087 images).

---

## 3. Training Execution

The training orchestrator (`src/training/trainer.py`) manages model instantiation, data loaders, callback pipelines, and metric logging.

### Basic Training Command (MobileNetV2 Transfer Learning)

```powershell
python -m src.training.trainer --model mobilenet_v2 --epochs 10 --batch-size 32 --lr 0.001
```

### Supported Command-Line Arguments

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--model` | `str` | `mobilenet_v2` | Model architecture (`mobilenet_v2`, `efficientnet_b0`, `custom_cnn`) |
| `--epochs` | `int` | `10` | Maximum training epochs |
| `--batch-size` | `int` | `32` | Training and validation batch size |
| `--lr` | `float` | `0.001` | Initial learning rate for Adam optimizer |
| `--image-size` | `int` | `224` | Image width and height in pixels ($224 \times 224$) |
| `--checkpoint-dir` | `str` | `model/checkpoints` | Directory to save best model weights |
| `--saved-model-dir`| `str` | `model/saved_models`| Directory for full `.keras` model persistence |

---

## 4. Callback Pipeline & Regularization

The training engine integrates four automated callbacks (`src/training/callbacks.py`):

1. **`ModelCheckpoint`**: Monitors validation loss and persists the best weights to `model/checkpoints/{model_name}_best.weights.h5`.
2. **`EarlyStopping`**: Halts training if validation loss does not improve for 5 consecutive epochs (`patience=5`), restoring best weights automatically.
3. **`ReduceLROnPlateau`**: Reduces the learning rate by a factor of 0.5 if validation loss stagnates for 2 epochs (`factor=0.5, patience=2, min_lr=1e-6`).
4. **`TensorBoard`**: Logs epoch metrics and weight histograms to `log/tensorboard/`.

---

## 5. Live Monitoring with TensorBoard

Monitor loss and accuracy curves in real time during training:

```powershell
tensorboard --logdir log/tensorboard --port 6006
```

Open your browser to `http://localhost:6006` to inspect:
- `epoch_loss` vs `epoch_val_loss`
- `epoch_accuracy` vs `epoch_val_accuracy`
- `epoch_top_3_accuracy` vs `epoch_val_top_3_accuracy`
- Dynamic learning rate decay curves

---

## 6. Post-Training Evaluation

Evaluate the saved model on the strict holdout test partition ($3,087$ images):

```powershell
python scripts/evaluate.py --model-path model/saved_models/best_model.keras --samples 500
```

This generates:
- Top-1 and Top-3 accuracy metrics
- Macro-averaged precision, recall, and F1-score
- Normalized 120-class confusion matrix (`outputs/evaluation/confusion_matrix_normalized.png`)
- High-volume Top-20 focus matrix (`outputs/evaluation/confusion_matrix_top20.png`)
- Misclassification error analysis (`outputs/evaluation/error_analysis.json`)

---

## 7. Model Checkpoint Resumption & Fine-Tuning

To resume training or initiate Stage 2 unfreezing (fine-tuning top layers of MobileNetV2):

```python
import tensorflow as tf
from src.models.mobilenet_model import build_mobilenet_v2

# 1. Instantiate model architecture
model = build_mobilenet_v2(input_shape=(224, 224, 3), num_classes=120, trainable_backbone=False)

# 2. Load Stage 1 best weights
model.load_weights("model/checkpoints/mobilenet_v2_best.weights.h5")

# 3. Unfreeze last 20 layers for fine-tuning
for layer in model.layers[-20:]:
    layer.trainable = True

# 4. Recompile with lower learning rate (1e-5)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")],
)
```
