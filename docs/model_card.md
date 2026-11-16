# CanineVision AI — Model Card

## Model Details
- **Developer:** AI Engineering Team
- **Model Date:** November 2026
- **Model Version:** v1.0.0
- **Model Type:** Deep Convolutional Neural Network (Transfer Learning via MobileNetV2)
- **License:** MIT

## Intended Use
- **Primary Use Case:** Automated classification of domestic dog breeds from visual photographs.
- **Intended Users:** Veterinarians, animal shelter intake teams, canine welfare organizations, and mobile application developers.
- **Out of Scope:** Identifying non-canine domestic animals, identifying wild animals without warning, or certifying purebred show compliance.

## Training Data
- **Dataset:** Stanford Dog Breeds Dataset subset (120 categories, 20,580 images).
- **Split:** 70% Training (14,406 images), 15% Validation (3,087 images), 15% Test (3,087 images).
- **Pre-processing:** Resized to 224x224, normalized to [-1.0, 1.0].

## Performance Metrics
- **Top-1 Categorical Accuracy:** 84.2%
- **Top-3 Categorical Accuracy:** 96.1%
- **Macro F1-Score:** 0.835
- **Weighted F1-Score:** 0.841
- **Average Inference Latency:** 48 ms (CPU)

## Limitations & Ethical Considerations
- **Confidence Calibration:** Softmax probabilities are not inherently calibrated. Inputs scoring below 0.20 trigger an Out-of-Distribution warning.
- **Visual Similarity:** Closely related breeds (e.g. Siberian Husky vs Alaskan Malamute) may feature high cross-entropy confusion.
