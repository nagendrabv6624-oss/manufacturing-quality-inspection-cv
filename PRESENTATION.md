# Project 3 — Manufacturing Quality Inspection

## Problem
Detect unusual or defective manufactured parts from inspection images when defect examples are limited.

## Dataset
MVTec AD: more than 5,000 high-resolution images across 15 object and texture categories, including normal training images, anomalous test images and pixel-level defect masks.

Source: Official MVTec AD benchmark. Licence: CC BY-NC-SA 4.0 (non-commercial).

## Model
Convolutional autoencoder:
- Encoder compresses the input image into a compact representation.
- Decoder reconstructs the image.
- Mean absolute reconstruction error is used as an anomaly score.
- Pixel-level error maps provide a basic defect-localization signal.

## Evaluation
- Image-level: precision, recall, F1, ROC-AUC, average precision and confusion matrix.
- Pixel-level: precision, recall, F1, ROC-AUC and average precision against ground-truth masks.
- Thresholds are calibrated using held-out normal training images only.

## Current status
The code and methodology are ready in the repository. Dataset images have not been supplied or trained in this workspace; therefore no model accuracy or trained weights are claimed. A selected MVTec category is required to produce genuine evaluation results.

## Important limitation
MVTec AD is a benchmark and is non-commercial under its licence. Performance on a benchmark does not prove readiness for a real production line.
