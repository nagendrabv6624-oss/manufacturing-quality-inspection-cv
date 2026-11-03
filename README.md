# Manufacturing Quality Inspection — Steel Surface Defect Classification

A reproducible six-class CNN image-classification project for identifying common defects in hot-rolled steel surfaces.

> **Dataset transparency:** This is an independent portfolio benchmark using the public NEU Surface Defect Database (NEU-DET). It is not presented as confidential data or work from a previous employer. The Kaggle page currently lists the dataset licence as unknown; review the source terms before reuse. Dataset images are downloaded at run time and are not redistributed in this repository.

## Problem

Classify a steel-surface inspection image into one of six defect categories:
- Crazing
- Inclusion
- Patches
- Pitted surface
- Rolled-in scale
- Scratches

## Dataset

- 1,800 grayscale images, 200×200 pixels
- 300 images per defect class
- Original dataset from Northeastern University
- Public mirror: [NEU Surface Defect Database on Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
- The project downloads the public dataset with KaggleHub when training starts.

The dataset is a controlled benchmark; it does not cover every factory camera, material, lighting condition or production line.

## Methodology

1. Download and index the six labelled image classes.
2. Create a reproducible stratified 70/15/15 train/validation/test split.
3. Resize grayscale images and normalize pixel values.
4. Train a CNN with convolution, batch normalization, dropout and light augmentation.
5. Use early stopping and learning-rate reduction.
6. Evaluate once on the held-out test set using accuracy, balanced accuracy, macro precision/recall/F1, weighted F1 and a confusion matrix.

## Run

```bash
pip install -r requirements.txt
python src/train.py
```

Optional settings:

```bash
python src/train.py --epochs 12 --image-size 128 --batch-size 32
```

Predict a single image or a directory:

```bash
python src/predict.py --input path/to/inspection_images/
```

## Results

GitHub Actions downloads the public dataset, trains the CNN on a hosted runner and commits the test metrics, plots, predictions and model artifact into `outputs/`. Results are valid only after a successful run; see `outputs/metrics.json`.

## Outputs

- `outputs/metrics.json` — dataset provenance, split, training and test metrics
- `outputs/quality_inspection_cnn.keras` — trained CNN
- `outputs/class_names.json`
- `outputs/test_predictions.csv`
- `outputs/confusion_matrix.png`
- `outputs/training_history.png`
- `outputs/per_class_f1.png`

## Limitations

- NEU-DET is a relatively small dataset collected under a controlled setup.
- The Kaggle listing reports no explicit licence; check source terms and do not redistribute the images.
- Random image splitting does not establish cross-factory, cross-camera or future-batch generalization.
- This model performs image-level six-class classification. It does not draw defect bounding boxes or segment defect pixels.
- Do not use the benchmark as a safety-critical inspection system without plant-specific validation.

## Skills

Python, TensorFlow/Keras, CNNs, image preprocessing, augmentation, multi-class classification, stratified splitting and model evaluation.
