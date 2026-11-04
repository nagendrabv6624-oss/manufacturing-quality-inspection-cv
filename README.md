# Manufacturing Quality Inspection — Steel Surface Defect Classification

An independently trained six-class steel surface defect classifier using the NEU-DET benchmark.

## Dataset actually used

This run used the NEU-DET archive uploaded for this project:
- 1,800 grayscale images, 200×200 pixels
- 1,440 training images (240 per class)
- 360 supplied validation images (60 per class)
- Six classes: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`
- The archive also includes Pascal VOC XML bounding-box annotations.

The supplied training split was used for fitting. The supplied validation split was stratified into 180 validation images and 180 final test images (30 per class). The final test subset was not used for fitting.

## Trained approach

- Convert images to grayscale and resize to 64×64.
- Extract 1,764 Histogram of Oriented Gradients (HOG) features.
- Standardize features, reduce to 128 dimensions with PCA, then classify with an RBF Support Vector Machine.
- Fixed random seed: 42.

This is an image-level classifier. It does not localize defects using the XML bounding boxes.

## Measured held-out test results

| Metric | Result |
|---|---:|
| Accuracy | **77.78%** |
| Macro precision | 79.59% |
| Macro recall | 77.78% |
| Macro F1 | 77.51% |
| Weighted F1 | 77.51% |

Per-class results and full provenance are in [outputs/metrics.json](outputs/metrics.json). Individual held-out predictions are in [outputs/test_predictions.csv](outputs/test_predictions.csv).

## Reproduce

Expected directory structure:

```
NEU-DET/
  train/images/<class>/*.jpg
  validation/images/<class>/*.jpg
```

Install dependencies and run:

```bash
pip install -r requirements.txt
python src/train.py --data-dir /path/to/NEU-DET
python src/predict.py --input /path/to/image.jpg
```

Training writes the fitted pipeline and metrics to `outputs/`. The dataset images are not included in this repository.

## Dataset provenance and limitations

NEU-DET is a controlled public research benchmark originating from Northeastern University. The Kaggle mirror currently does not specify a clear dataset licence; check the source terms before reuse. This repository does not redistribute the raw images.

The dataset is small and does not establish performance across different mills, cameras, materials, or lighting conditions. The test set is a held-out subset of the supplied validation split, not a separate factory or acquisition campaign. Do not use this benchmark model for safety-critical production decisions without plant-specific validation.
