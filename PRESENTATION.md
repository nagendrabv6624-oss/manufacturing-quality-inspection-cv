# Project 3 — Manufacturing Quality Inspection CV

## Industrial problem
Classify hot-rolled steel surface images into common defect categories to support a first-pass visual quality-inspection workflow.

## Public benchmark
NEU Surface Defect Database (NEU-DET): 1,800 grayscale 200×200 images across six classes, 300 per class. Original dataset from Northeastern University; accessed through a public Kaggle mirror. The Kaggle listing currently reports no explicit licence. This is an independent benchmark, not previous-employer data.

Source: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

## Defect classes
Crazing, inclusion, patches, pitted surface, rolled-in scale and scratches.

## Model
A CNN with convolution blocks, batch normalization, pooling, augmentation, dropout and softmax classification.

## Evaluation
Stratified 70/15/15 train/validation/test split. Report held-out accuracy, balanced accuracy, macro precision/recall/F1, weighted F1 and confusion matrix. Per-class F1 and training curves are saved.

## Reproducible results
GitHub Actions downloads the dataset, trains the CNN and commits metrics, plots, test predictions and model weights. Only cite results visible in outputs/metrics.json after a successful run.

## Limitations
The dataset is small and controlled; no cross-factory/camera validation is claimed. The classifier predicts defect category but does not localize defects. The dataset licence must be checked before reuse; images are not redistributed here.
