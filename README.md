# Manufacturing Quality Inspection Using Computer Vision

An image-classification starter project for distinguishing acceptable and defective manufactured parts using a convolutional neural network (CNN).

> **Data transparency:** No factory images or trained model are included. This project requires a labelled image dataset supplied by the user. It does not claim validated defect-detection performance.

## Dataset layout
Organize your labelled images into folders as follows:

```text
data/
  train/
    good/
    defective/
  validation/
    good/
    defective/
```

Use images you own or are licensed to use. The folder names are the class labels. Keep validation images separate from training images; do not place duplicate or near-duplicate images in both sets. For a dataset such as MVTec AD, first choose a specific object category and document how normal and defect images were mapped into these folders. Dataset structure and labels may require preprocessing.

## Install and train
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --data data
```

The script expects `data/train` and `data/validation` with the two class folders above.

## Outputs
The script saves `outputs/quality_inspection_cnn.keras`, `outputs/class_names.json` and `outputs/training_history.png`.

## Evaluation and limitations
This starter reports validation accuracy and loss. For a serious inspection system, add a held-out test set, per-class precision/recall/F1, a confusion matrix, inspection of false negatives, and (for localization) a segmentation or anomaly-localization approach. Image classification alone does not identify the defect's location. Check performance across lighting, camera, part orientation and production batches.

## Skills
Python, TensorFlow/Keras, CNNs, image preprocessing, binary classification and model evaluation.
