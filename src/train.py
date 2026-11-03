"""Unsupervised CNN autoencoder for MVTec AD industrial anomaly detection.

Train using only defect-free training images. Calibrate thresholds on a
held-out subset of normal training images. Evaluate on the official test
folder containing good and defective images.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path):
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_image(path, image_size):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size], antialias=True)
    return tf.cast(image, tf.float32) / 255.0


def make_dataset(paths, image_size, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices([str(p) for p in paths])
    if shuffle:
        ds = ds.shuffle(min(len(paths), 1000), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(lambda p: load_image(p, image_size),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_autoencoder(image_size):
    inputs = tf.keras.layers.Input((image_size, image_size, 3))
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    outputs = tf.keras.layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="industrial_anomaly_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
    return model


def reconstruction_scores(model, paths, image_size, batch_size):
    ds = make_dataset(paths, image_size, batch_size, shuffle=False)
    maps = []
    for batch in ds:
        reconstructed = model.predict(batch, verbose=0)
        maps.append(np.mean(np.abs(batch.numpy() - reconstructed), axis=-1))
    maps = np.concatenate(maps, axis=0)
    image_scores = maps.mean(axis=(1, 2))
    return image_scores, maps


def read_mask(mask_path, size):
    if mask_path is None or not mask_path.exists():
        raise FileNotFoundError(f"Missing ground-truth mask: {mask_path}")
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((size, size), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="Root folder containing the extracted MVTec AD category")
    parser.add_argument("--category", default="bottle", help="MVTec AD category, e.g. bottle, cable, capsule")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)
    category_root = Path(args.data) / args.category
    train_good = list_images(category_root / "train" / "good")
    if len(train_good) < 20:
        raise FileNotFoundError(
            f"Expected at least 20 normal training images in {category_root / 'train' / 'good'}. "
            "See data/README.md for MVTec AD folder setup."
        )
    train_paths, validation_paths = train_test_split(
        train_good, test_size=0.15, random_state=42
    )
    test_root = category_root / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(f"Missing test folder: {test_root}")

    test_items = []
    for class_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        label = 0 if class_dir.name == "good" else 1
        for image_path in list_images(class_dir):
            mask_path = None if label == 0 else (
                category_root / "ground_truth" / class_dir.name /
                f"{image_path.stem}_mask.png"
            )
            test_items.append((image_path, label, mask_path))
    if not test_items or len({label for _, label, _ in test_items}) != 2:
        raise ValueError("Test folder must contain both good images and at least one defect class.")

    model = build_autoencoder(args.image_size)
    train_ds = make_dataset(train_paths, args.image_size, args.batch_size, shuffle=True)
    validation_ds = make_dataset(validation_paths, args.image_size, args.batch_size)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
    ]
    history = model.fit(
        train_ds, validation_data=validation_ds, epochs=args.epochs,
        callbacks=callbacks, verbose=1,
    )

    val_scores, val_maps = reconstruction_scores(
        model, validation_paths, args.image_size, args.batch_size
    )
    image_threshold = float(np.quantile(val_scores, 0.99))
    pixel_threshold = float(np.quantile(val_maps.reshape(-1), 0.995))

    test_paths = [item[0] for item in test_items]
    test_labels = np.asarray([item[1] for item in test_items], dtype=np.uint8)
    test_scores, test_maps = reconstruction_scores(
        model, test_paths, args.image_size, args.batch_size
    )
    test_predictions = (test_scores > image_threshold).astype(np.uint8)

    image_metrics = {
        "image_threshold": image_threshold,
        "accuracy": float(accuracy_score(test_labels, test_predictions)),
        "precision": float(precision_score(test_labels, test_predictions, zero_division=0)),
        "recall": float(recall_score(test_labels, test_predictions, zero_division=0)),
        "f1": float(f1_score(test_labels, test_predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(test_labels, test_scores)),
        "average_precision": float(average_precision_score(test_labels, test_scores)),
        "confusion_matrix": confusion_matrix(test_labels, test_predictions).tolist(),
    }

    pixel_scores, pixel_truth = [], []
    pixel_size = 64
    for (image_path, label, mask_path), anomaly_map in zip(test_items, test_maps):
        resized_map = Image.fromarray(anomaly_map.astype(np.float32)).resize(
            (pixel_size, pixel_size), Image.Resampling.BILINEAR
        )
        pixel_scores.append(np.asarray(resized_map, dtype=np.float32).reshape(-1))
        if label == 0:
            truth = np.zeros((pixel_size, pixel_size), dtype=np.uint8)
        else:
            truth = read_mask(mask_path, pixel_size)
        pixel_truth.append(truth.reshape(-1))
    flat_scores = np.concatenate(pixel_scores)
    flat_truth = np.concatenate(pixel_truth)
    pixel_predictions = (flat_scores > pixel_threshold).astype(np.uint8)
    pixel_metrics = {
        "pixel_threshold": pixel_threshold,
        "evaluation_resolution": f"{pixel_size}x{pixel_size}",
        "precision": float(precision_score(flat_truth, pixel_predictions, zero_division=0)),
        "recall": float(recall_score(flat_truth, pixel_predictions, zero_division=0)),
        "f1": float(f1_score(flat_truth, pixel_predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(flat_truth, flat_scores)),
        "average_precision": float(average_precision_score(flat_truth, flat_scores)),
    }

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    model.save(out / "quality_inspection_autoencoder.keras")
    (out / "thresholds.json").write_text(json.dumps({
        "category": args.category,
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "threshold_calibration": "99th percentile of held-out normal training image scores; 99.5th percentile of normal validation pixel errors",
    }, indent=2))
    (out / "image_metrics.json").write_text(json.dumps(image_metrics, indent=2))
    (out / "pixel_metrics.json").write_text(json.dumps(pixel_metrics, indent=2))
    (out / "dataset_summary.json").write_text(json.dumps({
        "category": args.category,
        "train_normal_images": len(train_paths),
        "validation_normal_images": len(validation_paths),
        "test_images": len(test_paths),
        "test_good_images": int((test_labels == 0).sum()),
        "test_defective_images": int((test_labels == 1).sum()),
        "image_size": args.image_size,
        "note": "Metrics are for the selected MVTec AD category only.",
    }, indent=2))

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute reconstruction error")
    plt.title("Convolutional autoencoder training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "training_history.png", dpi=160)
    plt.close()

    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_predictions(
        test_labels, test_predictions, display_labels=["Good", "Defect"], cmap="Blues"
    )
    plt.title(f"{args.category} — image-level anomaly detection")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {image_metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Image-level ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "roc_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(test_scores[test_labels == 0], bins=25, alpha=0.7, label="Good")
    plt.hist(test_scores[test_labels == 1], bins=25, alpha=0.7, label="Defect")
    plt.axvline(image_threshold, linestyle="--", label="Normal-only threshold")
    plt.xlabel("Mean reconstruction error")
    plt.ylabel("Image count")
    plt.title("Image anomaly-score distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "anomaly_score_distribution.png", dpi=160)
    plt.close()

    # A compact qualitative example panel for the presentation.
    defect_index = next(i for i, item in enumerate(test_items) if item[1] == 1)
    original = Image.open(test_paths[defect_index]).convert("RGB").resize(
        (args.image_size, args.image_size)
    )
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Test image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(test_maps[defect_index], cmap="inferno")
    plt.title("Reconstruction error")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(read_mask(test_items[defect_index][2], args.image_size), cmap="gray")
    plt.title("Ground-truth mask")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out / "sample_anomaly_map.png", dpi=160)
    plt.close()

    print(json.dumps({"image_metrics": image_metrics, "pixel_metrics": pixel_metrics}, indent=2))


if __name__ == "__main__":
    main()
