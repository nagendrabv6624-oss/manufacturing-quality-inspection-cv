"""Score one image or a folder using a trained anomaly autoencoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image path or folder of images")
    parser.add_argument("--output", default="outputs/predictions.csv")
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    model_path = ROOT / "outputs" / "quality_inspection_autoencoder.keras"
    threshold_path = ROOT / "outputs" / "thresholds.json"
    if not model_path.exists() or not threshold_path.exists():
        raise FileNotFoundError("Train the model first with python src/train.py.")
    source = Path(args.input)
    paths = ([source] if source.is_file() else sorted(
        p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS
    ))
    if not paths:
        raise ValueError("No supported images found.")

    model = tf.keras.models.load_model(model_path)
    threshold = json.loads(threshold_path.read_text())["image_threshold"]
    rows = []
    for path in paths:
        raw = tf.io.read_file(str(path))
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, [args.image_size, args.image_size])
        image = tf.cast(image, tf.float32) / 255.0
        batch = tf.expand_dims(image, 0)
        reconstructed = model.predict(batch, verbose=0)[0]
        score = float(tf.reduce_mean(tf.abs(image - reconstructed)).numpy())
        rows.append({
            "image": str(path),
            "anomaly_score": score,
            "threshold": threshold,
            "prediction": "defect" if score > threshold else "good",
        })

    result = pd.DataFrame(rows)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Saved predictions to {output}")


if __name__ == "__main__":
    main()
