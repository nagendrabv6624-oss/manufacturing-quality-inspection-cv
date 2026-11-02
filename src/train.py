"""Train a simple CNN for good/defective product image classification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="Root folder containing train/ and validation/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    root = Path(args.data)
    train_dir, val_dir = root / "train", root / "validation"
    for folder in (train_dir / "good", train_dir / "defective",
                   val_dir / "good", val_dir / "defective"):
        if not folder.is_dir() or not any(folder.iterdir()):
            raise FileNotFoundError(
                f"Expected non-empty class folder: {folder}. "
                "See data/README.md for the required dataset layout."
            )

    tf.keras.utils.set_random_seed(42)
    image_size = (args.image_size, args.image_size)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=image_size, batch_size=32,
        label_mode="binary", seed=42
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=image_size, batch_size=32,
        label_mode="binary", shuffle=False
    )
    class_names = train_ds.class_names
    if class_names != ["defective", "good"]:
        raise ValueError(
            "Class folder names must be exactly 'defective' and 'good' "
            "so the label mapping is explicit."
        )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*image_size, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        callbacks=callbacks
    )
    evaluation = model.evaluate(val_ds, return_dict=True, verbose=0)

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    model.save(out / "quality_inspection_cnn.keras")
    (out / "class_names.json").write_text(json.dumps(class_names, indent=2))
    (out / "validation_metrics.json").write_text(json.dumps(
        {key: float(value) for key, value in evaluation.items()}, indent=2
    ))

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "training_history.png", dpi=150)
    print("Class mapping:", dict(enumerate(class_names)))
    print("Validation metrics:", evaluation)


if __name__ == "__main__":
    main()
