"""
Train a sequence-based model for dynamic ASL signs.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import (
    DYNAMIC_LABELS_PATH,
    DYNAMIC_MODEL_PATH,
    DYNAMIC_SEQUENCE_LENGTH,
    DYNAMIC_TRAINING_REPORT_PATH,
    SEQUENCE_DATASET_DIR,
    ensure_project_directories,
)


def load_dynamic_training_dependencies():
    from tensorflow.keras import callbacks, layers, models, utils

    return callbacks, layers, models, utils


def sanitize_label(label: str) -> str:
    return label.strip().replace("_", " ")


def pad_or_trim_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    if sequence.ndim != 2:
        raise ValueError(f"Expected sequence shape [frames, features], received {sequence.shape}")

    frame_count, feature_count = sequence.shape
    if feature_count != 63:
        raise ValueError(f"Expected 63 features per frame, received {feature_count}")

    if frame_count > target_length:
        return sequence[:target_length]
    if frame_count < target_length:
        padding = np.zeros((target_length - frame_count, feature_count), dtype=np.float32)
        return np.vstack([sequence, padding])
    return sequence.astype(np.float32)


def load_sequence_dataset(sequence_root: Path, target_length: int) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    label_names: list[str] = []
    samples_per_label: dict[str, int] = {}

    label_directories = sorted([path for path in sequence_root.iterdir() if path.is_dir()])
    if not label_directories:
        raise FileNotFoundError(f"No sequence directories found in {sequence_root}. Collect dynamic data first.")

    for label_index, label_dir in enumerate(label_directories):
        sequence_files = sorted(label_dir.glob("*.npy"))
        if not sequence_files:
            continue

        label_name = sanitize_label(label_dir.name)
        label_names.append(label_name)
        samples_per_label[label_name] = len(sequence_files)

        for sequence_file in sequence_files:
            sequence_array = np.load(sequence_file)
            sequence_array = pad_or_trim_sequence(sequence_array, target_length)
            sequences.append(sequence_array)
            labels.append(label_index)

    if not sequences:
        raise ValueError("No valid dynamic sequence files were found.")

    x_data = np.array(sequences, dtype=np.float32)
    y_data = np.array(labels, dtype=np.int32)
    return x_data, y_data, label_names, samples_per_label


def build_dynamic_model(sequence_length: int, feature_count: int, class_count: int):
    _, layers, models, _ = load_dynamic_training_dependencies()

    model = models.Sequential(
        [
            layers.Input(shape=(sequence_length, feature_count)),
            layers.Masking(mask_value=0.0),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.25),
            layers.Bidirectional(layers.GRU(64)),
            layers.Dropout(0.25),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def split_sequence_dataset_by_class(
    x_data: np.ndarray,
    y_data: np.ndarray,
    label_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split each class independently into 70/15/15.
    This avoids the second stratified split failure that happens with very small classes.
    """
    rng = np.random.default_rng(42)

    x_train_parts: list[np.ndarray] = []
    x_validation_parts: list[np.ndarray] = []
    x_test_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []
    y_validation_parts: list[np.ndarray] = []
    y_test_parts: list[np.ndarray] = []

    for class_index, class_name in enumerate(label_names):
        class_indices = np.where(y_data == class_index)[0]
        sample_count = len(class_indices)

        if sample_count < 3:
            raise ValueError(
                f"Dynamic sign '{class_name}' has only {sample_count} sequence(s). "
                "At least 3 sequences are required, and 10+ are recommended."
            )

        shuffled_indices = rng.permutation(class_indices)
        train_count = max(1, int(round(sample_count * 0.70)))
        validation_count = max(1, int(round(sample_count * 0.15)))
        test_count = sample_count - train_count - validation_count

        if test_count < 1:
            test_count = 1
            if train_count > validation_count and train_count > 1:
                train_count -= 1
            elif validation_count > 1:
                validation_count -= 1

        if train_count < 1 or validation_count < 1 or test_count < 1:
            raise ValueError(
                f"Could not split dynamic sign '{class_name}' into train/validation/test. "
                "Collect more sequences for this sign."
            )

        train_end = train_count
        validation_end = train_count + validation_count

        train_indices = shuffled_indices[:train_end]
        validation_indices = shuffled_indices[train_end:validation_end]
        test_indices = shuffled_indices[validation_end:]

        x_train_parts.append(x_data[train_indices])
        x_validation_parts.append(x_data[validation_indices])
        x_test_parts.append(x_data[test_indices])
        y_train_parts.append(y_data[train_indices])
        y_validation_parts.append(y_data[validation_indices])
        y_test_parts.append(y_data[test_indices])

    x_train = np.concatenate(x_train_parts, axis=0)
    x_validation = np.concatenate(x_validation_parts, axis=0)
    x_test = np.concatenate(x_test_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    y_validation = np.concatenate(y_validation_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def train_sequence_model() -> None:
    ensure_project_directories()
    callbacks, _, _, _ = load_dynamic_training_dependencies()

    x_data, y_data, label_names, samples_per_label = load_sequence_dataset(
        SEQUENCE_DATASET_DIR,
        DYNAMIC_SEQUENCE_LENGTH,
    )

    x_train, x_validation, x_test, y_train, y_validation, y_test = split_sequence_dataset_by_class(
        x_data,
        y_data,
        label_names,
    )

    model = build_dynamic_model(DYNAMIC_SEQUENCE_LENGTH, x_data.shape[2], len(label_names))

    training_callbacks = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=40,
        batch_size=16,
        callbacks=training_callbacks,
        verbose=1,
    )

    train_predictions = np.argmax(model.predict(x_train, verbose=0), axis=1)
    validation_predictions = np.argmax(model.predict(x_validation, verbose=0), axis=1)
    test_probabilities = model.predict(x_test, verbose=0)
    test_predictions = np.argmax(test_probabilities, axis=1)

    train_accuracy = accuracy_score(y_train, train_predictions)
    validation_accuracy = accuracy_score(y_validation, validation_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    report = classification_report(
        y_test,
        test_predictions,
        target_names=label_names,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, test_predictions)
    matrix_df = pd.DataFrame(matrix, index=label_names, columns=label_names)

    DYNAMIC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(DYNAMIC_MODEL_PATH)
    DYNAMIC_LABELS_PATH.write_text(json.dumps(label_names, ensure_ascii=False, indent=2), encoding="utf-8")

    warning_text = ""
    if test_accuracy >= 1.0 or validation_accuracy >= 1.0:
        warning_text = "Warning: Possible overfitting or data leakage detected.\n\n"

    report_text = (
        "Dynamic ASL Sequence Model Training Report\n"
        "=========================================\n"
        f"Sequence length: {DYNAMIC_SEQUENCE_LENGTH}\n"
        f"Samples used: {len(x_data)}\n"
        f"Training samples: {len(x_train)}\n"
        f"Validation samples: {len(x_validation)}\n"
        f"Test samples: {len(x_test)}\n"
        f"Samples per label: {samples_per_label}\n\n"
        f"Train accuracy: {train_accuracy:.4f}\n"
        f"Validation accuracy: {validation_accuracy:.4f}\n"
        f"Test accuracy: {test_accuracy:.4f}\n\n"
        f"{warning_text}"
        "Confusion matrix:\n"
        f"{matrix_df.to_string()}\n\n"
        "Classification report:\n"
        f"{report}\n"
    )
    DYNAMIC_TRAINING_REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(f"Dynamic model saved to: {DYNAMIC_MODEL_PATH}")
    print(f"Dynamic labels saved to: {DYNAMIC_LABELS_PATH}")
    print(f"Dynamic training report saved to: {DYNAMIC_TRAINING_REPORT_PATH}")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    if warning_text:
        print("Warning: Possible overfitting or data leakage detected.")
    print("\nConfusion matrix:")
    print(matrix_df.to_string())
    print("\nClassification report:")
    print(report)
    print(f"Best validation accuracy: {max(history.history.get('val_accuracy', [0.0])):.4f}")


def main() -> None:
    try:
        train_sequence_model()
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
