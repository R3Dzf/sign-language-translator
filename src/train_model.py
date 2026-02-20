"""
Train a classical machine learning model on collected ASL landmark data.
Session-based splitting is used to avoid leakage between train/validation/test.
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from src.config import (
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    NUM_FEATURES,
    PROCESSED_DATASET_FILE,
    RAW_DATASET_DIR,
    TRAINING_REPORT_PATH,
    ensure_project_directories,
)
from src.utils import build_feature_column_names, merge_raw_dataset_files


METADATA_COLUMNS = {"label", "session_id", "sample_index", "source_file"}
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
def clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"label", "session_id", "sample_index"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"The dataset is missing required columns: {missing_text}")

    feature_columns = build_feature_column_names()
    missing_feature_columns = [column for column in feature_columns if column not in dataframe.columns]
    if missing_feature_columns:
        raise ValueError(
            "The dataset is missing feature columns. "
            f"Expected {NUM_FEATURES} features but found {NUM_FEATURES - len(missing_feature_columns)}."
        )

    cleaned = dataframe.dropna(subset=["label", "session_id"]).copy()
    cleaned["label"] = cleaned["label"].astype(str).str.strip()
    cleaned["session_id"] = cleaned["session_id"].astype(str).str.strip()
    cleaned["sample_index"] = pd.to_numeric(cleaned["sample_index"], errors="coerce")
    cleaned = cleaned.dropna(subset=["sample_index"]).copy()
    cleaned["sample_index"] = cleaned["sample_index"].astype(int)
    cleaned[feature_columns] = cleaned[feature_columns].apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.dropna(subset=feature_columns).copy()
    return cleaned


def summarize_samples_per_class(dataframe: pd.DataFrame) -> dict[str, int]:
    return dataframe.groupby("label").size().sort_index().astype(int).to_dict()


def summarize_sessions_per_class(dataframe: pd.DataFrame) -> dict[str, int]:
    return dataframe.groupby("label")["session_id"].nunique().sort_index().astype(int).to_dict()


def summarize_session_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
    session_labels = dataframe.groupby("session_id")["label"].nunique()
    mixed_sessions = session_labels[session_labels > 1]
    if not mixed_sessions.empty:
        mixed_list = ", ".join(mixed_sessions.index.tolist())
        raise ValueError(
            "A static recording session must contain only one label. "
            f"Mixed sessions found: {mixed_list}"
        )

    return (
        dataframe.groupby("session_id", as_index=False)
        .agg(label=("label", "first"))
        .sort_values(["label", "session_id"])
        .reset_index(drop=True)
    )


def ensure_minimum_sessions_per_class(session_table: pd.DataFrame) -> None:
    sessions_per_class = session_table.groupby("label").size().sort_index()
    weak_labels = sessions_per_class[sessions_per_class < 3]
    if not weak_labels.empty:
        details = ", ".join(f"{label}={count}" for label, count in weak_labels.items())
        raise ValueError(
            "Session-based train/validation/test evaluation requires at least 3 recording sessions "
            f"per class. Add more sessions for: {details}"
        )


def split_sessions_by_group(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    session_table = summarize_session_labels(dataframe)
    train_session_ids: list[str] = []
    validation_session_ids: list[str] = []
    test_session_ids: list[str] = []

    for label in sorted(session_table["label"].unique().tolist()):
        label_sessions = (
            session_table.loc[session_table["label"] == label, "session_id"]
            .sample(frac=1.0, random_state=42)
            .tolist()
        )
        session_count = len(label_sessions)
        if session_count < 3:
            raise ValueError(
                f"Label '{label}' needs at least 3 sessions for session-based splitting, "
                f"but only {session_count} were found."
            )

        if session_count == 3:
            train_count = 1
            validation_count = 1
        else:
            test_count = max(1, round(session_count * TEST_RATIO))
            validation_count = max(1, round(session_count * VALIDATION_RATIO))
            train_count = session_count - test_count - validation_count

            while train_count < 1 and validation_count > 1:
                validation_count -= 1
                train_count = session_count - test_count - validation_count
            while train_count < 1 and test_count > 1:
                test_count -= 1
                train_count = session_count - test_count - validation_count

            if train_count < 1:
                raise ValueError(
                    f"Could not allocate at least one training session for label '{label}'. "
                    "Record more sessions for this label."
                )

            train_session_ids.extend(label_sessions[:train_count])
            validation_session_ids.extend(label_sessions[train_count:train_count + validation_count])
            test_session_ids.extend(label_sessions[train_count + validation_count:])
            continue

        train_session_ids.append(label_sessions[0])
        validation_session_ids.append(label_sessions[1])
        test_session_ids.append(label_sessions[2])

    train_set = dataframe[dataframe["session_id"].isin(train_session_ids)].copy()
    validation_set = dataframe[dataframe["session_id"].isin(validation_session_ids)].copy()
    test_set = dataframe[dataframe["session_id"].isin(test_session_ids)].copy()

    return train_set, validation_set, test_set


def verify_no_session_leakage(
    train_sessions: set[str],
    validation_sessions: set[str],
    test_sessions: set[str],
) -> None:
    overlapping_sessions = (
        (train_sessions & validation_sessions)
        | (train_sessions & test_sessions)
        | (validation_sessions & test_sessions)
    )
    if overlapping_sessions:
        overlaps = ", ".join(sorted(overlapping_sessions))
        raise RuntimeError(f"Data leakage detected between splits. Overlapping sessions: {overlaps}")


def format_confusion_matrix(confusion: pd.DataFrame) -> str:
    return confusion.to_string()


def train_model() -> None:
    ensure_project_directories()

    merged_dataset = merge_raw_dataset_files(RAW_DATASET_DIR)
    cleaned_dataset = clean_dataset(merged_dataset)
    cleaned_dataset.to_csv(PROCESSED_DATASET_FILE, index=False)

    if cleaned_dataset.empty:
        raise ValueError("The dataset is empty after cleaning. Collect more valid samples.")

    session_table = summarize_session_labels(cleaned_dataset)
    ensure_minimum_sessions_per_class(session_table)

    train_set, validation_set, test_set = split_sessions_by_group(cleaned_dataset)

    train_sessions = set(train_set["session_id"].unique())
    validation_sessions = set(validation_set["session_id"].unique())
    test_sessions = set(test_set["session_id"].unique())
    verify_no_session_leakage(train_sessions, validation_sessions, test_sessions)

    feature_columns = build_feature_column_names()
    x_train = train_set[feature_columns].copy()
    x_validation = validation_set[feature_columns].copy()
    x_test = test_set[feature_columns].copy()

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_set["label"])
    y_validation = label_encoder.transform(validation_set["label"])
    y_test = label_encoder.transform(test_set["label"])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    train_predictions = model.predict(x_train)
    validation_predictions = model.predict(x_validation)
    test_predictions = model.predict(x_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    validation_accuracy = accuracy_score(y_validation, validation_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    report = classification_report(
        y_test,
        test_predictions,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
    confusion = confusion_matrix(y_test, test_predictions, labels=range(len(label_encoder.classes_)))
    confusion_frame = pd.DataFrame(
        confusion,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    samples_per_class = summarize_samples_per_class(cleaned_dataset)
    sessions_per_class = summarize_sessions_per_class(cleaned_dataset)

    warning_lines: list[str] = []
    if test_accuracy >= 1.0:
        warning_lines.append(
            "Warning: 100% accuracy may still indicate overfitting. Test with a new recording session."
        )

    report_text = (
        "ASL Landmark Model Training Report\n"
        "=================================\n"
        "Evaluation method: session-based split using GroupShuffleSplit\n\n"
        f"Number of classes: {len(label_encoder.classes_)}\n"
        f"Samples used: {len(cleaned_dataset)}\n"
        f"Samples per class: {samples_per_class}\n"
        f"Sessions per class: {sessions_per_class}\n\n"
        f"Train sessions ({len(train_sessions)}): {sorted(train_sessions)}\n"
        f"Validation sessions ({len(validation_sessions)}): {sorted(validation_sessions)}\n"
        f"Test sessions ({len(test_sessions)}): {sorted(test_sessions)}\n\n"
        f"Training samples: {len(train_set)}\n"
        f"Validation samples: {len(validation_set)}\n"
        f"Test samples: {len(test_set)}\n\n"
        f"Train accuracy: {train_accuracy:.4f}\n"
        f"Validation accuracy: {validation_accuracy:.4f}\n"
        f"Test accuracy: {test_accuracy:.4f}\n\n"
        f"{chr(10).join(warning_lines)}\n\n"
        "Confusion matrix:\n"
        f"{format_confusion_matrix(confusion_frame)}\n\n"
        "Classification report:\n"
        f"{report}\n"
    )
    TRAINING_REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(f"Merged dataset saved to: {PROCESSED_DATASET_FILE}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")
    print(f"Training report saved to: {TRAINING_REPORT_PATH}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Sessions per class: {sessions_per_class}")
    print(f"Train sessions: {sorted(train_sessions)}")
    print(f"Validation sessions: {sorted(validation_sessions)}")
    print(f"Test sessions: {sorted(test_sessions)}")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    for warning_line in warning_lines:
        print(warning_line)
    print("\nConfusion matrix:")
    print(confusion_frame)
    print("\nClassification report:")
    print(report)


def main() -> None:
    try:
        train_model()
    except FileNotFoundError as error:
        print(f"Error: {error}")
    except ValueError as error:
        print(f"Error: {error}")
    except Exception as error:
        print(f"Unexpected error: {error}")


if __name__ == "__main__":
    main()
