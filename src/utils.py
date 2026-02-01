"""
Reusable utility functions for hand landmark extraction, drawing, saving, and loading.
"""

from __future__ import annotations

import re
from pathlib import Path
import sys
from typing import List, Optional, Sequence, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.runtime_setup import suppress_runtime_warnings

suppress_runtime_warnings()

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from src.config import (
    CAMERA_BACKEND,
    CAMERA_FRAME_HEIGHT,
    CAMERA_FRAME_WIDTH,
    CAMERA_INDEX,
    CAMERA_WARMUP_FRAMES,
    LABEL_ENCODER_PATH,
    LANDMARK_DIMENSIONS,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODEL_PATH,
    NUM_FEATURES,
    NUM_LANDMARKS,
)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def initialize_hands(
    static_image_mode: bool = False,
    max_num_hands: int = MAX_NUM_HANDS,
    min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
):
    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def open_camera(camera_index: int = CAMERA_INDEX):
    if CAMERA_BACKEND == "dshow" and hasattr(cv2, "CAP_DSHOW"):
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        return capture

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(CAMERA_WARMUP_FRAMES):
        capture.read()

    return capture


def extract_hand_landmarks(frame: np.ndarray, hands) -> Tuple[Optional[List[float]], Optional[object]]:
    """
    Returns one hand's landmarks as a flat list of 63 values and the MediaPipe result.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if not results.multi_hand_landmarks:
        return None, results

    hand_landmarks = results.multi_hand_landmarks[0]
    flattened_landmarks: List[float] = []
    for landmark in hand_landmarks.landmark:
        flattened_landmarks.extend([landmark.x, landmark.y, landmark.z])

    if len(flattened_landmarks) != NUM_FEATURES:
        return None, results

    return flattened_landmarks, results


def normalize_landmarks(landmarks: Sequence[float]) -> List[float]:
    """
    Makes the landmarks relative to the wrist and scales them to a stable range.
    """
    if len(landmarks) != NUM_FEATURES:
        raise ValueError(f"Expected {NUM_FEATURES} values, received {len(landmarks)}")

    points = np.array(landmarks, dtype=np.float32).reshape(NUM_LANDMARKS, LANDMARK_DIMENSIONS)
    wrist = points[0].copy()
    points = points - wrist

    max_value = np.max(np.abs(points))
    if max_value > 0:
        points = points / max_value

    return points.flatten().astype(float).tolist()


def draw_hand_landmarks(frame: np.ndarray, results) -> np.ndarray:
    """
    Draws hand landmarks on the frame when a hand is detected.
    """
    if not results or not results.multi_hand_landmarks:
        return frame

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
    return frame


def build_feature_column_names() -> List[str]:
    columns = []
    for landmark_index in range(NUM_LANDMARKS):
        columns.append(f"x_{landmark_index}")
        columns.append(f"y_{landmark_index}")
        columns.append(f"z_{landmark_index}")
    return columns


def derive_session_id_from_filename(dataset_file: Path) -> str:
    stem = dataset_file.stem
    return re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_") or "unknown_session"


def save_sample_data(
    dataset_file: Path,
    label: str,
    session_id: str,
    sample_index: int,
    normalized_landmarks: Sequence[float],
) -> None:
    """
    Appends one sample to the class CSV file.
    """
    if len(normalized_landmarks) != NUM_FEATURES:
        raise ValueError("Invalid sample length. The sample was not saved.")

    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "label": label,
        "session_id": session_id,
        "sample_index": int(sample_index),
    }
    for column_name, value in zip(build_feature_column_names(), normalized_landmarks):
        row[column_name] = value

    dataframe = pd.DataFrame([row])
    write_header = not dataset_file.exists()
    dataframe.to_csv(dataset_file, mode="a", index=False, header=write_header)


def load_model(model_path: Path = MODEL_PATH, label_encoder_path: Path = LABEL_ENCODER_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")

    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    return model, label_encoder


def preprocess_input_for_prediction(landmarks: Sequence[float]) -> np.ndarray:
    normalized = normalize_landmarks(landmarks)
    feature_names = build_feature_column_names()
    return pd.DataFrame([normalized], columns=feature_names, dtype=np.float32)


def merge_raw_dataset_files(raw_dataset_dir: Path) -> pd.DataFrame:
    """
    Loads every class CSV from dataset/raw and returns one merged dataset.
    """
    csv_files = sorted(raw_dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV dataset files were found in {raw_dataset_dir}. Collect data first."
        )

    dataframes = []
    for csv_file in csv_files:
        try:
            dataframe = pd.read_csv(csv_file)
        except Exception as error:
            print(f"Skipping unreadable file {csv_file.name}: {error}")
            continue

        if dataframe.empty:
            print(f"Skipping empty file {csv_file.name}")
            continue
        dataframe = dataframe.copy()
        if "session_id" not in dataframe.columns:
            dataframe["session_id"] = derive_session_id_from_filename(csv_file)
        dataframe["session_id"] = (
            dataframe["session_id"]
            .fillna(derive_session_id_from_filename(csv_file))
            .astype(str)
            .str.strip()
        )
        dataframe.loc[dataframe["session_id"] == "", "session_id"] = derive_session_id_from_filename(csv_file)
        if "sample_index" not in dataframe.columns:
            dataframe["sample_index"] = range(len(dataframe))
        else:
            dataframe["sample_index"] = pd.to_numeric(dataframe["sample_index"], errors="coerce")
            missing_mask = dataframe["sample_index"].isna()
            if missing_mask.any():
                dataframe.loc[missing_mask, "sample_index"] = list(range(len(dataframe)))
            dataframe["sample_index"] = dataframe["sample_index"].astype(int)
        dataframe["source_file"] = csv_file.name
        dataframes.append(dataframe)

    if not dataframes:
        raise ValueError("No valid dataset files were found.")

    merged = pd.concat(dataframes, ignore_index=True)
    return merged
