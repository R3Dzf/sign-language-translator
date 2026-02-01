"""
Central configuration for the Sign Language Translator project.
"""

from __future__ import annotations

import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent
    RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RESOURCE_ROOT = PROJECT_ROOT

ASSETS_DIR = PROJECT_ROOT / "assets"
BUNDLED_ASSETS_DIR = RESOURCE_ROOT / "assets"
DATASET_DIR = PROJECT_ROOT / "dataset"
BUNDLED_DATASET_DIR = RESOURCE_ROOT / "dataset"
RAW_DATASET_DIR = DATASET_DIR / "raw"
PROCESSED_DATASET_DIR = DATASET_DIR / "processed"
PROCESSED_DATASET_FILE = PROCESSED_DATASET_DIR / "asl_landmarks_dataset.csv"
SEQUENCE_DATASET_DIR = DATASET_DIR / "sequences"

MODELS_DIR = PROJECT_ROOT / "models"
BUNDLED_MODELS_DIR = RESOURCE_ROOT / "models"
MODEL_PATH = MODELS_DIR / "sign_model.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
TRAINING_REPORT_PATH = MODELS_DIR / "training_report.txt"
DYNAMIC_MODEL_PATH = MODELS_DIR / "dynamic_sign_model.keras"
DYNAMIC_LABELS_PATH = MODELS_DIR / "dynamic_sign_labels.json"
DYNAMIC_TRAINING_REPORT_PATH = MODELS_DIR / "dynamic_training_report.txt"

if not MODEL_PATH.exists() and (BUNDLED_MODELS_DIR / "sign_model.pkl").exists():
    MODEL_PATH = BUNDLED_MODELS_DIR / "sign_model.pkl"
if not LABEL_ENCODER_PATH.exists() and (BUNDLED_MODELS_DIR / "label_encoder.pkl").exists():
    LABEL_ENCODER_PATH = BUNDLED_MODELS_DIR / "label_encoder.pkl"
if not DYNAMIC_MODEL_PATH.exists() and (BUNDLED_MODELS_DIR / "dynamic_sign_model.keras").exists():
    DYNAMIC_MODEL_PATH = BUNDLED_MODELS_DIR / "dynamic_sign_model.keras"
if not DYNAMIC_LABELS_PATH.exists() and (BUNDLED_MODELS_DIR / "dynamic_sign_labels.json").exists():
    DYNAMIC_LABELS_PATH = BUNDLED_MODELS_DIR / "dynamic_sign_labels.json"

CAMERA_INDEX = 0
CAMERA_BACKEND = "dshow" if sys.platform.startswith("win") else "default"
CAMERA_FRAME_WIDTH = 960
CAMERA_FRAME_HEIGHT = 720
CAMERA_WARMUP_FRAMES = 4
NUM_LANDMARKS = 21
LANDMARK_DIMENSIONS = 3
NUM_FEATURES = NUM_LANDMARKS * LANDMARK_DIMENSIONS

MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 1

CONFIDENCE_THRESHOLD = 0.55
AUTO_ACCEPT_CONFIDENCE_THRESHOLD = 0.80
STABLE_FRAMES_REQUIRED = 8
PREDICTION_HISTORY_SIZE = 12
MAJORITY_VOTE_RATIO = 0.75
AUTO_ACCEPT_COOLDOWN_FRAMES = 18
COLLECTION_PREPARE_DELAY_SECONDS = 3
DYNAMIC_SEQUENCE_LENGTH = 30
DYNAMIC_COLLECTION_COOLDOWN_SECONDS = 1
DYNAMIC_MIN_SEQUENCE_FRAMES = 20
DYNAMIC_CONFIDENCE_THRESHOLD = 0.70
DYNAMIC_PREDICTION_HISTORY_SIZE = 6
DYNAMIC_STABLE_COUNT = 4
DYNAMIC_ACCEPT_COOLDOWN_FRAMES = 24
HYBRID_MOTION_THRESHOLD = 0.045

DEFAULT_SAMPLES_PER_CLASS = 300
SUPPORTED_FILE_EXTENSION = ".csv"

OVERLAY_BOX_X = 18
OVERLAY_BOX_Y = 18
OVERLAY_BOX_WIDTH = 520
OVERLAY_BOX_HEIGHT = 190
OVERLAY_TEXT_X = 34
OVERLAY_TEXT_LINE1_Y = 54
OVERLAY_TEXT_LINE_GAP = 36

CLASS_LABELS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "Hello",
    "Thank you",
    "Yes",
    "No",
    "Help",
    "Eat",
    "Drink",
    "Love",
    "Sorry",
    "Please",
    "Stop",
    "Good",
    "Bad",
]

DYNAMIC_SIGN_LABELS = [
    "J",
    "Z",
    "Hello",
    "Thank you",
    "Yes",
    "No",
    "Help",
    "Sorry",
    "Please",
]


def ensure_project_directories() -> None:
    RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    SEQUENCE_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
