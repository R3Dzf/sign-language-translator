"""
Evaluate the saved static model on a brand-new live webcam session.
This gives a more realistic estimate than replaying old training sessions.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.collect_data import countdown_overlay, prompt_for_label
from src.config import CAMERA_INDEX, COLLECTION_PREPARE_DELAY_SECONDS, DEFAULT_SAMPLES_PER_CLASS
from src.utils import (
    build_feature_column_names,
    draw_hand_landmarks,
    extract_hand_landmarks,
    initialize_hands,
    load_model,
    normalize_landmarks,
    open_camera,
)


def prompt_for_sample_count() -> int:
    raw_value = input(f"Enter number of live evaluation samples [{DEFAULT_SAMPLES_PER_CLASS}]: ").strip()
    if not raw_value:
        return DEFAULT_SAMPLES_PER_CLASS
    sample_count = int(raw_value)
    if sample_count <= 0:
        raise ValueError("Sample count must be a positive integer.")
    return sample_count


def collect_live_session_samples(label: str, sample_count: int, camera_index: int = CAMERA_INDEX) -> pd.DataFrame:
    capture = open_camera(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Could not open the webcam. Check your camera and try again.")

    hands = initialize_hands(static_image_mode=False)
    collected_rows: list[dict[str, float | int | str]] = []
    skipped_frames = 0
    start_time = time.time()
    session_id = f"live_eval_{label.replace(' ', '_')}"
    feature_columns = build_feature_column_names()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Could not read a frame from the webcam.")
                break

            frame = cv2.flip(frame, 1)
            elapsed = time.time() - start_time

            if elapsed < COLLECTION_PREPARE_DELAY_SECONDS:
                seconds_left = int(COLLECTION_PREPARE_DELAY_SECONDS - elapsed) + 1
                countdown_overlay(frame, seconds_left, label)
                cv2.imshow("ASL Live Session Evaluation", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Live evaluation stopped by user.")
                    break
                continue

            landmarks, results = extract_hand_landmarks(frame, hands)
            display_frame = draw_hand_landmarks(frame.copy(), results)

            if landmarks is not None:
                normalized_landmarks = normalize_landmarks(landmarks)
                row = {
                    "label": label,
                    "session_id": session_id,
                    "sample_index": len(collected_rows),
                }
                for column_name, value in zip(feature_columns, normalized_landmarks):
                    row[column_name] = value
                collected_rows.append(row)
                status_color = (0, 255, 0)
                status_text = "Hand detected - sample saved"
            else:
                skipped_frames += 1
                status_color = (0, 0, 255)
                status_text = "No hand detected - sample skipped"

            cv2.putText(
                display_frame,
                f"Class: {label}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display_frame,
                f"Saved: {len(collected_rows)}/{sample_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display_frame,
                status_text,
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )
            cv2.putText(
                display_frame,
                "Press Q to stop",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.imshow("ASL Live Session Evaluation", display_frame)

            if len(collected_rows) >= sample_count:
                print(f"Finished collecting {len(collected_rows)} live evaluation samples for {label}.")
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Live evaluation stopped by user.")
                break

    finally:
        hands.close()
        capture.release()
        cv2.destroyAllWindows()

    print(f"Collected live samples: {len(collected_rows)}")
    print(f"Skipped frames without a visible hand: {skipped_frames}")

    if not collected_rows:
        raise ValueError("No live samples were collected. Try again with your hand clearly visible.")

    return pd.DataFrame(collected_rows)


def evaluate_live_session(label: str, sample_count: int, camera_index: int = CAMERA_INDEX) -> None:
    model, label_encoder = load_model()
    live_session = collect_live_session_samples(label, sample_count, camera_index=camera_index)
    feature_columns = build_feature_column_names()

    x_live = live_session[feature_columns].copy()
    y_true_labels = live_session["label"].astype(str)
    y_true = label_encoder.transform(y_true_labels)
    y_pred = model.predict(x_live)

    live_accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0,
    )
    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(label_encoder.classes_)),
    )
    confusion_frame = pd.DataFrame(
        confusion,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )

    print(f"Real-world session accuracy: {live_accuracy:.4f}")
    print("\nConfusion matrix:")
    print(confusion_frame)
    print("\nClassification report:")
    print(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the static ASL model on a new live session")
    parser.add_argument("--label", type=str, help="Ground-truth label for the live session")
    parser.add_argument("--samples", type=int, help="Number of live samples to collect")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        label = args.label if args.label else prompt_for_label()
        sample_count = args.samples if args.samples else prompt_for_sample_count()
        evaluate_live_session(label, sample_count, camera_index=args.camera)
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
