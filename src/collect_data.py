"""
Collect ASL landmark samples from a webcam and save them to dataset/raw.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys
import re

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2

from src.config import (
    CAMERA_INDEX,
    CLASS_LABELS,
    COLLECTION_PREPARE_DELAY_SECONDS,
    DEFAULT_SAMPLES_PER_CLASS,
    RAW_DATASET_DIR,
    SUPPORTED_FILE_EXTENSION,
    ensure_project_directories,
)
from src.utils import (
    draw_hand_landmarks,
    extract_hand_landmarks,
    initialize_hands,
    normalize_landmarks,
    open_camera,
    save_sample_data,
)


def prompt_for_label() -> str:
    print("\nAvailable classes:")
    print(", ".join(CLASS_LABELS))
    label = input("Enter the class name to collect: ").strip()
    if label not in CLASS_LABELS:
        raise ValueError("Invalid class name. Use one of the configured ASL classes.")
    return label


def prompt_for_sample_count() -> int:
    raw_value = input(f"Enter number of samples to record [{DEFAULT_SAMPLES_PER_CLASS}]: ").strip()
    if not raw_value:
        return DEFAULT_SAMPLES_PER_CLASS
    count = int(raw_value)
    if count <= 0:
        raise ValueError("Sample count must be a positive integer.")
    return count


def prompt_for_save_directory() -> Path:
    raw_value = input(f"Enter save folder [{RAW_DATASET_DIR}]: ").strip()
    if not raw_value:
        return RAW_DATASET_DIR
    return Path(raw_value).expanduser().resolve()


def countdown_overlay(frame, seconds_left: int, label: str) -> None:
    cv2.putText(
        frame,
        f"Get ready for: {label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Recording starts in {seconds_left}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )


def build_next_session_file(label: str, save_dir: Path) -> tuple[str, Path]:
    safe_label = re.sub(r"[^A-Za-z0-9_]+", "_", label.replace(" ", "_")).strip("_")
    existing_indices = []
    for csv_file in save_dir.glob(f"{safe_label}_session_*{SUPPORTED_FILE_EXTENSION}"):
        match = re.fullmatch(
            rf"{re.escape(safe_label)}_session_(\d+){re.escape(SUPPORTED_FILE_EXTENSION)}",
            csv_file.name,
        )
        if match:
            existing_indices.append(int(match.group(1)))

    next_index = max(existing_indices, default=0) + 1
    session_id = f"{safe_label}_session_{next_index:03d}"
    session_file = save_dir / f"{session_id}{SUPPORTED_FILE_EXTENSION}"
    return session_id, session_file


def collect_samples_for_class(
    label: str,
    sample_count: int,
    save_dir: Path,
    camera_index: int = CAMERA_INDEX,
) -> None:
    ensure_project_directories()
    save_dir.mkdir(parents=True, exist_ok=True)
    session_id, class_file = build_next_session_file(label, save_dir)

    capture = open_camera(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Could not open the webcam. Check your camera and try again.")

    hands = initialize_hands(static_image_mode=False)
    collected = 0
    skipped_frames = 0
    start_time = time.time()

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
                cv2.imshow("ASL Data Collection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Collection stopped by user.")
                    break
                continue

            landmarks, results = extract_hand_landmarks(frame, hands)
            display_frame = draw_hand_landmarks(frame.copy(), results)

            if landmarks is not None:
                normalized_landmarks = normalize_landmarks(landmarks)
                save_sample_data(
                    class_file,
                    label,
                    session_id,
                    collected,
                    normalized_landmarks,
                )
                collected += 1
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
                f"Saved: {collected}/{sample_count}",
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

            cv2.imshow("ASL Data Collection", display_frame)

            if collected >= sample_count:
                print(f"Finished collecting {collected} samples for {label}.")
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Collection stopped by user.")
                break

    finally:
        hands.close()
        capture.release()
        cv2.destroyAllWindows()

    print(f"Saved samples to: {class_file}")
    print(f"Session ID: {session_id}")
    print(f"Collected samples: {collected}")
    print(f"Skipped frames without a visible hand: {skipped_frames}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ASL landmark samples from a webcam")
    parser.add_argument("--label", type=str, help="Class name to record")
    parser.add_argument("--samples", type=int, help="Number of samples to collect")
    parser.add_argument("--save_dir", type=str, help="Folder where CSV files will be saved")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        label = args.label if args.label else prompt_for_label()
        if label not in CLASS_LABELS:
            raise ValueError("Invalid class name. Use one of the configured ASL classes.")

        sample_count = args.samples if args.samples else prompt_for_sample_count()
        if sample_count <= 0:
            raise ValueError("Sample count must be a positive integer.")

        save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else prompt_for_save_directory()
        collect_samples_for_class(label, sample_count, save_dir, camera_index=args.camera)
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
