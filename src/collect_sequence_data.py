"""
Collect dynamic ASL sign sequences from a webcam and save them as .npy files.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.config import (
    CAMERA_INDEX,
    CLASS_LABELS,
    COLLECTION_PREPARE_DELAY_SECONDS,
    DYNAMIC_COLLECTION_COOLDOWN_SECONDS,
    DYNAMIC_SEQUENCE_LENGTH,
    DYNAMIC_SIGN_LABELS,
    SEQUENCE_DATASET_DIR,
    ensure_project_directories,
)
from src.utils import (
    draw_hand_landmarks,
    extract_hand_landmarks,
    initialize_hands,
    normalize_landmarks,
    open_camera,
)


def prompt_for_label() -> str:
    print("\nDynamic sign classes:")
    print(", ".join(DYNAMIC_SIGN_LABELS))
    label = input("Enter the dynamic sign label to record: ").strip()
    if not label:
        raise ValueError("Label cannot be empty.")
    return label


def prompt_for_sequence_count() -> int:
    raw_value = input("Enter number of sequences to record [30]: ").strip()
    if not raw_value:
        return 30
    count = int(raw_value)
    if count <= 0:
        raise ValueError("Sequence count must be a positive integer.")
    return count


def prompt_for_save_directory() -> Path:
    raw_value = input(f"Enter save folder [{SEQUENCE_DATASET_DIR}]: ").strip()
    if not raw_value:
        return SEQUENCE_DATASET_DIR
    return Path(raw_value).expanduser().resolve()


def sanitize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def next_sequence_file(label_dir: Path, safe_label: str) -> Path:
    existing_files = sorted(label_dir.glob(f"{safe_label}_*.npy"))
    next_index = len(existing_files) + 1
    return label_dir / f"{safe_label}_{next_index:04d}.npy"


def draw_status_overlay(
    frame,
    label: str,
    sequence_index: int,
    total_sequences: int,
    collected_frames: int,
    warning_text: str,
) -> None:
    cv2.putText(frame, f"Dynamic sign: {label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(
        frame,
        f"Sequence: {sequence_index}/{total_sequences}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Frames: {collected_frames}/{DYNAMIC_SEQUENCE_LENGTH}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        warning_text,
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255) if "warning" not in warning_text.lower() else (0, 0, 255),
        2,
    )
    cv2.putText(
        frame,
        "Press Q to stop recording",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )


def countdown(frame, label: str, seconds_left: int) -> None:
    cv2.putText(frame, f"Get ready for dynamic sign: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"Recording starts in {seconds_left}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


def collect_sequence_data(
    label: str,
    sequence_count: int,
    save_dir: Path,
    camera_index: int = CAMERA_INDEX,
) -> None:
    ensure_project_directories()
    safe_label = sanitize_label(label)
    label_dir = save_dir / safe_label
    label_dir.mkdir(parents=True, exist_ok=True)

    capture = open_camera(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Could not open the webcam. Check your camera and try again.")

    hands = initialize_hands(static_image_mode=False)
    try:
        for sequence_index in range(1, sequence_count + 1):
            sequence_frames: list[list[float]] = []
            prepare_start = time.time()

            while True:
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("Could not read a frame from the webcam.")

                frame = cv2.flip(frame, 1)
                elapsed = time.time() - prepare_start
                if elapsed < COLLECTION_PREPARE_DELAY_SECONDS:
                    seconds_left = int(COLLECTION_PREPARE_DELAY_SECONDS - elapsed) + 1
                    countdown(frame, label, seconds_left)
                    cv2.imshow("ASL Dynamic Sequence Collection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Collection stopped by user.")
                        return
                    continue
                break

            while len(sequence_frames) < DYNAMIC_SEQUENCE_LENGTH:
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("Could not read a frame from the webcam.")

                frame = cv2.flip(frame, 1)
                landmarks, results = extract_hand_landmarks(frame, hands)
                display_frame = draw_hand_landmarks(frame.copy(), results)

                warning_text = "Recording sequence..."
                if landmarks is None:
                    warning_text = "Warning: no hand detected, frame skipped"
                else:
                    normalized = normalize_landmarks(landmarks)
                    sequence_frames.append(normalized)

                draw_status_overlay(
                    display_frame,
                    label,
                    sequence_index,
                    sequence_count,
                    len(sequence_frames),
                    warning_text,
                )
                cv2.imshow("ASL Dynamic Sequence Collection", display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Collection stopped by user.")
                    return

            sequence_array = np.array(sequence_frames, dtype=np.float32)
            output_file = next_sequence_file(label_dir, safe_label)
            np.save(output_file, sequence_array)
            print(f"Saved sequence {sequence_index}/{sequence_count} to: {output_file}")

            cooldown_end = time.time() + DYNAMIC_COLLECTION_COOLDOWN_SECONDS
            while time.time() < cooldown_end:
                success, frame = capture.read()
                if not success:
                    break
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, "Sequence saved. Prepare the next motion...", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
                cv2.imshow("ASL Dynamic Sequence Collection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Collection stopped by user.")
                    return
    finally:
        hands.close()
        capture.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect dynamic ASL sign sequences from a webcam")
    parser.add_argument("--label", type=str, help="Dynamic sign label to record")
    parser.add_argument("--sequences", type=int, help="Number of sequences to record")
    parser.add_argument("--save_dir", type=str, help="Folder where sequence files will be saved")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        label = args.label if args.label else prompt_for_label()
        if not label:
            raise ValueError("Dynamic label cannot be empty.")
        if label not in DYNAMIC_SIGN_LABELS and label not in CLASS_LABELS:
            print("Warning: recording a custom dynamic label that is not in the default list.")

        sequence_count = args.sequences if args.sequences else prompt_for_sequence_count()
        save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else prompt_for_save_directory()
        collect_sequence_data(label, sequence_count, save_dir, camera_index=args.camera)
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
