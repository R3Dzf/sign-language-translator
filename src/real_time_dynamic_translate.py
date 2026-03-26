"""
Run dynamic or hybrid ASL translation using landmark sequences.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Deque, List, Optional, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.config import (
    AUTO_ACCEPT_COOLDOWN_FRAMES,
    AUTO_ACCEPT_CONFIDENCE_THRESHOLD,
    CAMERA_INDEX,
    CONFIDENCE_THRESHOLD,
    DYNAMIC_ACCEPT_COOLDOWN_FRAMES,
    DYNAMIC_CONFIDENCE_THRESHOLD,
    DYNAMIC_LABELS_PATH,
    DYNAMIC_MODEL_PATH,
    DYNAMIC_PREDICTION_HISTORY_SIZE,
    DYNAMIC_SEQUENCE_LENGTH,
    DYNAMIC_STABLE_COUNT,
    HYBRID_MOTION_THRESHOLD,
)
from src.real_time_translate import (
    TranslatorFrameResult,
    TranslatorSession,
    draw_translator_overlay,
    predict_sign,
)
from src.utils import (
    draw_hand_landmarks,
    extract_hand_landmarks,
    initialize_hands,
    load_model,
    open_camera,
    preprocess_input_for_prediction,
)


def load_dynamic_model():
    from tensorflow.keras.models import load_model

    if not DYNAMIC_MODEL_PATH.exists():
        raise FileNotFoundError(f"Dynamic model file not found: {DYNAMIC_MODEL_PATH}")
    if not DYNAMIC_LABELS_PATH.exists():
        raise FileNotFoundError(f"Dynamic label file not found: {DYNAMIC_LABELS_PATH}")

    model = load_model(DYNAMIC_MODEL_PATH)
    labels = json.loads(DYNAMIC_LABELS_PATH.read_text(encoding="utf-8"))
    return model, labels


def compute_motion_score(sequence_frames: np.ndarray) -> float:
    if len(sequence_frames) < 2:
        return 0.0
    frame_deltas = np.diff(sequence_frames, axis=0)
    frame_distances = np.linalg.norm(frame_deltas, axis=2)
    return float(np.mean(frame_distances))


class DynamicPredictionBuffer:
    def __init__(self, history_size: int, stable_count: int) -> None:
        self.history_size = history_size
        self.stable_count = stable_count
        self.recent_predictions: Deque[str] = deque(maxlen=history_size)
        self.recent_confidences: Deque[float] = deque(maxlen=history_size)

    def clear(self) -> None:
        self.recent_predictions.clear()
        self.recent_confidences.clear()

    def update(self, label: str, confidence: float) -> Tuple[Optional[str], float]:
        self.recent_predictions.append(label)
        self.recent_confidences.append(confidence)
        if len(self.recent_predictions) < self.stable_count:
            return None, 0.0

        last_labels = list(self.recent_predictions)[-self.stable_count :]
        if len(set(last_labels)) != 1:
            return None, 0.0

        last_confidences = list(self.recent_confidences)[-self.stable_count :]
        return last_labels[-1], float(sum(last_confidences) / len(last_confidences))


class DynamicTranslatorSession:
    def __init__(self) -> None:
        self.dynamic_model = None
        self.dynamic_labels: list[str] = []
        self.hands = None
        self.capture = None
        self.sequence_buffer: Deque[list[float]] = deque(maxlen=DYNAMIC_SEQUENCE_LENGTH)
        self.prediction_buffer = DynamicPredictionBuffer(
            history_size=DYNAMIC_PREDICTION_HISTORY_SIZE,
            stable_count=DYNAMIC_STABLE_COUNT,
        )
        self.sentence_words: List[str] = []
        self.current_prediction = "Waiting for motion..."
        self.stable_label = "-"
        self.info_message = "Show one hand and perform a moving sign"
        self.confidence = 0.0
        self.last_accepted_label = ""
        self.cooldown_frames_left = 0
        self.motion_started = False

    def start(self, camera_index: int = CAMERA_INDEX) -> None:
        if self.capture is not None:
            return
        self.dynamic_model, self.dynamic_labels = load_dynamic_model()
        self.hands = initialize_hands(static_image_mode=False)
        self.capture = open_camera(camera_index)
        if not self.capture.isOpened():
            self.stop()
            raise RuntimeError("Could not open the webcam. Check your camera and try again.")

    def stop(self) -> None:
        if self.hands is not None:
            self.hands.close()
            self.hands = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.sequence_buffer.clear()
        self.prediction_buffer.clear()
        self.motion_started = False

    def clear_sentence(self) -> None:
        self.sentence_words.clear()
        self.info_message = "Sentence cleared"

    def remove_last_word(self) -> None:
        if self.sentence_words:
            self.sentence_words.pop()
            self.info_message = "Removed last word"

    def accept_current_prediction(self) -> None:
        if self.current_prediction not in {"Unknown", "No hand detected", "Waiting for motion..."}:
            self.sentence_words.append(self.current_prediction)
            self.info_message = "Prediction added manually"

    def _predict_dynamic(self, sequence_array: np.ndarray) -> Tuple[str, float]:
        probabilities = self.dynamic_model.predict(sequence_array[np.newaxis, ...], verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        return self.dynamic_labels[best_index], float(probabilities[best_index])

    def read_result(self) -> TranslatorFrameResult:
        if self.capture is None or self.hands is None:
            return TranslatorFrameResult(
                success=False,
                frame=None,
                current_prediction="Camera is stopped",
                confidence=0.0,
                stable_label="-",
                sentence_words=self.sentence_words.copy(),
                info_message="Start the camera first",
            )

        success, frame = self.capture.read()
        if not success:
            return TranslatorFrameResult(
                success=False,
                frame=None,
                current_prediction="Camera read failed",
                confidence=0.0,
                stable_label="-",
                sentence_words=self.sentence_words.copy(),
                info_message="Could not read a frame from the webcam",
            )

        frame = cv2.flip(frame, 1)
        landmarks, results = extract_hand_landmarks(frame, self.hands)
        display_frame = draw_hand_landmarks(frame.copy(), results)

        if landmarks is None:
            self.current_prediction = "No hand detected"
            self.confidence = 0.0
            self.stable_label = "-"
            self.info_message = "No hand detected"
            self.sequence_buffer.clear()
            self.prediction_buffer.clear()
            self.motion_started = False
        else:
            normalized_frame = preprocess_input_for_prediction(landmarks).iloc[0].to_numpy(dtype=np.float32)
            self.sequence_buffer.append(normalized_frame.tolist())

            if len(self.sequence_buffer) < DYNAMIC_SEQUENCE_LENGTH:
                self.current_prediction = "Waiting for motion..."
                self.confidence = 0.0
                self.stable_label = "-"
                self.info_message = f"Collecting motion frames: {len(self.sequence_buffer)}/{DYNAMIC_SEQUENCE_LENGTH}"
            else:
                sequence_array = np.array(self.sequence_buffer, dtype=np.float32)
                motion_score = compute_motion_score(sequence_array.reshape(DYNAMIC_SEQUENCE_LENGTH, 21, 3))
                if motion_score >= HYBRID_MOTION_THRESHOLD:
                    self.motion_started = True

                if not self.motion_started:
                    self.current_prediction = "Waiting for motion..."
                    self.confidence = 0.0
                    self.stable_label = "-"
                    self.info_message = "Waiting for stronger motion..."
                else:
                    predicted_label, self.confidence = self._predict_dynamic(sequence_array)
                    if self.confidence < DYNAMIC_CONFIDENCE_THRESHOLD:
                        self.current_prediction = "Unknown"
                        self.stable_label = "-"
                        self.info_message = "Dynamic prediction below confidence threshold"
                        self.prediction_buffer.clear()
                    else:
                        self.current_prediction = predicted_label
                        stable_result, stable_confidence = self.prediction_buffer.update(predicted_label, self.confidence)
                        if stable_result is not None:
                            self.stable_label = stable_result
                            self.info_message = "Stable dynamic sign detected"
                            if (
                                self.cooldown_frames_left <= 0
                                and stable_result != self.last_accepted_label
                                and stable_confidence >= DYNAMIC_CONFIDENCE_THRESHOLD
                            ):
                                self.sentence_words.append(stable_result)
                                self.last_accepted_label = stable_result
                                self.cooldown_frames_left = DYNAMIC_ACCEPT_COOLDOWN_FRAMES
                                self.sequence_buffer.clear()
                                self.prediction_buffer.clear()
                                self.motion_started = False
                                self.info_message = "Dynamic sign added to sentence"
                        else:
                            self.stable_label = "-"
                            self.info_message = "Waiting for a more stable dynamic prediction"

        if self.cooldown_frames_left > 0:
            self.cooldown_frames_left -= 1
            if self.cooldown_frames_left == 0:
                self.last_accepted_label = ""

        return TranslatorFrameResult(
            success=True,
            frame=display_frame,
            current_prediction=self.current_prediction,
            confidence=self.confidence,
            stable_label=self.stable_label,
            sentence_words=self.sentence_words.copy(),
            info_message=self.info_message,
        )


class HybridTranslatorSession:
    def __init__(self) -> None:
        self.static_session = TranslatorSession()
        self.dynamic_session = DynamicTranslatorSession()
        self.sentence_words: List[str] = []
        self.last_mode = "Static"
        self.current_prediction = "No hand detected"

    def start(self, camera_index: int = CAMERA_INDEX) -> None:
        self.static_session.start(camera_index)
        self.dynamic_session.dynamic_model, self.dynamic_session.dynamic_labels = load_dynamic_model()
        self.dynamic_session.hands = self.static_session.hands
        self.dynamic_session.capture = self.static_session.capture

    def stop(self) -> None:
        self.static_session.stop()
        self.dynamic_session.capture = None
        self.dynamic_session.hands = None
        self.dynamic_session.sequence_buffer.clear()
        self.dynamic_session.prediction_buffer.clear()
        self.dynamic_session.motion_started = False

    def clear_sentence(self) -> None:
        self.sentence_words.clear()

    def remove_last_word(self) -> None:
        if self.sentence_words:
            self.sentence_words.pop()

    def accept_current_prediction(self) -> None:
        if self.current_prediction not in {"Unknown", "No hand detected", "Waiting for motion..."}:
            self.sentence_words.append(self.current_prediction)

    def read_result(self) -> TranslatorFrameResult:
        if self.static_session.capture is None or self.static_session.hands is None:
            return TranslatorFrameResult(
                success=False,
                frame=None,
                current_prediction="Camera is stopped",
                confidence=0.0,
                stable_label="-",
                sentence_words=[],
                info_message="Start the camera first",
            )

        success, frame = self.static_session.capture.read()
        if not success:
            return TranslatorFrameResult(
                success=False,
                frame=None,
                current_prediction="Camera read failed",
                confidence=0.0,
                stable_label="-",
                sentence_words=[],
                info_message="Could not read a frame from the webcam",
            )

        frame = cv2.flip(frame, 1)
        landmarks, results = extract_hand_landmarks(frame, self.static_session.hands)
        display_frame = draw_hand_landmarks(frame.copy(), results)

        if landmarks is None:
            self.static_session.buffer.clear()
            self.dynamic_session.sequence_buffer.clear()
            self.dynamic_session.prediction_buffer.clear()
            self.dynamic_session.motion_started = False
            self.current_prediction = "No hand detected"
            return TranslatorFrameResult(
                success=True,
                frame=display_frame,
                current_prediction="No hand detected",
                confidence=0.0,
                stable_label="-",
                sentence_words=self.sentence_words.copy(),
                info_message="No hand detected",
            )

        normalized_row = preprocess_input_for_prediction(landmarks)
        dynamic_frame = normalized_row.iloc[0].to_numpy(dtype=np.float32)
        self.dynamic_session.sequence_buffer.append(dynamic_frame.tolist())

        motion_score = 0.0
        if len(self.dynamic_session.sequence_buffer) >= 2:
            motion_score = compute_motion_score(
                np.array(self.dynamic_session.sequence_buffer, dtype=np.float32).reshape(
                    len(self.dynamic_session.sequence_buffer),
                    21,
                    3,
                )
            )
            if motion_score >= HYBRID_MOTION_THRESHOLD:
                self.dynamic_session.motion_started = True

        if (
            len(self.dynamic_session.sequence_buffer) >= DYNAMIC_SEQUENCE_LENGTH
            and self.dynamic_session.motion_started
        ):
            sequence_array = np.array(self.dynamic_session.sequence_buffer, dtype=np.float32)
            predicted_label, confidence = self.dynamic_session._predict_dynamic(sequence_array)
            if confidence >= DYNAMIC_CONFIDENCE_THRESHOLD:
                stable_result, stable_confidence = self.dynamic_session.prediction_buffer.update(predicted_label, confidence)
                self.last_mode = "Dynamic"
                self.current_prediction = predicted_label
                if (
                    stable_result is not None
                    and self.dynamic_session.cooldown_frames_left <= 0
                    and stable_result != self.dynamic_session.last_accepted_label
                ):
                    self.sentence_words.append(stable_result)
                    self.dynamic_session.last_accepted_label = stable_result
                    self.dynamic_session.cooldown_frames_left = DYNAMIC_ACCEPT_COOLDOWN_FRAMES
                    self.dynamic_session.sequence_buffer.clear()
                    self.dynamic_session.prediction_buffer.clear()
                    self.dynamic_session.motion_started = False
                if self.dynamic_session.cooldown_frames_left > 0:
                    self.dynamic_session.cooldown_frames_left -= 1
                    if self.dynamic_session.cooldown_frames_left == 0:
                        self.dynamic_session.last_accepted_label = ""
                return TranslatorFrameResult(
                    success=True,
                    frame=display_frame,
                    current_prediction=predicted_label,
                    confidence=confidence,
                    stable_label=stable_result or "-",
                    sentence_words=self.sentence_words.copy(),
                    info_message="Hybrid mode selected dynamic model",
                )

        predicted_label, confidence = predict_sign(self.static_session.model, self.static_session.label_encoder, normalized_row)
        self.last_mode = "Static"
        self.current_prediction = predicted_label
        if confidence < CONFIDENCE_THRESHOLD:
            self.static_session.buffer.clear()
            return TranslatorFrameResult(
                success=True,
                frame=display_frame,
                current_prediction="Unknown",
                confidence=confidence,
                stable_label="-",
                sentence_words=self.sentence_words.copy(),
                info_message="Hybrid mode fell back to static model",
            )

        stable_result, stable_confidence = self.static_session.buffer.update(predicted_label, confidence)
        if (
            stable_result is not None
            and self.static_session.cooldown_frames_left <= 0
            and stable_result != self.static_session.last_accepted_label
            and stable_confidence >= AUTO_ACCEPT_CONFIDENCE_THRESHOLD
        ):
            self.sentence_words.append(stable_result)
            self.static_session.last_accepted_label = stable_result
            self.static_session.cooldown_frames_left = AUTO_ACCEPT_COOLDOWN_FRAMES

        if self.static_session.cooldown_frames_left > 0:
            self.static_session.cooldown_frames_left -= 1
            if self.static_session.cooldown_frames_left == 0:
                self.static_session.last_accepted_label = ""

        return TranslatorFrameResult(
            success=True,
            frame=display_frame,
            current_prediction=predicted_label,
            confidence=confidence,
            stable_label=stable_result or "-",
            sentence_words=self.sentence_words.copy(),
            info_message="Hybrid mode selected static model",
        )


def run_dynamic_translator(mode: str = "dynamic") -> None:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "static":
        session = TranslatorSession()
        window_title = "ASL Static Translator"
    elif normalized_mode == "hybrid":
        session = HybridTranslatorSession()
        window_title = "ASL Hybrid Translator"
    else:
        session = DynamicTranslatorSession()
        window_title = "ASL Dynamic Translator"

    session.start(CAMERA_INDEX)
    try:
        while True:
            result = session.read_result()
            if not result.success or result.frame is None:
                print(result.info_message)
                break

            draw_translator_overlay(
                result.frame,
                result.current_prediction,
                result.confidence,
                result.sentence_words,
                result.stable_label,
                result.info_message,
            )
            cv2.imshow(window_title, result.frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                session.clear_sentence()
            if key == 32:
                session.accept_current_prediction()
            if key == 8:
                session.remove_last_word()
    finally:
        session.stop()
        cv2.destroyAllWindows()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run static, dynamic, or hybrid ASL translation")
    parser.add_argument(
        "--mode",
        choices=["static", "dynamic", "hybrid"],
        default="dynamic",
        help="Translator mode to run",
    )
    args = parser.parse_args()

    try:
        run_dynamic_translator(args.mode)
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
