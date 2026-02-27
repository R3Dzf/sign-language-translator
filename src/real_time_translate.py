"""
Run the trained ASL translator in real time with sentence building.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
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
    OVERLAY_BOX_HEIGHT,
    OVERLAY_BOX_WIDTH,
    OVERLAY_BOX_X,
    OVERLAY_BOX_Y,
    OVERLAY_TEXT_LINE1_Y,
    OVERLAY_TEXT_LINE_GAP,
    OVERLAY_TEXT_X,
    MAJORITY_VOTE_RATIO,
    PREDICTION_HISTORY_SIZE,
    STABLE_FRAMES_REQUIRED,
)
from src.utils import (
    draw_hand_landmarks,
    extract_hand_landmarks,
    initialize_hands,
    load_model,
    open_camera,
    preprocess_input_for_prediction,
)


class StablePredictionBuffer:
    def __init__(self, required_frames: int, history_size: int, majority_ratio: float) -> None:
        self.required_frames = required_frames
        self.history_size = history_size
        self.majority_ratio = majority_ratio
        self.recent_predictions: Deque[str] = deque(maxlen=history_size)
        self.recent_confidences: Deque[float] = deque(maxlen=history_size)

    def clear(self) -> None:
        self.recent_predictions.clear()
        self.recent_confidences.clear()

    def update(self, label: str, confidence: float) -> Tuple[Optional[str], float]:
        self.recent_predictions.append(label)
        self.recent_confidences.append(confidence)
        if len(self.recent_predictions) < self.required_frames:
            return None, 0.0

        label_counts = {}
        for prediction in self.recent_predictions:
            label_counts[prediction] = label_counts.get(prediction, 0) + 1

        dominant_label = max(label_counts, key=label_counts.get)
        dominant_count = label_counts[dominant_label]
        dominance_ratio = dominant_count / len(self.recent_predictions)
        if dominance_ratio < self.majority_ratio:
            return None, 0.0

        confidences = [
            conf
            for pred, conf in zip(self.recent_predictions, self.recent_confidences)
            if pred == dominant_label
        ]
        mean_confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
        return dominant_label, mean_confidence


@dataclass
class TranslatorFrameResult:
    success: bool
    frame: Optional[np.ndarray]
    current_prediction: str
    confidence: float
    stable_label: str
    sentence_words: List[str]
    info_message: str


def predict_sign(model, label_encoder, feature_row: np.ndarray) -> Tuple[str, float]:
    probabilities = model.predict_proba(feature_row)[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    label = label_encoder.inverse_transform([best_index])[0]
    return label, confidence


class TranslatorSession:
    def __init__(self) -> None:
        self.model = None
        self.label_encoder = None
        self.hands = None
        self.capture = None
        self.buffer = StablePredictionBuffer(
            required_frames=STABLE_FRAMES_REQUIRED,
            history_size=PREDICTION_HISTORY_SIZE,
            majority_ratio=MAJORITY_VOTE_RATIO,
        )
        self.sentence_words: List[str] = []
        self.current_prediction = "No hand detected"
        self.stable_label = "-"
        self.info_message = "Show one hand clearly to the camera"
        self.confidence = 0.0
        self.last_accepted_label = ""
        self.cooldown_frames_left = 0

    def start(self, camera_index: int = CAMERA_INDEX) -> None:
        if self.capture is not None:
            return
        self.model, self.label_encoder = load_model()
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
        self.buffer.clear()

    def clear_sentence(self) -> None:
        self.sentence_words.clear()
        self.info_message = "Sentence cleared"

    def remove_last_word(self) -> None:
        if self.sentence_words:
            self.sentence_words.pop()
            self.info_message = "Removed last word"

    def accept_current_prediction(self) -> None:
        if self.current_prediction not in {"Unknown", "No hand detected"}:
            self.sentence_words.append(self.current_prediction)
            self.info_message = "Prediction added manually"

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
            self.buffer.clear()
        else:
            feature_row = preprocess_input_for_prediction(landmarks)
            predicted_label, self.confidence = predict_sign(self.model, self.label_encoder, feature_row)

            if self.confidence < CONFIDENCE_THRESHOLD:
                self.current_prediction = "Unknown"
                self.stable_label = "-"
                self.info_message = "Prediction below confidence threshold"
                self.buffer.clear()
            else:
                self.current_prediction = predicted_label
                stable_result, stable_confidence = self.buffer.update(predicted_label, self.confidence)
                if stable_result is not None:
                    self.stable_label = stable_result
                    self.info_message = "Stable prediction detected"
                    if (
                        self.cooldown_frames_left <= 0
                        and stable_result != self.last_accepted_label
                        and stable_confidence >= AUTO_ACCEPT_CONFIDENCE_THRESHOLD
                    ):
                        self.sentence_words.append(stable_result)
                        self.last_accepted_label = stable_result
                        self.cooldown_frames_left = AUTO_ACCEPT_COOLDOWN_FRAMES
                        self.info_message = "Stable sign added to sentence"
                    elif stable_confidence < AUTO_ACCEPT_CONFIDENCE_THRESHOLD:
                        self.info_message = "Stable sign detected"
                else:
                    self.stable_label = "-"
                    self.info_message = "Waiting for a more stable prediction"

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


def draw_translator_overlay(
    frame,
    current_prediction: str,
    confidence: float,
    sentence_words: List[str],
    stable_label: str,
    info_message: str,
) -> None:
    x1 = OVERLAY_BOX_X
    y1 = OVERLAY_BOX_Y
    x2 = OVERLAY_BOX_X + OVERLAY_BOX_WIDTH
    y2 = OVERLAY_BOX_Y + OVERLAY_BOX_HEIGHT
    text_x = OVERLAY_TEXT_X
    line1_y = OVERLAY_TEXT_LINE1_Y

    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 36, 56), thickness=-1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (15, 118, 110), thickness=2)
    cv2.putText(
        frame,
        f"Prediction: {current_prediction}",
        (text_x, line1_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (text_x, line1_y + OVERLAY_TEXT_LINE_GAP),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (102, 255, 204),
        2,
    )
    cv2.putText(
        frame,
        f"Stable: {stable_label}",
        (text_x, line1_y + OVERLAY_TEXT_LINE_GAP * 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 230, 90),
        2,
    )
    cv2.putText(
        frame,
        f"Sentence: {' '.join(sentence_words) if sentence_words else '(empty)'}",
        (text_x, line1_y + OVERLAY_TEXT_LINE_GAP * 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (120, 255, 120),
        2,
    )
    cv2.putText(
        frame,
        info_message,
        (text_x, line1_y + OVERLAY_TEXT_LINE_GAP * 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (120, 220, 255),
        2,
    )


def run_translator() -> None:
    session = TranslatorSession()
    session.start(CAMERA_INDEX)

    try:
        while True:
            result = session.read_result()
            if not result.success or result.frame is None:
                print("Could not read a frame from the webcam.")
                break

            draw_translator_overlay(
                result.frame,
                result.current_prediction,
                result.confidence,
                result.sentence_words,
                result.stable_label,
                result.info_message,
            )
            cv2.imshow("ASL Real-Time Translator", result.frame)

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
    try:
        run_translator()
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
