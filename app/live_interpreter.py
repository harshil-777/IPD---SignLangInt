"""Live webcam interpreter: captures video, extracts gesture features,
gets predictions from the backend, stabilizes them into words, and walks
the user through the FIR questionnaire.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from sign_language.config import INFERENCE, PATHS, SEQUENCE
from sign_language.fir import FIRWorkflow
from sign_language.landmarker import HandLandmarkerService
from sign_language.prediction_client import HttpPredictionClient, PredictionClient
from sign_language.sentence import SentenceBuilder
from sign_language.smoothing import FeatureSmoother
from sign_language.stabilizer import PredictionStabilizer

from . import hud

KEY_ESC = 27
KEY_ENTER = 13
KEY_BACKSPACE = 8
KEY_DELETE = 127


class LiveInterpreterApp:
    def __init__(self, prediction_client: PredictionClient | None = None):
        self._landmarker = HandLandmarkerService(PATHS.hand_landmarker_task)
        self._client = prediction_client or HttpPredictionClient()
        self._stabilizer = PredictionStabilizer(INFERENCE.confidence_threshold, INFERENCE.stable_window)
        self._smoother = FeatureSmoother(INFERENCE.smoothing_alpha)
        self._sentence = SentenceBuilder()
        self._fir = FIRWorkflow()

        self._frame_history: deque[list[float]] = deque(maxlen=SEQUENCE.seq_len)
        self._no_hand_counter = 0
        self._frame_counter = 0
        self._current_prediction = ""
        self._generated_sentence = ""

    def run(self) -> None:
        cap = cv2.VideoCapture(0)
        try:
            with self._landmarker:
                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break

                    frame = cv2.flip(frame, 1)
                    self._process_frame(frame)

                    hud.draw_hud(
                        frame,
                        self._fir.current_question,
                        self._current_prediction,
                        self._sentence.text(last_n=6),
                        self._generated_sentence,
                    )
                    cv2.imshow("Unified Sign Language System", frame)

                    if self._handle_key(cv2.waitKey(1) & 0xFF):
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _process_frame(self, frame: np.ndarray) -> None:
        self._current_prediction = ""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect(rgb)

        if result.hand_landmarks:
            self._no_hand_counter = 0
            hud.draw_hand_landmarks(frame, result)

            features = np.array(self._landmarker.extract_frame_features(result), dtype=np.float32)
            features = self._smoother.smooth(features)
            self._frame_history.append(features.tolist())

            self._frame_counter += 1
            sequence_ready = len(self._frame_history) == SEQUENCE.seq_len
            if sequence_ready and self._frame_counter % INFERENCE.predict_every_n_frames == 0:
                self._client.request_prediction_async(list(self._frame_history))

            prediction, confidence, is_new = self._client.consume_prediction()
            if is_new:
                self._current_prediction = self._stabilizer.update(prediction, confidence)
        else:
            self._no_hand_counter += 1
            last_features = self._smoother.last
            if last_features is not None and self._no_hand_counter <= SEQUENCE.no_hand_reset_limit:
                self._frame_history.append(last_features.tolist())
            elif self._no_hand_counter > SEQUENCE.no_hand_reset_limit:
                self._reset_tracking_state()

        self._sentence.add(self._current_prediction)

    def _reset_tracking_state(self) -> None:
        self._stabilizer.reset()
        self._frame_history.clear()
        self._smoother.reset()
        self._sentence.reset_repeat_guard()

    def _handle_key(self, key: int) -> bool:
        """Returns True if the app should exit."""
        if key == ord("n"):
            self._sentence.set_words(self._fir.next_step(self._sentence.words))
            self._generated_sentence = ""
        elif key == ord("b"):
            self._sentence.set_words(self._fir.previous_step(self._sentence.words))
            self._generated_sentence = ""
        elif key == ord("c"):
            self._sentence.clear()
            self._stabilizer.reset()
            self._frame_history.clear()
            self._smoother.reset()
            self._generated_sentence = ""
        elif key in (KEY_BACKSPACE, KEY_DELETE, ord("d")):
            self._sentence.delete_last()
            self._stabilizer.reset()
            self._current_prediction = ""
        elif key == KEY_ENTER:
            compiled_inputs = self._fir.compile_report_input(self._sentence.words)
            self._generated_sentence = "Generating Incident Report... Please wait."
            self._client.generate_report(compiled_inputs, self._on_report_ready)
        elif key == KEY_ESC:
            return True

        return False

    def _on_report_ready(self, text: str) -> None:
        self._generated_sentence = text
