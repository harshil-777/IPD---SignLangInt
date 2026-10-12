"""On-screen drawing for the live interpreter.

Kept separate from recognition/state logic (`live_interpreter.py`) so
rendering can change — colors, layout, a future theme — without touching
gesture-recognition code, and vice versa.
"""
from __future__ import annotations

import textwrap

import cv2
import numpy as np

CONTROLS_TEXT = "[ENTER] Generate | [N] Next | [B] Back | [C] Clear | [D]/[BACKSPACE] Delete | [ESC] Exit"


def draw_hand_landmarks(frame: np.ndarray, detection_result) -> None:
    h, w, _ = frame.shape
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    for hand_landmarks in detection_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (80, 200, 80), 1, cv2.LINE_AA)
        for px, py in pts:
            cv2.circle(frame, (px, py), 3, (0, 255, 0), -1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, question: str, prediction: str, current_input: str, generated_sentence: str) -> None:
    h, _, _ = frame.shape

    cv2.putText(frame, "INCIDENT REPORT - SMART SIGN LANGUAGE SYSTEM", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
    cv2.putText(frame, question, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
    cv2.putText(frame, f"Prediction: {prediction}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
    cv2.putText(frame, f"Current Input: {current_input}", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 150), 2)
    cv2.putText(frame, CONTROLS_TEXT, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    if generated_sentence:
        _draw_report_box(frame, generated_sentence)


def _draw_report_box(frame: np.ndarray, generated_sentence: str) -> None:
    h, w, _ = frame.shape
    cv2.rectangle(frame, (5, 170), (w - 5, h - 40), (40, 40, 40), -1)

    wrapped = textwrap.wrap("FINAL INCIDENT NARRATIVE:", width=60)
    wrapped.extend(textwrap.wrap(generated_sentence, width=60))

    y = 195
    for i, line in enumerate(wrapped):
        color = (0, 255, 255) if i == 0 else (220, 220, 220)
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y += 25
