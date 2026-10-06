"""Interactive OpenCV-based dynamic dataset collector.

Controls (while the OpenCV window is focused):
  SPACE     – Toggle recording ON / OFF
  R         – Discard the current partial sequence and reset
  L         – Change label on the fly (prompts in terminal)
  ESC / Q   – Quit and print a summary

Run with:
    python -m tools.collect_dataset
    python -m tools.collect_dataset --label HELP
    python -m tools.collect_dataset --label ATTACK --samples 50
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

# Add project root to path so it works when run from tools/ dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from sign_language.config import PATHS, SEQUENCE
from sign_language.dataset import SequenceDataset
from sign_language.landmarker import HandLandmarkerService

# ──────────────────────────────────────────────────────────────
# HUD helpers
# ──────────────────────────────────────────────────────────────

_FONT  = cv2.FONT_HERSHEY_SIMPLEX

# Colors (BGR)
_WHITE  = (255, 255, 255)
_GREEN  = (0, 230, 80)
_RED    = (0, 60, 230)
_YELLOW = (0, 220, 255)
_CYAN   = (230, 210, 0)
_GRAY   = (150, 150, 150)
_DARK   = (20, 20, 20)
_ORANGE = (0, 140, 255)


def _filled_rect(frame: np.ndarray, pt1, pt2, color, alpha: float = 0.55) -> None:
    """Draw a semi-transparent filled rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    overlay = frame[y1:y2, x1:x2].copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, frame[y1:y2, x1:x2], 1 - alpha, 0, frame[y1:y2, x1:x2])


def _text(frame, text: str, pos, scale: float = 0.6, color=_WHITE, thickness: int = 1) -> None:
    cv2.putText(frame, text, pos, _FONT, scale, _DARK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, _FONT, scale, color, thickness, cv2.LINE_AA)


def _draw_progress_bar(frame: np.ndarray, x: int, y: int, width: int, height: int,
                       progress: float, color_fill=_GREEN, color_bg=(50, 50, 50)) -> None:
    cv2.rectangle(frame, (x, y), (x + width, y + height), color_bg, -1)
    filled = int(width * min(1.0, max(0.0, progress)))
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + height), color_fill, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), _WHITE, 1)


def _draw_hand_landmarks(frame: np.ndarray, result) -> None:
    h, w = frame.shape[:2]
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    for hand_lms in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 200, 80), 1, cv2.LINE_AA)
        for px, py in pts:
            cv2.circle(frame, (px, py), 3, _GREEN, -1, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, label: str, recording: bool,
              seq_len: int, seq_cap: int,
              samples_saved: int, target_samples: int,
              hands_detected: bool, fps: float,
              show_flash: bool) -> None:
    h, w = frame.shape[:2]

    # Top panel
    _filled_rect(frame, (0, 0), (w, 72), _DARK, alpha=0.72)

    status_color = _GREEN if recording else _ORANGE
    status_text  = "  REC" if recording else "  IDLE"
    _text(frame, status_text, (12, 28), scale=0.8, color=status_color, thickness=2)
    _text(frame, f"Label: {label}", (130, 28), scale=0.8, color=_YELLOW, thickness=2)
    hand_color = _GREEN if hands_detected else _RED
    hand_text  = "HAND OK" if hands_detected else "NO HAND"
    _text(frame, hand_text, (w - 185, 28), scale=0.6, color=hand_color, thickness=1)
    _text(frame, f"FPS: {fps:.1f}", (w - 185, 55), scale=0.5, color=_GRAY)
    _text(frame, f"Saved: {samples_saved}" +
          (f"/{target_samples}" if target_samples > 0 else ""), (12, 55), scale=0.55, color=_WHITE)

    # Frame sequence bar
    bar_x, bar_y, bar_w, bar_h = 12, 80, w - 24, 14
    bar_color = _GREEN if recording else _GRAY
    _draw_progress_bar(frame, bar_x, bar_y, bar_w, bar_h,
                       seq_len / seq_cap if seq_cap > 0 else 0,
                       color_fill=bar_color)
    _text(frame, f"Frames: {seq_len}/{seq_cap}",
          (bar_x + 4, bar_y + bar_h + 16), scale=0.45, color=_WHITE)

    # Overall collection bar
    sbar_y = bar_y + bar_h + 34
    if target_samples > 0:
        overall = samples_saved / target_samples
        sbar_color = _CYAN if overall < 1.0 else _GREEN
        _draw_progress_bar(frame, bar_x, sbar_y, bar_w, bar_h, overall, color_fill=sbar_color)
        _text(frame, f"Collection progress: {int(overall * 100)}%",
              (bar_x + 4, sbar_y + bar_h + 16), scale=0.45, color=_WHITE)

    # "SAMPLE SAVED!" flash
    if show_flash:
        flash_x = w // 2 - 150
        flash_y = h // 2 - 20
        _filled_rect(frame, (flash_x - 10, flash_y - 36), (flash_x + 310, flash_y + 10), _DARK, alpha=0.80)
        _text(frame, "SAMPLE SAVED!", (flash_x, flash_y),
              scale=1.2, color=_GREEN, thickness=2)

    # Bottom controls bar
    ctrl_y = h - 28
    _filled_rect(frame, (0, ctrl_y - 14), (w, h), _DARK, alpha=0.72)
    controls = "[SPACE] Start/Stop  [R] Reset seq  [L] Change label  [ESC/Q] Quit"
    _text(frame, controls, (8, ctrl_y + 4), scale=0.42, color=_GRAY)


# ──────────────────────────────────────────────────────────────
# Collector
# ──────────────────────────────────────────────────────────────

def collect(label: str, target_samples: int, dataset: SequenceDataset) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    window = "Dynamic Sign Collector"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)

    recording      = False
    sequence: list[list[float]] = []
    samples_saved  = 0
    no_hand_frames = 0
    flash_until    = 0.0
    fps_q: deque[float] = deque(maxlen=30)
    t_prev = time.monotonic()

    print(f"\n[Collector] Label='{label}' | Target={target_samples} samples | "
          f"Seq={SEQUENCE.seq_len} frames")
    print("  SPACE – start/stop  |  R – reset  |  L – change label  |  ESC/Q – quit\n")

    with HandLandmarkerService(PATHS.hand_landmarker_task) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            t_now = time.monotonic()
            fps_q.append(1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now
            fps = sum(fps_q) / len(fps_q)

            # Detection
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(rgb)
            hands_detected = bool(result.hand_landmarks)

            if hands_detected:
                _draw_hand_landmarks(frame, result)
                no_hand_frames = 0
                if recording:
                    features = landmarker.extract_frame_features(result)
                    sequence.append(features)
            else:
                no_hand_frames += 1

            # Reset partial if hand disappears too long during recording
            if recording and no_hand_frames >= SEQUENCE.no_hand_reset_limit:
                sequence.clear()
                no_hand_frames = 0

            # Save completed sequence
            if recording and len(sequence) == SEQUENCE.seq_len:
                dataset.save_sample(label, sequence)
                samples_saved += 1
                sequence.clear()
                flash_until = time.monotonic() + 0.6
                print(f"  Saved sample {samples_saved}"
                      + (f"/{target_samples}" if target_samples > 0 else "")
                      + f" for '{label}'")

                if target_samples > 0 and samples_saved >= target_samples:
                    print(f"\n[Collector] Target of {target_samples} samples reached — stopping.")
                    recording = False

            _draw_hud(
                frame,
                label          = label,
                recording      = recording,
                seq_len        = len(sequence),
                seq_cap        = SEQUENCE.seq_len,
                samples_saved  = samples_saved,
                target_samples = target_samples,
                hands_detected = hands_detected,
                fps            = fps,
                show_flash     = time.monotonic() < flash_until,
            )

            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                recording = not recording
                sequence.clear()
                no_hand_frames = 0
                print(f"  → {'RECORDING' if recording else 'PAUSED'}")

            elif key in (27, ord("q"), ord("Q")):
                break

            elif key in (ord("r"), ord("R")):
                sequence.clear()
                no_hand_frames = 0
                print("  → Sequence reset.")

            elif key in (ord("l"), ord("L")):
                recording = False
                sequence.clear()
                new_label = input("  Enter new label: ").strip().upper()
                if new_label:
                    label = new_label
                    print(f"  → Label changed to '{label}'")

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[Collector] Done. Saved {samples_saved} samples for '{label}'.")
    print(f"            Manifest : {dataset.manifest_path()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect dynamic gesture sequences into the dataset."
    )
    parser.add_argument(
        "--label", "-l", type=str, default=None,
        help="Gesture label (e.g. HELP, ATTACK, A). Prompted if omitted.",
    )
    parser.add_argument(
        "--samples", "-n", type=int, default=50,
        help="Target sample count (default: 50). 0 = unlimited.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help=f"Output dataset directory (default: {PATHS.dataset_dir}).",
    )
    args = parser.parse_args()

    label = (args.label or input("Enter gesture label (e.g. HELP, ATTACK, A): ").strip()).upper()
    if not label:
        raise ValueError("Label cannot be empty.")

    root_dir = args.output or PATHS.dataset_dir
    dataset  = SequenceDataset(root_dir)

    print("\n=== Dynamic Sign Collector ===")
    print(f"  Label       : {label}")
    print(f"  Target      : {'unlimited' if args.samples == 0 else args.samples} samples")
    print(f"  Sequence len: {SEQUENCE.seq_len} frames")
    print(f"  Output dir  : {root_dir}")
    print(f"  Press SPACE in the window to start collecting.\n")

    collect(label, args.samples, dataset)


if __name__ == "__main__":
    main()
