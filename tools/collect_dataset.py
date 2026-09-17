"""CLI for recording labeled gesture sequences from the webcam into the
dataset directory, for later training.

Run with: python -m tools.collect_dataset
"""
from __future__ import annotations

import cv2

from sign_language.config import PATHS, SEQUENCE
from sign_language.dataset import SequenceDataset
from sign_language.landmarker import HandLandmarkerService


def main() -> None:
    print("=== Dynamic Sign Collector ===")
    label = input("Enter gesture label (e.g., HELP, ATTACK): ").strip().upper()
    root_dir = input(f"Enter output folder (default: {PATHS.dataset_dir}): ").strip() or PATHS.dataset_dir

    dataset = SequenceDataset(root_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    sequence: list[list[float]] = []
    samples_saved = 0
    no_hand_frames = 0

    print(f"\nCollecting dynamic samples for '{label}'...")
    print(f"Each sample will contain {SEQUENCE.seq_len} frames.")
    print("Press ESC to stop.")

    with HandLandmarkerService(PATHS.hand_landmarker_task) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(rgb)
            hands_detected = bool(result.hand_landmarks)

            if hands_detected:
                no_hand_frames = 0
                sequence.append(landmarker.extract_frame_features(result))
            else:
                no_hand_frames += 1

            if no_hand_frames >= SEQUENCE.no_hand_reset_limit:
                sequence.clear()
                no_hand_frames = 0

            if len(sequence) == SEQUENCE.seq_len:
                dataset.save_sample(label, sequence)
                samples_saved += 1
                sequence.clear()

            status = f"Label: {label} | Saved: {samples_saved} | Seq: {len(sequence)}/{SEQUENCE.seq_len}"
            color = (0, 255, 0) if hands_detected else (0, 0, 255)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, "Press ESC to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.imshow("Dynamic Sign Collector", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nDone. Saved {samples_saved} dynamic samples for '{label}' in: {dataset.root_dir}")
    print(f"Manifest file: {dataset.manifest_path()}")


if __name__ == "__main__":
    main()
