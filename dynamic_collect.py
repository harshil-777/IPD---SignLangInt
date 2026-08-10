import cv2
import mediapipe as mp
import numpy as np
import os
import csv
from datetime import datetime

# ---------- MediaPipe Tasks API ----------
mp_hands = mp.tasks.vision.HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

# ---------- Feature Extraction ----------
def calculate_angle(v1, v2):
    v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle)

def extract_features(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
    wrist = coords[0]
    shifted_coords = coords - wrist

    max_dist = np.max(np.linalg.norm(shifted_coords, axis=1))
    normalized_coords = shifted_coords / max_dist if max_dist > 0 else shifted_coords
    flattened_coords = normalized_coords.flatten().tolist()

    angles = []
    joint_groups = [
        (1, 2, 3), (2, 3, 4), (5, 6, 7), (6, 7, 8),
        (9, 10, 11), (10, 11, 12), (13, 14, 15),
        (14, 15, 16), (17, 18, 19), (18, 19, 20)
    ]
    for a, b, c in joint_groups:
        v1 = coords[a] - coords[b]
        v2 = coords[c] - coords[b]
        angles.append(calculate_angle(v1, v2))

    return flattened_coords + angles  # 63 + 10 = 73 features

def build_frame_features(results):
    right_features = [0.0] * 73
    left_features = [0.0] * 73

    if results.hand_landmarks and results.handedness:
        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            handedness = results.handedness[idx][0].category_name
            features = extract_features(hand_landmarks)

            if handedness == "Right":
                right_features = features
            elif handedness == "Left":
                left_features = features

    return right_features + left_features  # 146 features per frame

# ---------- Settings ----------
SEQ_LEN = 30
NO_HAND_RESET_LIMIT = 10

print("=== Dynamic Sign Collector ===")
label = input("Enter gesture label (e.g., HELP, ATTACK): ").strip().upper()
root_dir = input("Enter output folder (default: dynamic_dataset): ").strip()
if not root_dir:
    root_dir = "dynamic_dataset"

label_dir = os.path.join(root_dir, label)
os.makedirs(label_dir, exist_ok=True)

manifest_path = os.path.join(root_dir, "manifest.csv")
manifest_exists = os.path.exists(manifest_path)

# ---------- Camera ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# ---------- MediaPipe Landmarker ----------
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=2
)

sequence = []
samples_saved = 0
no_hand_frames = 0

print(f"\nCollecting dynamic samples for '{label}'...")
print(f"Each sample will contain {SEQ_LEN} frames.")
print("Press ESC to stop.")

with mp_hands.create_from_options(options) as landmarker:
    with open(manifest_path, "a", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        if not manifest_exists:
            writer.writerow(["file_path", "label", "frames", "created_at"])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            results = landmarker.detect(mp_image)
            frame_features = build_frame_features(results)

            hands_detected = bool(results.hand_landmarks)

            if hands_detected:
                no_hand_frames = 0
                sequence.append(frame_features)
            else:
                no_hand_frames += 1

            # Reset if hands disappear for too long
            if no_hand_frames >= NO_HAND_RESET_LIMIT:
                sequence.clear()
                no_hand_frames = 0

            # Save one complete sequence automatically
            if len(sequence) == SEQ_LEN:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                sample_name = f"{label}_{timestamp}.npy"
                sample_path = os.path.join(label_dir, sample_name)

                np.save(sample_path, np.array(sequence, dtype=np.float32))
                writer.writerow([sample_path, label, SEQ_LEN, timestamp])
                manifest_file.flush()

                samples_saved += 1
                sequence.clear()

            # ---------- UI Overlay ----------
            status = f"Label: {label} | Saved: {samples_saved} | Seq: {len(sequence)}/{SEQ_LEN}"
            color = (0, 255, 0) if hands_detected else (0, 0, 255)

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, "Press ESC to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.imshow("Dynamic Sign Collector", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()

print(f"\nDone. Saved {samples_saved} dynamic samples for '{label}' in: {label_dir}")
print(f"Manifest file: {manifest_path}")
