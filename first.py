import cv2
import mediapipe as mp
import csv
import numpy as np

# MediaPipe Initialization (New API)
mp_hands = mp.tasks.vision.HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

# --- EXACT MATH FUNCTIONS ---
def calculate_angle(v1, v2):
    v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle)

def extract_features(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
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

    return flattened_coords + angles
# ----------------------------

print("=== Custom Dataset Collector ===")
label = input("Enter gesture label (e.g., HELP, ATTACK): ").upper()
csv_filename = input("Enter output CSV name (e.g., help_fixed.csv): ")

if not csv_filename.endswith(".csv"):
    csv_filename += ".csv"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    num_hands=2
)

cap = cv2.VideoCapture(0)
frames_collected = 0

print(f"\nRecording data for '{label}'.")
print("Press 'ESC' to stop recording and close.")

with mp_hands.create_from_options(options) as landmarker:

    with open(csv_filename, "a", newline="") as f:
        writer = csv.writer(f)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Flip frame for natural mirror view
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = landmarker.detect(mp_image)

            right_features = [0.0] * 73
            left_features = [0.0] * 73
            hands_detected = False

            if results.hand_landmarks:
                hands_detected = True
                for idx, hand_landmarks in enumerate(results.hand_landmarks):
                    # Draw manual circles just like in your predict_gestures.py
                    for landmark in hand_landmarks:
                        h, w, _ = frame.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                    handedness = results.handedness[idx][0].category_name
                    features = extract_features(hand_landmarks)
                    
                    if handedness == 'Right': right_features = features
                    elif handedness == 'Left': left_features = features

                final_row = right_features + left_features + [label]
                writer.writerow(final_row)
                frames_collected += 1

            status_color = (0, 255, 0) if hands_detected else (0, 0, 255)
            status_text = f"Recording: {label} | Frames: {frames_collected}" if hands_detected else "Waiting for hands..."
            
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, "Press 'ESC' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.imshow("Custom Collector", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()
print(f"\nSuccessfully collected {frames_collected} frames for label '{label}' into {csv_filename}.")