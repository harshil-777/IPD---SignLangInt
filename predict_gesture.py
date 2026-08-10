import os
import csv
import cv2
import joblib
import textwrap
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from google import genai

# -------------------- CONFIG --------------------
DYNAMIC_MODEL_PATH = "gesture_lstm.keras"
LABEL_ENCODER_PATH = "label_encoder.pkl"
HAND_LANDMARKER_TASK = "hand_landmarker.task"

SEQ_LEN = 30
DYNAMIC_CONF_THRESHOLD = 0.50
DYNAMIC_STABLE_WINDOW = 4

USE_GEMINI = True
GEMINI_API_KEY = ""  # put your key here

# -------------------- OPTIONAL GEMINI --------------------
client = None
if USE_GEMINI and GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------- LOAD MODELS --------------------
dynamic_model = None
label_encoder = None
if os.path.exists(DYNAMIC_MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
    dynamic_model = tf.keras.models.load_model(DYNAMIC_MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("Unified sequence model loaded.")
else:
    print("Unified sequence model not found. Please run dynamic_train.py first.")

# -------------------- MEDIA PIPE --------------------
mp_hands = mp.tasks.vision.HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_TASK),
    num_hands=2
)

# -------------------- FIR FLOW --------------------
fir_sequence = [
    "Step 1: Your Name? (Spell it)",
    "Step 2: What Happened? (Theft, Attack, Kidnap)",
    "Step 3: When did it happen? (HH:MM)",
    "Step 4: Where did it happen? (Location)",
    "Step 5: Any vehicles involved? (Car, Bike, Truck)",
    "Step 6: Describe the actions (Push, Pull, Drink, Drive)",
    "Step 7: Stolen items or injuries?",
    "Step 8: Describe the accused.",
    "Step 9: Any evidence or witnesses?",
    "Step 10: Any other information?"
]
current_step = 0
fir_data = {i: [] for i in range(len(fir_sequence))}

# -------------------- HELPERS --------------------
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

    return flattened_coords + angles  # 73 features per hand

def majority_vote(buffer):
    if not buffer:
        return ""
    non_empty = [x for x in buffer if x]
    if not non_empty:
        return ""
    best = max(set(non_empty), key=non_empty.count)
    return best

def predict_dynamic(sequence):
    if dynamic_model is None or label_encoder is None or len(sequence) < SEQ_LEN:
        return "", 0.0

    x = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
    probs = dynamic_model.predict(x, verbose=0)[0]
    conf = float(np.max(probs))
    pred_idx = int(np.argmax(probs))
    pred = label_encoder.inverse_transform([pred_idx])[0]
    return pred, conf

def append_word(sentence_words, stable_prediction, last_added_word):
    if not stable_prediction or stable_prediction == last_added_word:
        return sentence_words, last_added_word

    if len(stable_prediction) == 1:
        if sentence_words and len(sentence_words[-1]) <= 10 and len(last_added_word) == 1:
            sentence_words[-1] += stable_prediction
        else:
            sentence_words.append(stable_prediction)
    else:
        sentence_words.append(stable_prediction)

    return sentence_words, stable_prediction

# -------------------- MAIN --------------------
dynamic_prediction_buffer = deque(maxlen=DYNAMIC_STABLE_WINDOW)
frame_history = deque(maxlen=SEQ_LEN)

sentence_words = []
last_added_word = ""
stable_prediction = ""
generated_sentence = ""
no_hand_counter = 0

prev_features = None

cap = cv2.VideoCapture(0)

with mp_hands.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = landmarker.detect(mp_image)

        stable_prediction = ""

        if results.hand_landmarks:
            no_hand_counter = 0

            right_features = [0.0] * 73
            left_features = [0.0] * 73

            for idx, hand_landmarks in enumerate(results.hand_landmarks):
                for landmark in hand_landmarks:
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                handedness = results.handedness[idx][0].category_name
                features = extract_features(hand_landmarks)

                if handedness == "Right":
                    right_features = features
                elif handedness == "Left":
                    left_features = features

            curr = np.array(right_features + left_features, dtype=np.float32)
            
            # Feature Smoothing (EMA) to reduce camera/MediaPipe jitter
            if prev_features is not None:
                alpha = 0.5
                curr = alpha * curr + (1 - alpha) * prev_features

            frame_features = curr.tolist()
            frame_history.append(frame_features)
            prev_features = curr

            # Unified Prediction
            if dynamic_model is not None:
                if len(frame_history) == SEQ_LEN:
                    dyn_pred, dyn_conf = predict_dynamic(list(frame_history))
                    if dyn_conf >= DYNAMIC_CONF_THRESHOLD:
                        dynamic_prediction_buffer.append(dyn_pred)
                        common = majority_vote(list(dynamic_prediction_buffer))
                        if common and dynamic_prediction_buffer.count(common) >= DYNAMIC_STABLE_WINDOW:
                            stable_prediction = common
                    else:
                        dynamic_prediction_buffer.append("")
                else:
                    stable_prediction = ""

        else:
            no_hand_counter += 1
            if prev_features is not None and no_hand_counter <= 10:
                frame_history.append(prev_features.tolist())
            elif no_hand_counter > 10:
                dynamic_prediction_buffer.clear()
                frame_history.clear()
                prev_features = None
                stable_prediction = ""
                last_added_word = ""

        sentence_words, last_added_word = append_word(sentence_words, stable_prediction, last_added_word)

        # ---------------- UI HUD DESIGN ----------------
        h, w, _ = frame.shape
        
        # (Removed solid black background to keep camera feed full resolution)
        
        # Top Header & Steps
        cv2.putText(frame, "INCIDENT REPORT - SMART SIGN LANGUAGE SYSTEM", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
        
        # Current Step Highlight
        cv2.putText(frame, fir_sequence[current_step], (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2) # Gold/Amber

        # Predictions & Input
        cv2.putText(frame, f"Prediction: {stable_prediction}", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)

        display_words = " ".join(sentence_words[-6:])
        cv2.putText(frame, f"Current Input: {display_words}", (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 150), 2)

        # (Removed solid black background for bottom controls)

        controls = "[ENTER] Generate | [N] Next | [B] Back | [C] Clear | [D]/[BACKSPACE] Delete | [ESC] Exit"
        cv2.putText(frame, controls, (10, h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Draw Generated Incident Box
        if generated_sentence:
            cv2.rectangle(frame, (5, 170), (w - 5, h - 40), (40, 40, 40), -1)
            wrapped_text = textwrap.wrap(f"FINAL INCIDENT NARRATIVE:", width=60)
            wrapped_text.extend(textwrap.wrap(generated_sentence, width=60))
            
            y_offset = 195
            for i, line in enumerate(wrapped_text):
                color = (0, 255, 255) if i == 0 else (220, 220, 220)
                cv2.putText(frame, line, (15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                y_offset += 25

        cv2.imshow("Unified Sign Language System", frame)
        key = cv2.waitKey(1) & 0xFF

        # ---------------- KEYBOARD CONTROLS ----------------
        if key == ord('n'):
            fir_data[current_step] = list(sentence_words)
            if current_step < len(fir_sequence) - 1:
                current_step += 1
            sentence_words = list(fir_data[current_step])
            generated_sentence = ""

        elif key == ord('b'):
            fir_data[current_step] = list(sentence_words)
            if current_step > 0:
                current_step -= 1
            sentence_words = list(fir_data[current_step])
            generated_sentence = ""

        elif key == ord('c'):
            sentence_words = []
            last_added_word = ""
            generated_sentence = ""
            dynamic_prediction_buffer.clear()
            frame_history.clear()
            prev_features = None

        elif key == 8 or key == 127 or key == ord('d'):
            if sentence_words:
                if len(last_added_word) == 1 and len(sentence_words[-1]) > 1:
                    sentence_words[-1] = sentence_words[-1][:-1]
                else:
                    sentence_words.pop()

                last_added_word = "DELETED_LOCK"
                dynamic_prediction_buffer.clear()
                stable_prediction = ""

        elif key == 13:
            fir_data[current_step] = list(sentence_words)

            compiled_inputs = ""
            for i, question in enumerate(fir_sequence):
                if not fir_data[i]:
                    signs = "N/A"
                elif i == 2:
                    raw_time = "".join(fir_data[i]).replace(" ", "").replace(":", "")
                    try:
                        if len(raw_time) >= 2:
                            hh = int(raw_time[:2])
                            mm = raw_time[2:] if len(raw_time) > 2 else "00"
                        else:
                            hh = int(raw_time)
                            mm = "00"

                        period = "AM" if hh < 12 else "PM"
                        display_hh = hh if hh <= 12 else hh - 12
                        display_hh = 12 if display_hh == 0 else display_hh

                        signs = f"{display_hh}:{mm.ljust(2, '0')} {period}"
                    except ValueError:
                        signs = " ".join(fir_data[i])
                else:
                    signs = " ".join(fir_data[i])

                compiled_inputs += f"{question}: {signs}\n"

            if client is not None:
                try:
                    prompt = f"""
You are an expert Sign Language Translator drafting an Incident Report narrative.
A deaf complainant has provided the following signs across multiple questions.
"N/A" means the question was skipped.

{compiled_inputs}

Rules:
1. Single letters (e.g., K E N I L) must be combined into names.
2. Write ONE cohesive, formal, and legally appropriate paragraph summarizing the entire incident based on the inputs provided.
3. Do not invent details that were not signed. Ignore "N/A" entries.
4. Output ONLY the final formal paragraph. No greetings or bullet points.
"""
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
                    generated_sentence = response.text.strip()
                except Exception as e:
                    generated_sentence = f"Error generating Incident Report: {e}"
            else:
                generated_sentence = "Gemini API key not configured."

        elif key == 27:
            break

cap.release()
cv2.destroyAllWindows()
