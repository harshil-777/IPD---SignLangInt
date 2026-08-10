import os
import uuid
import glob
import random
import numpy as np

DATASET_DIR = "dynamic_dataset"
TARGET_SAMPLES = 500
SEQ_LEN = 30

def calculate_angle(v1, v2):
    v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle)

def recalculate_features(coords):
    # coords shape: (30, 21, 3)
    num_frames = coords.shape[0]
    all_features = []
    
    joint_groups = [
        (1, 2, 3), (2, 3, 4), (5, 6, 7), (6, 7, 8),
        (9, 10, 11), (10, 11, 12), (13, 14, 15),
        (14, 15, 16), (17, 18, 19), (18, 19, 20)
    ]
    
    for i in range(num_frames):
        frame_coords = coords[i] # (21, 3)
        
        # To perfectly match predict_gesture.py, re-center on the wrist and re-normalize max dist
        wrist = frame_coords[0]
        shifted_coords = frame_coords - wrist
        max_dist = np.max(np.linalg.norm(shifted_coords, axis=1))
        normalized_coords = shifted_coords / max_dist if max_dist > 0 else shifted_coords
        
        flattened = normalized_coords.flatten().tolist()
        
        angles = []
        for a, b, c in joint_groups:
            v1 = frame_coords[a] - frame_coords[b]
            v2 = frame_coords[c] - frame_coords[b]
            angles.append(calculate_angle(v1, v2))
            
        all_features.append(flattened + angles)
        
    return np.array(all_features, dtype=np.float32)

def augment_hand_sequence(coords):
    # coords shape: (30, 21, 3)
    num_frames = coords.shape[0]
    
    # A. Biomechanical Posture Variance (constant offset per joint)
    posture_offset = np.random.normal(loc=0.0, scale=0.01, size=(21, 3))
    coords = coords + posture_offset
    
    # B. Global Spatial Scaling
    scale = np.random.uniform(0.95, 1.05)
    coords = coords * scale
    
    # C. Sensor Jitter (frame-by-frame noise)
    jitter = np.random.normal(loc=0.0, scale=0.002, size=(num_frames, 21, 3))
    coords = coords + jitter
    
    return coords

def main():
    print(f"Balancing all classes in '{DATASET_DIR}' to {TARGET_SAMPLES} samples...")
    for label in os.listdir(DATASET_DIR):
        class_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(class_dir):
            continue
            
        files = glob.glob(os.path.join(class_dir, "*.npy"))
        count = len(files)
        
        if count == 0:
            print(f"Skipping {label} (0 files)")
            continue
            
        if count < TARGET_SAMPLES:
            needed = TARGET_SAMPLES - count
            print(f"Class '{label}' has {count} files. Synthesizing {needed} new files...")
            
            for _ in range(needed):
                base_file = random.choice(files)
                seq = np.load(base_file) # shape (30, 146)
                
                # Split features: Right (0:73), Left (73:146)
                right_features = seq[:, 0:73]
                left_features = seq[:, 73:146]
                
                # Extract coordinates (first 63 values)
                right_coords = right_features[:, 0:63].reshape(SEQ_LEN, 21, 3)
                left_coords = left_features[:, 0:63].reshape(SEQ_LEN, 21, 3)
                
                # Apply augmentation independently
                new_right_coords = augment_hand_sequence(right_coords)
                new_left_coords = augment_hand_sequence(left_coords)
                
                # Recalculate full features
                new_right_features = recalculate_features(new_right_coords)
                new_left_features = recalculate_features(new_left_coords)
                
                # Recombine
                new_seq = np.concatenate([new_right_features, new_left_features], axis=1) # (30, 146)
                
                # Save
                new_filename = f"{label}_synth_{uuid.uuid4().hex[:8]}.npy"
                new_filepath = os.path.join(class_dir, new_filename)
                np.save(new_filepath, new_seq)
        else:
            pass # Keep it clean

    print("\nDataset augmentation and balancing complete! All classes have at least 500 samples.")

if __name__ == '__main__':
    main()
