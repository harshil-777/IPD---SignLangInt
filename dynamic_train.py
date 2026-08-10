import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

DATASET_DIR = "dynamic_dataset"
SEQ_LEN = 30
MODEL_PATH = "gesture_lstm.keras"
LABEL_ENCODER_PATH = "label_encoder.pkl"

def load_sequences(dataset_dir):
    X = []
    y = []

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    for label in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_path):
            continue

        for file_name in os.listdir(label_path):
            if file_name.endswith(".npy"):
                file_path = os.path.join(label_path, file_name)
                seq = np.load(file_path)

                if seq.shape[0] != SEQ_LEN:
                    print(f"Skipping {file_path}: expected {SEQ_LEN} frames, got {seq.shape[0]}")
                    continue

                X.append(seq)
                y.append(label)

    if not X:
        raise ValueError("No valid .npy sequence files found in the dataset.")

    return np.array(X, dtype=np.float32), np.array(y)

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train():
    print("Loading dynamic sequence dataset...")
    X, y = load_sequences(DATASET_DIR)

    print(f"Loaded {len(X)} samples")
    print(f"Input shape per sample: {X.shape[1:]}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_cat = to_categorical(y_encoded)

    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"Saved label encoder to {LABEL_ENCODER_PATH}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    model = build_model(input_shape=X.shape[1:], num_classes=y_cat.shape[1])
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test Loss: {loss:.4f}")

    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    train()