"""Gesture classification model: architecture, training, and inference,
wrapped behind one class so callers don't need to know it's a Keras
Bidirectional-LSTM under the hood.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from .config import PATHS


def _build_architecture(input_shape: tuple[int, int], num_classes: int) -> Sequential:
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


class GestureClassifier:
    """Loads/holds a trained LSTM model + label encoder and exposes a
    single `predict` used by the inference server.
    """

    def __init__(self, model: tf.keras.Model, label_encoder: LabelEncoder):
        self._model = model
        self._label_encoder = label_encoder

    @classmethod
    def load(
        cls,
        model_path: Path = PATHS.gesture_model,
        label_encoder_path: Path = PATHS.label_encoder,
    ) -> "GestureClassifier":
        model = load_model(model_path)
        label_encoder = joblib.load(label_encoder_path)
        return cls(model, label_encoder)

    def warm_up(self, seq_len: int, num_features: int) -> None:
        """Runs a dummy forward pass so the first real prediction isn't
        slowed down by lazy graph construction.
        """
        dummy = tf.constant(np.zeros((1, seq_len, num_features), dtype=np.float32))
        self._model(dummy, training=False)

    def predict(self, sequence: list[list[float]]) -> tuple[str, float]:
        x = tf.constant(np.array(sequence, dtype=np.float32).reshape(1, len(sequence), -1))
        probs = self._model(x, training=False)[0].numpy()
        confidence = float(np.max(probs))
        label = self._label_encoder.inverse_transform([int(np.argmax(probs))])[0]
        return str(label), confidence

    @staticmethod
    def train(
        X: np.ndarray,
        y: np.ndarray,
        model_path: Path = PATHS.gesture_model,
        label_encoder_path: Path = PATHS.label_encoder,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> "GestureClassifier":
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_cat = to_categorical(y_encoded)
        joblib.dump(label_encoder, label_encoder_path)
        print(f"Saved label encoder to {label_encoder_path}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
        )

        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_encoded), y=y_encoded
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        model = _build_architecture(input_shape=X.shape[1:], num_classes=y_cat.shape[1])
        model.summary()

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ModelCheckpoint(str(model_path), monitor="val_accuracy", save_best_only=True),
        ]

        model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1,
        )

        loss, acc = model.evaluate(X_test, y_test, verbose=1)
        print(f"\nTest Accuracy: {acc:.4f}")
        print(f"Test Loss: {loss:.4f}")

        model.save(model_path)
        print(f"Saved model to {model_path}")

        return GestureClassifier(model, label_encoder)
