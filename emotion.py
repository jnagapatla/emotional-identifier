# emotion.py
# Handles emotion classification from facial images

import tensorflow as tf
import numpy as np
import cv2


class EmotionAnalyzer:
    """
    Analyzes emotions from cropped face images.
    Uses a pre-trained deep learning model for classification.
    Supports multiple emotion categories: angry, disgust, fear, happy, neutral, sad, surprise.
    """

    EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    def __init__(self, model_path: str = None) -> None:
        """
        Initializes the emotion analyzer with a pre-trained model.
        If no path provided, attempts to load a default model or will require external setup.
        """
        self.model_path = model_path
        try:
            self.model = tf.keras.models.load_model(model_path) if model_path else self._load_default_model()
        except Exception as e:
            print(f"Warning: Could not load model - {e}")
            print("Ensure a trained emotion model is available at the specified path.")
            self.model = None

    def _load_default_model(self):
        """
        Attempts to load a default pre-trained model from TensorFlow Hub or similar.
        """
        try:
            return tf.keras.models.load_model('emotion_model.h5')
        except:
            return None

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocesses face image to model input format (48x48 grayscale).
        """
        face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48))
        face_array = np.array(face_resized, dtype=np.float32) / 255.0
        return np.expand_dims(face_array, axis=(0, -1))

    def predict(self, face_image: np.ndarray) -> tuple[str, float]:
        """
        Predicts the emotion present in the given face image.
        Returns tuple of (emotion_label, confidence_score).
        """
        if self.model is None:
            return "unknown", 0.0

        try:
            processed = self.preprocess(face_image)
            predictions = self.model.predict(processed, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_idx])
            emotion = self.EMOTIONS[emotion_idx]
            return emotion, confidence
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "error", 0.0


if __name__ == "__main__":
    exit("This script cannot be run on its own.\033[K")