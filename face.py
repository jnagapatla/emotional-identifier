# face.py
# Handles face detection and landmark tracking using MediaPipe

import mediapipe as mp
import cv2
import numpy as np


class FaceDetector:
    """
    Detects faces in images and extracts facial landmarks using MediaPipe Face Mesh.
    Returns bounding boxes and landmark coordinates for detected faces.
    """

    def __init__(self) -> None:
        """
        Initializes MediaPipe Face Mesh for face detection and landmark tracking.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, image: np.ndarray) -> tuple[list[np.ndarray], list[dict]]:
        """
        Detects all faces in image and extracts landmarks.
        Returns tuple of (cropped_faces, face_data)
        """
        h, w, _ = image.shape
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        cropped_faces = []
        face_data = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract bounding box from landmarks
                x_coords = [lm.x for lm in face_landmarks.landmark]
                y_coords = [lm.y for lm in face_landmarks.landmark]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Add padding
                padding = 0.1
                x_min = max(0, int((x_min - padding) * w))
                x_max = min(w, int((x_max + padding) * w))
                y_min = max(0, int((y_min - padding) * h))
                y_max = min(h, int((y_max + padding) * h))

                cropped = image[y_min:y_max, x_min:x_max].copy()
                if cropped.size > 0:
                    cropped_faces.append(cropped)
                    face_data.append({
                        "landmarks": [(lm.x, lm.y) for lm in face_landmarks.landmark],
                        "bbox": (x_min, y_min, x_max, y_max)
                    })

        return cropped_faces, face_data

    def release(self) -> None:
        """
        Releases resources held by face detector.
        """
        self.face_mesh.close()


if __name__ == "__main__":
    exit("This script cannot be run on its own.\033[K")