# image.py
# Utility functions for image processing and encoding

import cv2
import numpy as np
import base64


def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Converts OpenCV image (BGR) to base64 string for JSON transmission.
    """
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def decode_base64_to_image(base64_str: str) -> np.ndarray:
    """
    Converts base64 string back to OpenCV image (BGR).
    """
    buffer = base64.b64decode(base64_str)
    nparr = np.frombuffer(buffer, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def create_annotated_frame(image: np.ndarray,
                           emotion: str,
                           confidence: float,
                           bbox: tuple) -> np.ndarray:
    """
    Creates annotated image with emotion label and bounding box.
    """
    x_min, y_min, x_max, y_max = bbox
    annotated = image.copy()

    cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    label = f"{emotion} ({confidence:.2%})"
    cv2.putText(annotated, label, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return annotated


if __name__ == "__main__":
    exit("This script cannot be run on its own.\033[K")