# server.py
# Flask server handling frontend requests and backend processing

from flask import Flask, request, jsonify
from face import FaceDetector
from emotion import EmotionAnalyzer
from image import encode_image_to_base64, decode_base64_to_image, create_annotated_frame


class EmotionPredictionServer:
    """
    Flask server that processes video frames from frontend.
    Detects faces and analyzes emotions in real-time at 10 Hz.
    """

    def __init__(self, model_path: str = None, port: int = 5000) -> None:
        """
        Initializes the Flask server with face detector and emotion analyzer.
        """
        self.app = Flask(__name__)
        self.port = port

        self.face_detector = FaceDetector()
        self.emotion_analyzer = EmotionAnalyzer(model_path)

        self._setup_routes()

    def _setup_routes(self) -> None:
        """
        Sets up Flask routes for frontend communication.
        """

        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status": "healthy"}), 200

        @self.app.route('/process_frame', methods=['POST'])
        def process_frame():
            """
            Receives a frame from frontend, detects faces, analyzes emotions.
            Returns list of cropped faces with emotions.
            """
            try:
                data = request.json
                if 'frame' not in data:
                    return jsonify({"error": "No frame provided"}), 400

                frame_base64 = data['frame']
                frame = decode_base64_to_image(frame_base64)

                cropped_faces, face_data = self.face_detector.detect(frame)

                results = []
                for i, (face_image, fdata) in enumerate(zip(cropped_faces, face_data)):
                    emotion, confidence = self.emotion_analyzer.predict(face_image)

                    results.append({
                        "face_id": i,
                        "emotion": emotion,
                        "confidence": float(confidence),
                        "cropped_face": encode_image_to_base64(face_image),
                        "bbox": fdata["bbox"]
                    })

                return jsonify({
                    "success": True,
                    "num_faces": len(results),
                    "faces": results
                }), 200

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/process_frame_annotated', methods=['POST'])
        def process_frame_annotated():
            """
            Same as process_frame but returns full frame with annotations.
            """
            try:
                data = request.json
                if 'frame' not in data:
                    return jsonify({"error": "No frame provided"}), 400

                frame_base64 = data['frame']
                frame = decode_base64_to_image(frame_base64)
                original_frame = frame.copy()

                cropped_faces, face_data = self.face_detector.detect(frame)

                results = []
                for face_image, fdata in zip(cropped_faces, face_data):
                    emotion, confidence = self.emotion_analyzer.predict(face_image)
                    bbox = fdata["bbox"]

                    original_frame = create_annotated_frame(
                        original_frame, emotion, confidence, bbox
                    )

                    results.append({
                        "emotion": emotion,
                        "confidence": float(confidence),
                        "bbox": bbox
                    })

                return jsonify({
                    "success": True,
                    "num_faces": len(results),
                    "annotated_frame": encode_image_to_base64(original_frame),
                    "faces": results
                }), 200

            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def run(self, debug: bool = False) -> None:
        """
        Starts the Flask server.
        """
        print(f"Starting Emotion Prediction Server on http://localhost:{self.port}", flush=True)
        self.app.run(host='127.0.0.1', port=self.port, debug=debug, threaded=True)

    def shutdown(self) -> None:
        """
        Cleans up resources.
        """
        self.face_detector.release()


if __name__ == "__main__":
    exit("This script cannot be run on its own.\033[K")