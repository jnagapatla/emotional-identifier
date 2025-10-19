# main.py
# Entry point for the emotion prediction backend server

from server import EmotionPredictionServer

print("\033c", end="", flush=True)
print("Facial Expression Prediction Backend", end="\033[K\n")
print("By Janav Nagapatla and Cayden Lim", end="\033[K\n")
print("Mentored by Dr Wong, Ms James, and Mr Gupta", end="\033[K\n")
print("All rights reserved", end="\033[K\n")

model_path = input("\n> Path to emotion model file (.h5): \a\033[K")

server = EmotionPredictionServer(model_path=model_path, port=5000)

print("> Server initialized successfully", end="\033[K\n")
print("> Detectors and analyzers loaded", end="\033[K\n")
print("\n> Starting server...", end="\033[K\n")

try:
    server.run()
except KeyboardInterrupt:
    print("\n> Shutting down server...", end="\033[K\n")
    server.shutdown()
    print("> Goodbye", end="\033[K\n")