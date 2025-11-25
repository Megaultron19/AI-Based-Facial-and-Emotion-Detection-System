AI-Based Facial and Emotion Detection System

A lightweight and efficient AI project that uses OpenCV for face detection and a Keras-based CNN model for emotion classification.
The main script video.py captures live webcam feed, detects faces using Haarcascades, preprocesses each face, and predicts emotions such as Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.

This project is perfect for academic demonstrations and AI mini-projects.

🚀 Features
👁️ Face Detection

1. Implemented using OpenCV Haarcascade Frontal Face Detector

2. Fast CPU-based detection suitable for real-time use

3. Detects multiple faces simultaneously

😊 Emotion Recognition

1. Uses a pretrained CNN model loaded from:

   a. model.json — model architecture

   b. model.h5 — model weights

Predictions mapped to labeled emotions:

{0: Angry, 
1: Disgusted,
2: Fearful,
3: Happy,
4: Neutral,
5: Sad,
6: Surprised}

🎥 Webcam Processing

Live video feed captured with:

cap = cv2.VideoCapture(0)

Real-time face detection + emotion prediction

Bounding boxes and emotion text drawn on video frames

Press 'a' to exit the window

📁 Project Structure
AI-Based-Facial-and-Emotion-Detection-System/
│
├── video.py               # Main script (webcam emotion detector)
├── model.json             # CNN architecture
├── model.h5               # Trained weights
├── haarcascades.zip       # Haarcascade XML files for face detection
└── README.md

🔧 Dependencies

The project uses:

OpenCV (cv2) for:

face detection

video capture

frame processing

NumPy for preprocessing

Keras for loading the trained model

from keras.models import model_from_json

⚠️ Disclaimer

For academic/demo purposes only

Not intended for production

No biometric data stored or transmitted

👨‍💻 Author

Harshit Singh
GitHub: Megaultron19
Email: harshitkatiyar2003@gmail.com

Aditya Rungta
Github: adityarungta2048
Email: adityarungta2048@gmail.com
