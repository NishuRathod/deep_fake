# DeepFake Detector 

This project loads a Keras/TensorFlow model (.h5) and performs deepfake detection on videos.

# Project Brief — Deepfake Detection on Video

This project focuses on building an automated system that detects deepfake content in video streams using deep learning–based computer vision techniques. Deepfakes—synthetic media generated using advanced neural networks (GANs, autoencoders)—pose major risks to digital trust, misinformation, privacy, and security. The aim of this system is to accurately classify video frames as real or fake by analyzing facial regions and identifying manipulation artifacts.

# Objective

Develop a robust deepfake detection pipeline capable of:

Processing uploaded or streamed video.

Extracting frames and detecting faces.

Running each face through a trained deepfake-classification model.

Aggregating frame-level predictions into a final video-level verdict.

## Steps
1. Place your `model.h5` inside `model/`.
2. Install dependencies:
   pip install -r requirements.txt
3. Run:
   uvicorn app.main:app --reload
4. Open browser:
   http://localhost:8000

## Results

The model typically achieves high accuracy (80–90%+) depending on dataset quality and architecture. Misclassifications usually occur in low-light or heavily compressed videos.

Applications

Preventing misinformation and fake news

Social media content verification

Forensic investigations

Corporate and government security

# Conclusion

The project delivers an end-to-end deepfake detection solution designed for practical usage. By combining frame-level analysis, deep learning–based classification, and scalable inference, the system provides reliable detection of manipulated video content.