# DeepFake Detector 

This project loads a Keras/TensorFlow model (.h5) and performs deepfake detection on videos.

## Steps
1. Place your `model.h5` inside `model/`.
2. Install dependencies:
   pip install -r requirements.txt
3. Run:
   uvicorn app.main:app --reload
4. Open browser:
   http://localhost:8000
