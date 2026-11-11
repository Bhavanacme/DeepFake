# Deepfake Model Directory

This directory stores your pre-trained Deep Learning models (e.g., CNN, MesoNet, EfficientNet, or Xception).

### How to use your trained model:

1. Save your trained Keras/TensorFlow model as:
   `model/deepfake_model.h5`
2. Install TensorFlow in your environment (if not already installed):
   ```bash
   pip install tensorflow
   ```
3. The application will automatically detect `deepfake_model.h5`, load it in `detection/ml_model.py`, and use it for real-time predictions!
4. If no `.h5` model is found, the system gracefully falls back to the Computer Vision Forensic Spectral & Texture Inconsistency Analyzer.
