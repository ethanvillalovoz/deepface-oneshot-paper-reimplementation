# App Folder

This folder contains the code for the **real-time face verification application** built on top of the trained Siamese neural network.

## Contents

- **`faceid.py`**  
  A Kivy-based graphical application that uses your webcam to capture a live image and compares it against a set of verification images. It loads your trained Siamese model and provides instant feedback ("Verified" or "Unverified") based on the model's prediction.

- **`layers.py`**  
  Contains the custom `L1Dist` Keras layer, which is required to load and run the trained Siamese model.

## How It Works

1. **Live Webcam Feed:**  
   The app displays a live feed from your webcam.

2. **Verification:**  
   When you click the "Verify" button, the app captures an image from the webcam and compares it to all images in `application_data/verification_images` using the Siamese model.

3. **Result:**  
   If the model predicts a match above the set thresholds, the app displays "Verified"; otherwise, it shows "Unverified".

## Running the App

Make sure you have:
- Trained and saved your model as `model_weights.h5` in the root directory.
- Populated the `application_data/verification_images` folder with reference images.

Then run:

```bash
cd app
python faceid.py
```

## Requirements

- Python 3.7+
- Kivy
- OpenCV
- TensorFlow (same version as used for training)

Install dependencies (from your main environment):

```bash
pip install kivy opencv-python tensorflow
```

---

**Note:**  
The app is designed to work with the same preprocessing and model architecture as used in your main training notebook.  