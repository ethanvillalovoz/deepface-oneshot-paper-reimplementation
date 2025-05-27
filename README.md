<div align="center">

# 🧑‍🤝‍🧑 DeepFaceOneShot: Siamese Networks for One-shot Face Recognition

**Paper Reimplementation & Real-Time Demo**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[Original Paper (Koch et al., 2015)](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

</div>

---

## Overview

This project is a faithful, modern reimplementation of the influential 2015 paper  
*"Siamese Neural Networks for One-shot Image Recognition"* (Koch et al., 2015),  
adapted for deep facial recognition. Instead of Omniglot characters, the model is trained and evaluated on facial images—demonstrating how one-shot learning can identify people with just **one example per identity**.

---

## Features

- Siamese CNN for one-shot face recognition
- Real-time webcam verification app (Kivy)
- Modular code: custom layers, data pipeline, training notebook
- Training/evaluation plots and metrics
- Apple Silicon and cross-platform support

---

## Project Structure

```
deepface-oneshot-paper-reimplementation/
│
├── app/
│   ├── faceid.py         # Kivy GUI for real-time verification
│   └── layers.py         # Custom L1 distance layer
├── images/
│   └── training_curves.png
├── facial_verfication_siamese_network.ipynb  # Main training notebook
├── requirements.txt
└── README.md
```

---

## Environment Setup

**Apple Silicon (M1/M2/M3/M4):**

```bash
pip install tensorflow-macos tensorflow-metal
```

**All platforms (recommended):**

```bash
conda create -n deepface-oneshot python=3.9
conda activate deepface-oneshot
pip install -r requirements.txt
```

- For Apple Silicon, `requirements.txt` includes `tensorflow-macos` and `tensorflow-metal`.
- For other platforms, you may need to replace these with `tensorflow`.

---

## Quick Start

**Train and evaluate the model:**

```bash
jupyter notebook facial_verfication_siamese_network.ipynb
```

**Run the real-time verification app:**

```bash
cd app
python faceid.py
```

---

## Approach

- **Data:** Positive/anchor images from webcam, negatives from LFW dataset.
- **Model:** Siamese CNN with L1 distance and sigmoid output.
- **Training:** Binary cross-entropy loss, Adam optimizer, precision/recall metrics.
- **Evaluation:** Plots of loss, precision, recall; real-time webcam verification.

---

## Example Training Curve

![Training Curves](images/training_curves.png)

---

## Results

- Achieved near-perfect precision and recall on training data.
- Real-time verification works with webcam input.

---

## Challenges

- **Data Scarcity:** Collecting enough positive and negative pairs for robust training was non-trivial.
- **Overfitting:** The model quickly achieved perfect precision and recall on the training set, indicating possible overfitting. Addressed by evaluating on a separate test set and considering data augmentation.
- **Large Model Files:** Saving and pushing large model weights to GitHub required using `.gitignore` and cleaning git history.
- **Real-Time Testing:** Integrating OpenCV for real-time webcam verification required careful preprocessing and threshold tuning.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## References

- Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese Neural Networks for One-shot Image Recognition. [Paper link](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)