# Handwriting Digit Recognition

A real-time handwritten digit recognizer built with a Convolutional Neural Network (CNN) and a clean Tkinter UI. Draw any digit (0–9) on the canvas and the model predicts it instantly.

---

## Features

- Freehand drawing canvas with smooth stroke interpolation
- CNN trained on the MNIST dataset (60,000 samples)
- Confidence score displayed alongside the prediction
- Model saved after first training — subsequent runs load instantly
- Keyboard shortcuts for fast interaction
- Dark-themed UI (Catppuccin Mocha palette)

---

## Requirements

- Python 3.9+
- TensorFlow 2.x
- Pillow
- NumPy

Install all dependencies:

```bash
pip install tensorflow pillow numpy
```

---

## How to Run

```bash
python hand_writing.py
```

On first run, the model trains on MNIST (~1–2 minutes). The trained model is saved as `digit_cnn.keras`. Every subsequent run loads the saved model in seconds.

---

## Usage

| Action | How |
|---|---|
| Draw a digit | Click and drag on the canvas |
| Predict | Click **Predict** or press `P` |
| Clear canvas | Click **Clear** or press `C` |
| Quit | Click **Quit** or press `Q` |

> Draw inside the dashed guide box for best accuracy. Keep strokes proportional — avoid filling the entire canvas.

---

## Project Structure

```
cvpr_project/
├── hand_writing.py      # Main application
├── digit_cnn.keras      # Saved model (created after first run)
└── README.md            # This file
```

---

## How It Works

### 1. Model — CNN on MNIST

The model is a Convolutional Neural Network trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which contains 60,000 grayscale images of handwritten digits (28×28 pixels).

**Architecture:**

```
Input (28×28×1)
  → Conv2D(32, 3×3, ReLU)
  → MaxPooling2D
  → Conv2D(64, 3×3, ReLU)
  → MaxPooling2D
  → Flatten
  → Dense(128, ReLU)
  → Dropout(0.3)
  → Dense(10, Softmax)   ← outputs probability for each digit 0–9
```

- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 5, Batch size: 128
- Typical accuracy: ~99% on MNIST test set

Training runs in a background thread so the UI stays responsive.

### 2. Drawing Canvas

- Built with `tkinter.Canvas` (300×300 pixels)
- A mirrored `PIL.Image` (grayscale) tracks pixel data for the model
- Stroke interpolation fills gaps when drawing fast — each drag event draws intermediate circles between the last and current mouse position
- Brush radius: 8px

### 3. Preprocessing Pipeline

Before prediction, the drawn image is processed to match MNIST's format:

```
Raw canvas (300×300)
  → Gaussian Blur (radius=2)     — smooths sparse strokes
  → Bounding box crop            — isolates the drawn digit
  → Pad to square + 28px margin  — centers the digit
  → Resize to 28×28 (LANCZOS)    — matches model input size
  → Normalize to [0.0, 1.0]      — matches training data scale
```

This step is critical — MNIST digits are small, centered, and padded. Without this normalization the model sees something very different from its training data.

### 4. Prediction

The preprocessed 28×28 array is passed to the CNN. The output is a softmax probability vector of 10 values. The digit with the highest probability is shown, along with its confidence percentage.

---

## Tips for Best Accuracy

- Draw digits in a similar style to printed/handwritten numbers
- Use the dashed guide box as a boundary — don't draw too large
- Write one digit at a time and clear between predictions
- Avoid very thin or dotty strokes — draw with smooth continuous motion

---

## Extending to Letters (A–Z)

To support alphabets, the model can be retrained on the **EMNIST byclass** dataset (62 classes: 0–9, A–Z, a–z). Install the `emnist` package:

```bash
pip install emnist scipy
```

The EMNIST variant of this project uses the same CNN architecture with the output layer expanded to 62 classes and a deeper network with BatchNormalization for better generalization.
