import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf

from src.config import CNN_IMG_SIZE, MODELS_DIR, TL_IMG_SIZE
from src.gradcam import make_gradcam_overlay

MOBILENET_PATH = os.path.join(MODELS_DIR, "mobilenet_best.keras")
CNN_PATH = os.path.join(MODELS_DIR, "cnn_best.keras")

if os.path.exists(MOBILENET_PATH):
    MODEL = tf.keras.models.load_model(MOBILENET_PATH)
    IMG_SIZE = TL_IMG_SIZE
    MODEL_NAME = "mobilenet"
elif os.path.exists(CNN_PATH):
    MODEL = tf.keras.models.load_model(CNN_PATH)
    IMG_SIZE = CNN_IMG_SIZE
    MODEL_NAME = "cnn"
else:
    raise FileNotFoundError(
        "No trained model found in models/. Train a model first "
        "(python -m src.train --model cnn|mobilenet) or download the "
        "Colab-trained .keras files into models/."
    )

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")

DESCRIPTION = (
    "Upload a face photo to see whether it looks like a real photo or an "
    "AI-generated (StyleGAN) face, with a Grad-CAM explanation of which "
    "regions the model focused on.\n\n"
    "Trained on StyleGAN faces; works best on front-facing face photos."
)


def predict(image):
    if image is None:
        return None, None

    img = cv2.resize(image, IMG_SIZE).astype(np.float32)
    prob_real = float(MODEL.predict(np.expand_dims(img, 0), verbose=0)[0, 0])

    labels = {"Real": prob_real, "AI-Generated": 1 - prob_real}

    _, overlay = make_gradcam_overlay(MODEL, img)
    return labels, overlay


def build_examples():
    if not os.path.isdir(EXAMPLES_DIR):
        return []
    return [
        os.path.join(EXAMPLES_DIR, f)
        for f in sorted(os.listdir(EXAMPLES_DIR))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload a face image"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Image(label="Grad-CAM overlay"),
    ],
    title="Real vs AI-Generated Face Detection",
    description=DESCRIPTION,
    examples=build_examples() or None,
)

if __name__ == "__main__":
    demo.launch(share=False)
