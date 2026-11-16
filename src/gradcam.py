import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.config import (
    CNN_IMG_SIZE,
    DATA_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    SEED,
    TL_IMG_SIZE,
)

IMG_SIZE = {"cnn": CNN_IMG_SIZE, "mobilenet": TL_IMG_SIZE}
MODEL_PATH = {
    "cnn": os.path.join(MODELS_DIR, "cnn_best.keras"),
    "mobilenet": os.path.join(MODELS_DIR, "mobilenet_best.keras"),
}


def _find_last_conv_layer(model):
    if model.name == "mobilenet_v2_tl":
        return "Conv_1"
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model")


def make_gradcam_overlay(model, img_array, layer_name=None):
    """img_array: a single image, shape (H, W, 3), values in [0, 255].

    Returns (heatmap, overlay) as uint8 RGB arrays sized to the input image.
    """
    if layer_name is None:
        layer_name = _find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(layer_name).output, model.output]
    )

    x = np.expand_dims(img_array.astype("float32"), axis=0)
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(x)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    h, w = img_array.shape[0], img_array.shape[1]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    base_img = img_array.astype(np.uint8)
    overlay = cv2.addWeighted(base_img, 0.6, heatmap_color, 0.4, 0)

    return heatmap_color, overlay


def _load_image(path, img_size):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    return img.astype(np.float32)


def run(name):
    model = tf.keras.models.load_model(MODEL_PATH[name])
    img_size = IMG_SIZE[name]

    random.seed(SEED)
    samples = []
    for cls in ("real", "fake"):
        folder = os.path.join(DATA_DIR, "test", cls)
        files = random.sample(os.listdir(folder), 3)
        samples.extend((os.path.join(folder, f), cls) for f in files)

    fig, axes = plt.subplots(len(samples), 3, figsize=(10, 3.3 * len(samples)))
    for row, (path, true_label) in enumerate(samples):
        img = _load_image(path, img_size)
        heatmap, overlay = make_gradcam_overlay(model, img)

        prob_real = float(model.predict(np.expand_dims(img, 0), verbose=0)[0, 0])
        pred_label = "real" if prob_real >= 0.5 else "fake"
        confidence = prob_real if pred_label == "real" else 1 - prob_real

        axes[row, 0].imshow(img.astype(np.uint8))
        axes[row, 0].set_title(f"original ({true_label})")
        axes[row, 1].imshow(heatmap)
        axes[row, 1].set_title("Grad-CAM heatmap")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f"pred={pred_label} ({confidence:.2f})")
        for ax in axes[row]:
            ax.axis("off")

    out_path = os.path.join(FIGURES_DIR, f"{name}_gradcam.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "mobilenet"], required=True)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
