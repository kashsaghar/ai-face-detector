import json
import os
import random

import cv2
import matplotlib.pyplot as plt

from src.config import DATA_DIR, FIGURES_DIR, METRICS_DIR, SEED

SPLITS = ["train", "valid", "test"]
CLASSES = ["fake", "real"]


def count_images():
    counts = {}
    for split in SPLITS:
        counts[split] = {}
        for cls in CLASSES:
            split_dir = os.path.join(DATA_DIR, split, cls)
            n = len(
                [
                    f
                    for f in os.listdir(split_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            counts[split][cls] = n
    return counts


def save_counts(counts):
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, "dataset_counts.json")
    with open(path, "w") as f:
        json.dump(counts, f, indent=2)
    print(f"Saved counts to {path}")
    return path


def plot_class_distribution(counts):
    x = range(len(SPLITS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    fake_counts = [counts[s]["fake"] for s in SPLITS]
    real_counts = [counts[s]["real"] for s in SPLITS]

    ax.bar([i - width / 2 for i in x], fake_counts, width, label="fake")
    ax.bar([i + width / 2 for i in x], real_counts, width, label="real")

    ax.set_xticks(list(x))
    ax.set_xticklabels(SPLITS)
    ax.set_ylabel("Number of images")
    ax.set_title("Class distribution per split")
    ax.legend()

    out_path = os.path.join(FIGURES_DIR, "class_distribution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_sample_grid():
    random.seed(SEED)

    samples = []
    for cls in CLASSES:
        split_dir = os.path.join(DATA_DIR, "train", cls)
        files = os.listdir(split_dir)
        chosen = random.sample(files, 8)
        samples.extend((os.path.join(split_dir, f), cls) for f in chosen)
    random.shuffle(samples)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, (path, label) in zip(axes.flat, samples):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")

    fig.suptitle("Random sample faces (train split)")
    out_path = os.path.join(FIGURES_DIR, "sample_grid.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    counts = count_images()
    print(json.dumps(counts, indent=2))
    save_counts(counts)
    plot_class_distribution(counts)
    plot_sample_grid()


if __name__ == "__main__":
    main()
