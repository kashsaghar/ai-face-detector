import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from src.config import CNN_IMG_SIZE, FIGURES_DIR, METRICS_DIR, MODELS_DIR, TL_IMG_SIZE
from src.data import get_datasets

CLASS_NAMES = ["fake", "real"]

IMG_SIZE = {"cnn": CNN_IMG_SIZE, "mobilenet": TL_IMG_SIZE}
MODEL_PATH = {
    "cnn": os.path.join(MODELS_DIR, "cnn_best.keras"),
    "mobilenet": os.path.join(MODELS_DIR, "mobilenet_best.keras"),
}


def _collect_predictions(model, test_ds):
    # Images are stored as uint8 (not the default float32) -- 4x less RAM across a
    # 20k-image test set, since pixel values are already plain 0-255 integers here.
    y_true, y_prob, images = [], [], []
    for x, y in test_ds:
        p = model.predict(x, verbose=0).flatten()
        y_true.append(y.numpy().flatten())
        y_prob.append(p)
        images.append(x.numpy().astype(np.uint8))
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    images = np.concatenate(images)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_prob, y_pred, images


def evaluate_one(name):
    model = tf.keras.models.load_model(MODEL_PATH[name])
    _, _, test_ds, _ = get_datasets(IMG_SIZE[name])

    y_true, y_prob, y_pred, images = _collect_predictions(model, test_ds)

    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )
    report_txt = classification_report(y_true, y_pred, target_names=CLASS_NAMES)

    with open(os.path.join(METRICS_DIR, f"{name}_report.txt"), "w") as f:
        f.write(report_txt)
    with open(os.path.join(METRICS_DIR, f"{name}_report.json"), "w") as f:
        json.dump(report_dict, f, indent=2)
    print(report_txt)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} confusion matrix")
    fig.savefig(
        os.path.join(FIGURES_DIR, f"{name}_confusion_matrix.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{name} ROC curve")
    ax.legend()
    fig.savefig(
        os.path.join(FIGURES_DIR, f"{name}_roc.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    wrong_idx = np.where(y_pred != y_true)[0]
    n_show = min(8, len(wrong_idx))
    if n_show > 0:
        chosen = np.random.choice(wrong_idx, n_show, replace=False)
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        for ax, idx in zip(axes.flat, chosen):
            img = images[idx].astype(np.uint8) if images[idx].max() > 1.5 else images[idx]
            ax.imshow(np.clip(img / 255.0 if img.max() > 1.5 else img, 0, 1))
            ax.set_title(
                f"true={CLASS_NAMES[int(y_true[idx])]}\npred={CLASS_NAMES[int(y_pred[idx])]} ({y_prob[idx]:.2f})"
            )
            ax.axis("off")
        for ax in axes.flat[n_show:]:
            ax.axis("off")
        fig.savefig(
            os.path.join(FIGURES_DIR, f"{name}_errors.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    n_params = model.count_params()
    return {
        "model": name,
        "accuracy": report_dict["accuracy"],
        "precision_real": report_dict["real"]["precision"],
        "recall_real": report_dict["real"]["recall"],
        "f1_real": report_dict["real"]["f1-score"],
        "auc": roc_auc,
        "params": n_params,
        "fpr": fpr,
        "tpr": tpr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "mobilenet", "all"], required=True)
    args = parser.parse_args()

    names = ["cnn", "mobilenet"] if args.model == "all" else [args.model]
    results = []
    for name in names:
        t0 = time.time()
        results.append(evaluate_one(name))
        results[-1]["train_time_s"] = round(time.time() - t0, 1)

    if args.model == "all":
        rows = [
            {
                "model": r["model"],
                "accuracy": r["accuracy"],
                "precision_real": r["precision_real"],
                "recall_real": r["recall_real"],
                "f1_real": r["f1_real"],
                "auc": r["auc"],
                "params": r["params"],
                "eval_time_s": r["train_time_s"],
            }
            for r in results
        ]
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(METRICS_DIR, "model_comparison.csv"), index=False)

        headers = list(df.columns)
        md_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in df.iterrows():
            md_lines.append("| " + " | ".join(str(v) for v in row) + " |")
        with open(os.path.join(METRICS_DIR, "model_comparison.md"), "w") as f:
            f.write("\n".join(md_lines))

        fig, ax = plt.subplots(figsize=(6, 6))
        for r in results:
            ax.plot(r["fpr"], r["tpr"], label=f"{r['model']} (AUC = {r['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC comparison")
        ax.legend()
        fig.savefig(
            os.path.join(FIGURES_DIR, "roc_comparison.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print(df)


if __name__ == "__main__":
    main()
