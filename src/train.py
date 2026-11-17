import argparse
import os

import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import (
    BACKUP_DIR,
    CNN_IMG_SIZE,
    EPOCHS_CNN,
    EPOCHS_TL_FINETUNE,
    EPOCHS_TL_HEAD,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    SEED,
    TL_IMG_SIZE,
)
from src.data import get_datasets
from src.models import build_cnn, build_mobilenet, unfreeze_mobilenet_top

tf.random.set_seed(SEED)


def make_callbacks(name, stage=None):
    """Callbacks for one `model.fit()` call.

    `stage` names this fit call for BackupAndRestore (e.g. "head"/"finetune" for the
    two MobileNetV2 stages) so resuming one stage never picks up another stage's
    epoch/optimizer state. Re-running the same training command after a crash or a
    disconnect resumes from the last completed epoch automatically; `{name}_last.keras`
    is also saved every epoch as a plain, independently loadable fallback.
    """
    backup_dir = os.path.join(BACKUP_DIR, name if stage is None else f"{name}_{stage}")
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, f"{name}_best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, f"{name}_last.keras"),
            save_best_only=False,
        ),
        tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
        tf.keras.callbacks.CSVLogger(
            os.path.join(METRICS_DIR, f"{name}_history.csv"), append=True
        ),
    ]


def plot_curves(history_dicts, name):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in history_dicts:
        acc += h.get("accuracy", [])
        val_acc += h.get("val_accuracy", [])
        loss += h.get("loss", [])
        val_loss += h.get("val_loss", [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(acc, label="train")
    axes[0].plot(val_acc, label="val")
    axes[0].set_title(f"{name} accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="val")
    axes[1].set_title(f"{name} loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    out_path = os.path.join(FIGURES_DIR, f"{name}_training_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def train_cnn(epochs=None):
    train_ds, val_ds, _, class_names = get_datasets(CNN_IMG_SIZE)
    print("class_names:", class_names)

    model = build_cnn(lr=1e-3)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs or EPOCHS_CNN,
        callbacks=make_callbacks("cnn", stage="main"),
    )
    plot_curves([history.history], "cnn")


def train_mobilenet(epochs_head=None, epochs_finetune=None):
    train_ds, val_ds, _, class_names = get_datasets(TL_IMG_SIZE)
    print("class_names:", class_names)

    model = build_mobilenet(lr=1e-3)
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head or EPOCHS_TL_HEAD,
        callbacks=make_callbacks("mobilenet", stage="head"),
    )

    model = unfreeze_mobilenet_top(model, n_layers=30)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_finetune or EPOCHS_TL_FINETUNE,
        callbacks=make_callbacks("mobilenet", stage="finetune"),
    )

    plot_curves([history_head.history, history_finetune.history], "mobilenet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "mobilenet"], required=True)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.model == "cnn":
        train_cnn(epochs=args.epochs)
    else:
        train_mobilenet(epochs_head=args.epochs, epochs_finetune=args.epochs)


if __name__ == "__main__":
    main()
