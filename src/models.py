import tensorflow as tf
from tensorflow.keras import layers, models

from src.config import CNN_IMG_SIZE, TL_IMG_SIZE


def _compile(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_cnn(lr=1e-3):
    inputs = layers.Input(shape=CNN_IMG_SIZE + (3,))
    x = layers.Rescaling(1.0 / 255)(inputs)

    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="custom_cnn")
    return _compile(model, lr)


MOBILENET_HEAD_LAYERS = 3  # GlobalAveragePooling2D, Dropout, Dense


def build_mobilenet(lr=1e-3):
    inputs = layers.Input(shape=TL_IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    base = tf.keras.applications.MobileNetV2(
        input_tensor=x, include_top=False, weights="imagenet"
    )
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    # Built functionally from base.input/base.output, so all of MobileNetV2's
    # layers (including "Conv_1") end up flattened directly into this model's
    # layer graph rather than nested inside a sub-model -- this keeps
    # model.get_layer("Conv_1") working for Grad-CAM.
    model = models.Model(inputs, outputs, name="mobilenet_v2_tl")
    return _compile(model, lr)


def unfreeze_mobilenet_top(model, n_layers=30):
    base_layers = model.layers[:-MOBILENET_HEAD_LAYERS]
    for layer in base_layers:
        layer.trainable = True
    for layer in base_layers[:-n_layers]:
        layer.trainable = False
    for layer in base_layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return model


if __name__ == "__main__":
    import os

    from src.config import METRICS_DIR

    cnn = build_cnn()
    mobilenet = build_mobilenet()

    for name, model in (("cnn", cnn), ("mobilenet", mobilenet)):
        summary_lines = []
        model.summary(print_fn=lambda line: summary_lines.append(line))
        out_path = os.path.join(METRICS_DIR, f"{name}_summary.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
        print(f"Saved {out_path}")

    assert mobilenet.get_layer("Conv_1") is not None
    print("MobileNetV2 Conv_1 layer reachable: OK")
