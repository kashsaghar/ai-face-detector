import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real_vs_fake", "real-vs-fake")

SEED = 42

CNN_IMG_SIZE = (128, 128)
TL_IMG_SIZE = (224, 224)

BATCH_SIZE = 32
EPOCHS_CNN = 15
EPOCHS_TL_HEAD = 8
EPOCHS_TL_FINETUNE = 5

# Set to an int (e.g. 40000) to limit the number of training images for
# faster local runs. Leave as None for the real Colab GPU training run.
TRAIN_SUBSET = None

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
METRICS_DIR = os.path.join(PROJECT_ROOT, "outputs", "metrics")

for _dir in (MODELS_DIR, FIGURES_DIR, METRICS_DIR):
    os.makedirs(_dir, exist_ok=True)
