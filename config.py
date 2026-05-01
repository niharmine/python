"""
Configuration centrale du projet Tifinagh (architecture CRNN : ResNet + BiLSTM + CTC).
Tous les hyperparamètres et chemins sont définis ici.
"""
import os
import torch

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR    = os.path.join(DATA_DIR, "train")
VAL_DIR      = os.path.join(DATA_DIR, "val")
TEST_DIR     = os.path.join(DATA_DIR, "test")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

# CSV bruts (utilisés une seule fois par prepare_data.py pour générer les images)
RAW_TRAIN_CSV = os.path.join(DATA_DIR, "train2020.csv")
RAW_TEST_CSV  = os.path.join(DATA_DIR, "test2020.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
IMG_SIZE     = 28           # images 28x28
NUM_CLASSES  = 33           # 33 caractères Tifinagh
SAMPLES_PER_CLASS = 585     # train2020.csv : 33 * 585 = 19305 lignes
VAL_RATIO    = 0.15         # 15% du train pour la validation

# Caractères Tifinagh (référence)
TIFINAGH_CHARS = [
    "ⵢⴰ", "ⵢⴰⴱ", "ⵢⴰⴳ", "ⵢⴰⴳⵯ", "ⵢⴰⴷ", "ⵢⴰⴹ", "ⵢⴰⴻ", "ⵢⴰⴼ",
    "ⵢⴰⴽ", "ⵢⴰⴽⵯ", "ⵢⴰⵀ", "ⵢⴰⵃ", "ⵢⴰⵄ", "ⵢⴰⵅ", "ⵢⴰⵇ", "ⵢⴰⵉ",
    "ⵢⴰⵊ", "ⵢⴰⵍ", "ⵢⴰⵎ", "ⵢⴰⵏ", "ⵢⴰⵓ", "ⵢⴰⵔ", "ⵢⴰⵕ", "ⵢⴰⵖ",
    "ⵢⴰⵙ", "ⵢⴰⵚ", "ⵢⴰⵛ", "ⵢⴰⵜ", "ⵢⴰⵟ", "ⵢⴰⵡ", "ⵢⴰⵢ", "ⵢⴰⵣ",
    "ⵢⴰⵥ"
]

# ---------------------------------------------------------------------------
# Architecture CRNN
# ---------------------------------------------------------------------------
LSTM_HIDDEN   = 128
LSTM_LAYERS   = 2
LSTM_DROPOUT  = 0.2
CTC_BLANK_IDX = 0           # blank token pour CTC ; classes réelles = 1..NUM_CLASSES

# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 64
NUM_WORKERS  = 2
EPOCHS       = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 5.0          # important pour stabiliser l'entraînement CTC + LSTM
SEED         = 42

# Augmentation
USE_AUGMENTATION = True

# Sauvegarde
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(MODELS_DIR, "last_model.pth")
SUBMISSION_PATH = os.path.join(PROJECT_ROOT, "submission.csv")
