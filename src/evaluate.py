"""
evaluate.py
-----------
Charge le meilleur CRNN puis :
1. évalue sur le set de validation (greedy CTC decode + accuracy + matrice de confusion)
2. génère submission.csv au format Kaggle (ImageId, Label)

Usage :
    python -m src.evaluate
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from src.data_loader import get_val_loader, get_test_loader, get_class_names
from src.model import TifinaghCRNN, ctc_greedy_decode


def load_best_model() -> nn.Module:
    model = TifinaghCRNN(num_classes=cfg.NUM_CLASSES,
                         lstm_hidden=cfg.LSTM_HIDDEN,
                         lstm_layers=cfg.LSTM_LAYERS,
                         lstm_dropout=cfg.LSTM_DROPOUT).to(cfg.DEVICE)
    if not os.path.exists(cfg.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle introuvable : {cfg.BEST_MODEL_PATH}. Lance d'abord src/train.py"
        )
    state = torch.load(cfg.BEST_MODEL_PATH, map_location=cfg.DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def decode_to_class_id(decoded_seq: list, num_classes: int) -> int:
    """
    Convertit une séquence décodée par CTC en un seul ID de classe (0..num_classes-1).

    Convention :
      - le blank est 0, les classes réelles 1..num_classes en sortie modèle
      - on retire le décalage de +1 pour retomber sur 0..num_classes-1
      - si la séquence décodée est vide ou aberrante, on renvoie -1 (compté comme erreur)
    """
    if len(decoded_seq) == 1 and 1 <= decoded_seq[0] <= num_classes:
        return decoded_seq[0] - 1
    # Cas de bord : séquence vide -> on renvoie -1 ; séquence plus longue -> on prend le 1er
    if len(decoded_seq) >= 1 and 1 <= decoded_seq[0] <= num_classes:
        return decoded_seq[0] - 1
    return -1


@torch.no_grad()
def evaluate_validation(model: nn.Module) -> None:
    val_loader = get_val_loader()
    class_names = get_class_names()

    all_preds, all_labels = [], []
    for images, labels in val_loader:
        images = images.to(cfg.DEVICE)
        log_probs = model(images)
        decoded = ctc_greedy_decode(log_probs, blank=cfg.CTC_BLANK_IDX)
        for seq in decoded:
            all_preds.append(decode_to_class_id(seq, cfg.NUM_CLASSES))
        all_labels.extend(labels.numpy().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Pour le rapport, on remplace les -1 (décodage vide) par une classe invalide
    # mais qui sera comptée comme erreur dans confusion_matrix
    valid_mask = all_preds >= 0
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"[WARN] {n_invalid} séquence(s) vide(s) ou invalide(s) en décodage CTC.")
        # Les remplace par 0 pour permettre les métriques (compte comme erreur sauf classe 0)
        all_preds = np.where(valid_mask, all_preds, -1)

    acc = (all_preds == all_labels).mean()
    print(f"\n[RESULT] Validation accuracy (greedy CTC) : {acc*100:.2f}%\n")

    # Pour le rapport scikit-learn, ne garde que les prédictions valides
    keep = all_preds >= 0
    print("[INFO] Rapport de classification :")
    print(classification_report(all_labels[keep], all_preds[keep],
                                labels=list(range(cfg.NUM_CLASSES)),
                                target_names=class_names, digits=3,
                                zero_division=0))

    cm = confusion_matrix(all_labels[keep], all_preds[keep],
                          labels=list(range(cfg.NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(f"Matrice de confusion (val acc = {acc*100:.2f}%)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    out_path = os.path.join(cfg.MODELS_DIR, "confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[INFO] Matrice de confusion sauvegardée dans {out_path}")


@torch.no_grad()
def predict_test_and_submit(model: nn.Module) -> None:
    if not os.path.exists(cfg.TEST_DIR) or not os.listdir(cfg.TEST_DIR):
        print("[WARN] data/test/ vide — pas de soumission générée.")
        return

    test_loader = get_test_loader()
    image_ids, predictions = [], []

    for images, ids in test_loader:
        images = images.to(cfg.DEVICE)
        log_probs = model(images)
        decoded = ctc_greedy_decode(log_probs, blank=cfg.CTC_BLANK_IDX)
        for seq, image_id in zip(decoded, ids.tolist()):
            cls = decode_to_class_id(seq, cfg.NUM_CLASSES)
            # Si invalide, prédit 0 par défaut (rare en pratique)
            predictions.append(cls if cls >= 0 else 0)
            image_ids.append(image_id)

    df = pd.DataFrame({"ImageId": image_ids, "Label": predictions})
    df = df.sort_values("ImageId").reset_index(drop=True)
    df.to_csv(cfg.SUBMISSION_PATH, index=False)
    print(f"\n[OK] Soumission Kaggle sauvegardée dans {cfg.SUBMISSION_PATH}")
    print(df.head())


def main() -> None:
    print(f"[INFO] Device : {cfg.DEVICE}")
    model = load_best_model()
    evaluate_validation(model)
    predict_test_and_submit(model)


if __name__ == "__main__":
    main()
