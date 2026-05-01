"""
train.py
--------
Boucle d'entraînement pour le CRNN (ResNet + BiLSTM + CTC).

Notes importantes sur CTC :
  - blank index = 0 ; les classes Tifinagh réelles sont indexées 1..NUM_CLASSES.
  - Les targets sont des séquences de longueur 1 (un caractère par image),
    mais on les concatène en un grand vecteur 1D `targets` de longueur B
    accompagné de `target_lengths` (ici tous 1).
  - `input_lengths` = T pour chaque exemple du batch (T = nb de timesteps en sortie
    du BiLSTM, ici T = 7).
"""
import os
import sys
import time
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from src.data_loader import get_train_loader, get_val_loader
from src.model import TifinaghCRNN, ctc_greedy_decode, count_parameters


def set_seed(seed: int = cfg.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_ctc_targets(labels: torch.Tensor) -> tuple:
    """
    Transforme un tenseur de labels (B,) en (targets, target_lengths) pour CTCLoss.

    Important : les labels ImageFolder sont 0..NUM_CLASSES-1, mais le blank CTC
    occupe l'indice 0. On décale donc tous les labels de +1 pour libérer le 0.
    """
    targets = (labels + 1).long()                            # (B,)
    target_lengths = torch.ones_like(targets, dtype=torch.long)
    return targets, target_lengths


def train_one_epoch(model, loader, criterion, optimizer, device, T):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        targets, target_lengths = to_ctc_targets(labels)

        optimizer.zero_grad()
        log_probs = model(images)                            # (T, B, V)
        b = images.size(0)
        input_lengths = torch.full((b,), T, dtype=torch.long, device=device)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()

        # Décodage greedy pour mesurer l'accuracy
        decoded = ctc_greedy_decode(log_probs, blank=cfg.CTC_BLANK_IDX)
        for pred_seq, gt in zip(decoded, labels.cpu().tolist()):
            # gt est dans 0..NUM_CLASSES-1 ; le décodage renvoie 1..NUM_CLASSES
            ok = (len(pred_seq) == 1 and pred_seq[0] == gt + 1)
            correct += int(ok)
        total += b

        running_loss += loss.item() * b
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, T):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        targets, target_lengths = to_ctc_targets(labels)

        log_probs = model(images)
        b = images.size(0)
        input_lengths = torch.full((b,), T, dtype=torch.long, device=device)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        decoded = ctc_greedy_decode(log_probs, blank=cfg.CTC_BLANK_IDX)
        for pred_seq, gt in zip(decoded, labels.cpu().tolist()):
            ok = (len(pred_seq) == 1 and pred_seq[0] == gt + 1)
            correct += int(ok)
        total += b
        running_loss += loss.item() * b
    return running_loss / total, correct / total


def plot_curves(history: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("CTC Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"],   label="val")
    axes[1].set_title("Sequence Accuracy (greedy decode)")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[INFO] Courbes sauvegardées dans {out_path}")


def main() -> None:
    set_seed()
    device = cfg.DEVICE
    print(f"[INFO] Device : {device}")

    train_loader = get_train_loader()
    val_loader   = get_val_loader()
    print(f"[INFO] Train batches : {len(train_loader)} | Val batches : {len(val_loader)}")

    model = TifinaghCRNN(num_classes=cfg.NUM_CLASSES,
                         lstm_hidden=cfg.LSTM_HIDDEN,
                         lstm_layers=cfg.LSTM_LAYERS,
                         lstm_dropout=cfg.LSTM_DROPOUT).to(device)
    print(f"[INFO] Paramètres : {count_parameters(model):,}")

    # Détecte la longueur de séquence T une fois pour toutes
    with torch.no_grad():
        dummy = torch.zeros(1, 1, cfg.IMG_SIZE, cfg.IMG_SIZE, device=device)
        T = model(dummy).size(0)
    print(f"[INFO] Longueur de séquence T = {T}")

    criterion = nn.CTCLoss(blank=cfg.CTC_BLANK_IDX, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.LEARNING_RATE,
                           weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    start = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, T)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device, T)
        scheduler.step(va_acc)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d}/{cfg.EPOCHS} | "
              f"train CTC {tr_loss:.4f} acc {tr_acc*100:.2f}% | "
              f"val CTC {va_loss:.4f} acc {va_acc*100:.2f}% | "
              f"lr {lr_now:.2e} | {time.time()-t0:.1f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), cfg.BEST_MODEL_PATH)
            print(f"  [+] Nouveau meilleur modèle sauvegardé ({va_acc*100:.2f}%)")

    torch.save(model.state_dict(), cfg.LAST_MODEL_PATH)
    print(f"\n[DONE] Entraînement terminé en {(time.time()-start)/60:.1f} min")
    print(f"[DONE] Meilleure validation accuracy : {best_val_acc*100:.2f}%")

    plot_curves(history, os.path.join(cfg.MODELS_DIR, "training_curves.png"))


if __name__ == "__main__":
    main()
