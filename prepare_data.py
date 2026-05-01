"""
prepare_data.py
---------------
Convertit les CSV Kaggle (train2020.csv, test2020.csv) en arborescence d'images:

    data/
    ├── train/
    │   ├── class1/  ...  class33/
    ├── val/
    │   ├── class1/  ...  class33/
    └── test/
        └── *.png  (images sans label, nommées par ImageId)

Hypothèse vérifiée : train2020.csv contient 19 305 lignes = 33 classes × 585 échantillons
ordonnés séquentiellement (lignes 0-584 -> classe 0, 585-1169 -> classe 1, ...).

À exécuter UNE SEULE FOIS, depuis la racine du projet :
    python prepare_data.py
"""
import os
import sys
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# permet d'importer config.py situé à la racine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def csv_row_to_image(row: np.ndarray) -> Image.Image:
    """Reshape une ligne de pixels (784,) en image PIL 28x28 niveaux de gris."""
    arr = row.astype(np.uint8).reshape(cfg.IMG_SIZE, cfg.IMG_SIZE)
    return Image.fromarray(arr, mode="L")


def prepare_train_val():
    print(f"[INFO] Lecture de {cfg.RAW_TRAIN_CSV} ...")
    if not os.path.exists(cfg.RAW_TRAIN_CSV):
        raise FileNotFoundError(
            f"Place train2020.csv dans {cfg.DATA_DIR}/ avant d'exécuter ce script."
        )

    df = pd.read_csv(cfg.RAW_TRAIN_CSV)
    pixels = df.values  # (19305, 784)
    n = len(pixels)
    assert n == cfg.NUM_CLASSES * cfg.SAMPLES_PER_CLASS, (
        f"Nombre de lignes inattendu : {n} (attendu {cfg.NUM_CLASSES*cfg.SAMPLES_PER_CLASS})"
    )

    # Génère les labels à partir de l'index (labels séquentiels par bloc de 585)
    labels = np.repeat(np.arange(cfg.NUM_CLASSES), cfg.SAMPLES_PER_CLASS)

    # Split stratifié train/val
    idx = np.arange(n)
    train_idx, val_idx = train_test_split(
        idx,
        test_size=cfg.VAL_RATIO,
        stratify=labels,
        random_state=cfg.SEED,
    )

    # Crée les dossiers class01 ... class33 pour train et val
    # Padding sur 2 chiffres pour conserver l'ordre numérique == ordre alphabétique
    # (sinon ImageFolder trierait class1, class10, class11, ... class2, class20, ...)
    for split_dir in (cfg.TRAIN_DIR, cfg.VAL_DIR):
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        for c in range(cfg.NUM_CLASSES):
            os.makedirs(os.path.join(split_dir, f"class{c+1:02d}"), exist_ok=True)

    print(f"[INFO] Écriture de {len(train_idx)} images d'entraînement...")
    for i in train_idx:
        img = csv_row_to_image(pixels[i])
        cls = labels[i]
        img.save(os.path.join(cfg.TRAIN_DIR, f"class{cls+1:02d}", f"img_{i:05d}.png"))

    print(f"[INFO] Écriture de {len(val_idx)} images de validation...")
    for i in val_idx:
        img = csv_row_to_image(pixels[i])
        cls = labels[i]
        img.save(os.path.join(cfg.VAL_DIR, f"class{cls+1:02d}", f"img_{i:05d}.png"))

    print(f"[OK] Train : {len(train_idx)} | Val : {len(val_idx)}")


def prepare_test():
    print(f"[INFO] Lecture de {cfg.RAW_TEST_CSV} ...")
    if not os.path.exists(cfg.RAW_TEST_CSV):
        print("[WARN] test2020.csv introuvable, on saute la phase test.")
        return

    df = pd.read_csv(cfg.RAW_TEST_CSV)
    pixels = df.values

    if os.path.exists(cfg.TEST_DIR):
        shutil.rmtree(cfg.TEST_DIR)
    os.makedirs(cfg.TEST_DIR, exist_ok=True)

    print(f"[INFO] Écriture de {len(pixels)} images de test...")
    # ImageId commence à 1 (cf. sample_submission.csv)
    for i in range(len(pixels)):
        img = csv_row_to_image(pixels[i])
        img.save(os.path.join(cfg.TEST_DIR, f"{i+1}.png"))

    print(f"[OK] Test : {len(pixels)} images dans {cfg.TEST_DIR}")


if __name__ == "__main__":
    prepare_train_val()
    prepare_test()
    print("\n[DONE] Préparation terminée.")
