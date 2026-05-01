"""
data_loader.py
--------------
Définit les transformations et fournit les DataLoader train / val / test.

L'arborescence attendue (créée par prepare_data.py) :
    data/train/class1/*.png  ...  data/train/class33/*.png
    data/val/class1/*.png    ...  data/val/class33/*.png
    data/test/*.png          (images de test sans label, nommées <ImageId>.png)
"""
import os
import sys
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

# import config depuis la racine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------
def get_train_transform() -> transforms.Compose:
    """Augmentation légère : adaptée à des caractères manuscrits 28x28."""
    if cfg.USE_AUGMENTATION:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                fill=255,  # fond blanc
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    return get_eval_transform()


def get_eval_transform() -> transforms.Compose:
    """Transform pour validation et test : pas d'augmentation."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


# ---------------------------------------------------------------------------
# Dataset spécifique pour le dossier test/ (images sans label, ordonnées par ImageId)
# ---------------------------------------------------------------------------
class TifinaghTestDataset(Dataset):
    """Charge le contenu de data/test/ ; chaque image est nommée <ImageId>.png."""

    def __init__(self, test_dir: str = cfg.TEST_DIR, transform=None):
        self.test_dir = test_dir
        self.transform = transform or get_eval_transform()

        files = [f for f in os.listdir(test_dir) if f.lower().endswith(".png")]
        # Trie numériquement par ImageId (1.png, 2.png, ...)
        files.sort(key=lambda f: int(os.path.splitext(f)[0]))
        self.files: List[str] = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        fname = self.files[idx]
        image_id = int(os.path.splitext(fname)[0])
        path = os.path.join(self.test_dir, fname)
        img = Image.open(path).convert("L")
        img = self.transform(img)
        return img, image_id


# ---------------------------------------------------------------------------
# Constructeurs de DataLoaders
# ---------------------------------------------------------------------------
def get_train_loader() -> DataLoader:
    dataset = datasets.ImageFolder(cfg.TRAIN_DIR, transform=get_train_transform())
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(cfg.DEVICE.type == "cuda"),
    )


def get_val_loader() -> DataLoader:
    dataset = datasets.ImageFolder(cfg.VAL_DIR, transform=get_eval_transform())
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(cfg.DEVICE.type == "cuda"),
    )


def get_test_loader() -> DataLoader:
    dataset = TifinaghTestDataset(transform=get_eval_transform())
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(cfg.DEVICE.type == "cuda"),
    )


def get_class_names() -> List[str]:
    """Retourne la liste des classes telles que vues par ImageFolder (ordre alphabétique)."""
    dataset = datasets.ImageFolder(cfg.TRAIN_DIR)
    return dataset.classes
