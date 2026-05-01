# Tifinagh Project — Reconnaissance de caractères Tifinagh manuscrits (CRNN: ResNet + BiLSTM + CTC)

Projet de classification de caractères Tifinagh (alphabet amazigh) à partir d'images
manuscrites 28×28 en niveaux de gris. **33 classes**, ~19 300 échantillons d'entraînement.

**Architecture** : CRNN (Convolutional Recurrent Neural Network)
- **ResNet** backbone (mini-ResNet adapté à 28×28)
- **BiLSTM** (2 couches, hidden=128, bidirectionnel)
- **CTC** loss (Connectionist Temporal Classification)

Bien que les caractères soient isolés (séquence cible de longueur 1), cette
architecture est compatible avec une extension future à la reconnaissance
de mots/lignes Tifinagh.

## Structure du projet

```
tifinagh_project/
├── data/
│   ├── train2020.csv         <- à placer ici (CSV brut Kaggle)
│   ├── test2020.csv          <- à placer ici
│   ├── sample_submission.csv <- à placer ici
│   ├── train/  val/  test/   <- générés par prepare_data.py
├── models/
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── model.py              <- TifinaghCRNN (ResNet + BiLSTM + CTC)
│   ├── train.py              <- entraînement avec CTCLoss
│   └── evaluate.py           <- décodage CTC greedy + soumission
├── config.py
├── prepare_data.py
├── requirements.txt
└── README.md
```

## Utilisation

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Préparer les données
```bash
python prepare_data.py
```

### 3. Entraîner
```bash
python -m src.train
```

### 4. Évaluer & soumettre
```bash
python -m src.evaluate
```

## Détails de l'architecture

### Backbone ResNet
- Stem : Conv 1→32 (sans downsample initial, image trop petite)
- Layer 1 : 2 × BasicBlock, 32→64, downsample (2,2) → 14×14
- Layer 2 : 2 × BasicBlock, 64→128, downsample (2,1) → 7×14
- Layer 3 : 2 × BasicBlock, 128→256, downsample (2,1) → 3×14
- AdaptiveAvgPool → (1, 7) → séquence de **7 timesteps** de dimension **256**

### BiLSTM
- 2 couches, hidden=128, bidirectionnel, dropout=0.2
- Entrée : (T=7, B, 256) | Sortie : (7, B, 256)

### Tête CTC
- Linear(256 → 34) — 34 = 33 classes + 1 blank
- log_softmax sur la dim classes

### Convention CTC
- **blank index = 0**
- Classes Tifinagh réelles indexées **1..33**
- Targets de longueur 1 : un seul caractère par image
- Décodage greedy : argmax + collapse répétitions consécutives + suppression blank

## Hyperparamètres (config.py)

| Paramètre | Valeur |
|---|---|
| Batch size | 64 |
| Epochs | 30 |
| LR initial | 1e-3 (Adam) |
| Weight decay | 1e-4 |
| Grad clip | 5.0 |
| LSTM hidden | 128 |
| LSTM layers | 2 |
| LSTM dropout | 0.2 |
| Augmentation | rotation ±10°, translation ±10%, scale 0.9-1.1 |
| Val split | 15% stratifié |

## Différences avec la version v1 (CNN simple)

| | v1 (CNN) | v2 (CRNN) |
|---|---|---|
| Backbone | 3 conv blocks | ResNet (résiduel) |
| Tête | FC(1152→256→33) | BiLSTM + Linear(256→34) |
| Loss | CrossEntropy | CTCLoss |
| Décodage | argmax direct | greedy CTC (collapse+blank) |
| Extensible aux mots | ❌ | ✅ |
| Paramètres | ~443k | ~1.5M |
