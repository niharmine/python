"""
model.py
--------
Architecture CRNN pour reconnaissance de caractères Tifinagh :

    image (1x28x28)
        |
    ResNet backbone  ->  feature map (C, H', W')
        |
    reshape  ->  séquence (W', batch, C * H')
        |
    BiLSTM (2 couches)  ->  (W', batch, 2 * hidden)
        |
    Linear (head)  ->  (W', batch, num_classes + 1)   <- "+1" = blank CTC
        |
    log_softmax  ->  log-probabilités utilisées par CTCLoss

Les targets sont des séquences de longueur 1 (un caractère par image).
La perte est nn.CTCLoss(blank=0, zero_infinity=True).
Les classes "réelles" sont indexées 1..num_classes ; la classe 0 est le blank.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


# ---------------------------------------------------------------------------
# Bloc résiduel basique (style ResNet)
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    """Bloc résiduel à 2 convolutions 3x3, identique à ResNet-18."""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Projection raccourcie si la forme change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


# ---------------------------------------------------------------------------
# Backbone ResNet adapté pour 28x28
# ---------------------------------------------------------------------------
class ResNetBackbone(nn.Module):
    """
    ResNet compact pour images 28x28 en niveaux de gris.

    On NE DOWNSAMPLE QUE EN HAUTEUR à partir d'un certain stade : on veut
    conserver une dimension "largeur" suffisante pour que le BiLSTM ait
    plusieurs timesteps en entrée (essentiel à CTC).

    Sortie : (B, 256, 1, 7) -> séquence de 7 timesteps de dimension 256.
    """

    def __init__(self):
        super().__init__()
        # stem : 28x28 -> 28x28 (pas de downsample initial, image trop petite)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Bloc 1 : 28x28, 32 -> 64 canaux, downsample (2,2) -> 14x14
        self.layer1 = nn.Sequential(
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 64, stride=1),
        )

        # Bloc 2 : 14x14 -> 7x14 (downsample H seulement, on garde W)
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=(2, 1)),
            BasicBlock(128, 128, stride=1),
        )

        # Bloc 3 : 7x14 -> 3x14 (downsample H seulement) puis -> 1x7 final
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=(2, 1)),
            BasicBlock(256, 256, stride=1),
        )

        # Pooling final : on écrase la hauteur à 1, et on réduit la largeur de 14 à 7
        self.final_pool = nn.AdaptiveAvgPool2d(output_size=(1, 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_pool(x)   # (B, 256, 1, 7)
        return x


# ---------------------------------------------------------------------------
# CRNN complet : ResNet + BiLSTM + tête CTC
# ---------------------------------------------------------------------------
class TifinaghCRNN(nn.Module):
    """
    Architecture complète :
      - ResNetBackbone -> (B, C, H, W) avec H=1, W=7
      - reshape vers (W, B, C*H) = (7, B, 256)
      - BiLSTM (2 couches, hidden=128, bidir) -> (7, B, 256)
      - Linear -> (7, B, num_classes + 1)
      - log_softmax sur la dernière dim (pour CTCLoss)

    Convention CTC :
      - blank index = 0
      - classes Tifinagh réelles : indices 1 .. num_classes
      - en sortie, le décodage argmax + collapse renvoie 1 caractère
    """

    def __init__(self,
                 num_classes: int = cfg.NUM_CLASSES,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 lstm_dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.blank_idx = 0
        # +1 pour le blank
        self.output_dim = num_classes + 1

        self.backbone = ResNetBackbone()

        # Le backbone sort (B, 256, 1, 7) -> features par timestep = 256*1 = 256
        self.feature_dim = 256

        self.bilstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=False,           # (T, B, F)
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # Sortie BiLSTM = 2 * hidden
        self.head = nn.Linear(2 * lstm_hidden, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 1, 28, 28)
        Returns:
            log_probs : (T, B, num_classes + 1) — directement utilisable par CTCLoss
        """
        # 1. backbone CNN -> (B, C, H, W)
        feat = self.backbone(x)                       # (B, 256, 1, 7)
        b, c, h, w = feat.shape

        # 2. reshape en séquence le long de la largeur
        #    -> (B, C*H, W) -> (W, B, C*H)
        feat = feat.view(b, c * h, w)                  # (B, 256, 7)
        feat = feat.permute(2, 0, 1).contiguous()      # (7, B, 256)

        # 3. BiLSTM -> (T, B, 2*hidden)
        rnn_out, _ = self.bilstm(feat)

        # 4. tête classification
        logits = self.head(rnn_out)                    # (T, B, num_classes+1)

        # 5. log_softmax pour CTCLoss
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


# ---------------------------------------------------------------------------
# Décodage CTC greedy : (T, B, V) -> liste de listes d'indices [1..num_classes]
# ---------------------------------------------------------------------------
@torch.no_grad()
def ctc_greedy_decode(log_probs: torch.Tensor, blank: int = 0) -> list:
    """
    Décodage CTC "best-path" :
      - argmax sur la dim classes
      - collapse des répétitions consécutives
      - suppression des blank

    Args:
        log_probs : (T, B, V)
    Returns:
        preds : liste de longueur B, chaque élément est une liste d'indices décodés.
                Pour des caractères isolés, la liste a typiquement 1 élément.
    """
    pred_idx = log_probs.argmax(dim=-1)         # (T, B)
    pred_idx = pred_idx.transpose(0, 1)          # (B, T)
    decoded = []
    for seq in pred_idx.cpu().tolist():
        out, prev = [], -1
        for s in seq:
            if s != prev and s != blank:
                out.append(s)
            prev = s
        decoded.append(out)
    return decoded


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TifinaghCRNN()
    print(model)
    print(f"\nParamètres entraînables : {count_parameters(model):,}")
    dummy = torch.randn(4, 1, 28, 28)
    out = model(dummy)
    print(f"Forme entrée : {dummy.shape}")
    print(f"Forme sortie log_probs : {out.shape}  (attendu : T x B x {cfg.NUM_CLASSES+1})")
    decoded = ctc_greedy_decode(out)
    print(f"Décodage greedy : {decoded}")
