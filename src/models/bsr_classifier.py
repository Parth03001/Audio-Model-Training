"""
BSR Classifier Models
=====================
Two backbone options, both fine-tuned from pretrained weights:

  1. CNN14   — from PANNs (Pretrained Audio Neural Networks)
               Best default choice. Trained on AudioSet (2M clips).
               Already understands rattle, creak, mechanical transients.

  2. MobileNetV2 — lighter backbone for faster training / CPU inference
                   Good for prototyping or edge deployment.

Both share the same binary classification head and training interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────
# CNN14 Backbone (PANNs)
# ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Standard 2-conv block used in CNN14."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.bn2   = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, pool_size=(2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=pool_size)
        return x


class CNN14Backbone(nn.Module):
    """
    CNN14 architecture from PANNs.
    Input:  (B, 1, n_mels, T) — single-channel log-mel spectrogram
    Output: (B, 2048) embedding vector

    Reference: Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks
    for Audio Pattern Recognition", IEEE TASLP 2020.
    """

    def __init__(self):
        super().__init__()
        self.bn0    = nn.BatchNorm2d(64)
        self.conv0  = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.bn_in  = nn.BatchNorm2d(64)

        self.block1 = ConvBlock(64,   128)
        self.block2 = ConvBlock(128,  256)
        self.block3 = ConvBlock(256,  512)
        self.block4 = ConvBlock(512,  1024)
        self.block5 = ConvBlock(1024, 2048)
        self.block6 = ConvBlock(2048, 2048)

        self.fc = nn.Linear(2048, 2048, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        x = x.transpose(2, 3)       # (B, 1, T, n_mels)  — freq on last axis

        x = self.conv0(x)
        x = F.relu_(self.bn_in(x))

        x = self.block1(x, (2, 2))
        x = self.block2(x, (2, 2))
        x = self.block3(x, (2, 2))
        x = self.block4(x, (2, 2))
        x = self.block5(x, (2, 2))
        x = self.block6(x, (1, 1))

        # Global average + max pooling then concatenate
        x1 = torch.mean(x, dim=(2, 3))
        x2, _ = torch.max(x.view(x.size(0), x.size(1), -1), dim=2)
        x = x1 + x2                  # (B, 2048)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc(x))
        return x                      # (B, 2048)


def load_pretrained_cnn14(
    checkpoint_url: str = (
        "https://zenodo.org/record/3987831/files/"
        "Cnn14_mAP%3D0.431.pth"
    ),
    cache_dir: str = "checkpoints/pretrained",
) -> CNN14Backbone:
    """
    Download and load PANNs CNN14 pretrained weights.
    Weights are downloaded once and cached in cache_dir.
    """
    import os
    from pathlib import Path

    os.makedirs(cache_dir, exist_ok=True)
    local_path = Path(cache_dir) / "Cnn14_pretrained.pth"

    if not local_path.exists():
        print(f"Downloading CNN14 pretrained weights → {local_path}")
        torch.hub.download_url_to_file(checkpoint_url, str(local_path))

    model = CNN14Backbone()
    checkpoint = torch.load(local_path, map_location="cpu")

    # PANNs checkpoints are nested under 'model'
    state_dict = checkpoint.get("model", checkpoint)

    # Filter to only backbone keys (skip any classifier heads)
    backbone_keys = {
        k: v for k, v in state_dict.items()
        if not k.startswith("fc_audioset")
    }

    missing, unexpected = model.load_state_dict(backbone_keys, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}...")
    print("CNN14 pretrained weights loaded.")
    return model


# ─────────────────────────────────────────────
# BSR Classifier (backbone + binary head)
# ─────────────────────────────────────────────

class BSRClassifier(nn.Module):
    """
    Full BSR classification model.

    Args:
        backbone:         CNN14Backbone or any module returning (B, embed_dim) tensor
        embedding_dim:    Backbone output dimension (2048 for CNN14)
        dropout:          Dropout rate in head
        freeze_backbone:  If True, backbone gradients are frozen (phase 1 training)
    """

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int = 2048,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),          # binary logit (no sigmoid — use BCEWithLogitsLoss)
        )

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, 1, n_mels, T)
        returns: (B,) raw logits
        """
        embeddings = self.backbone(x)   # (B, 2048)
        logits     = self.head(embeddings).squeeze(1)  # (B,)
        return logits

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen — training head only")

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen — fine-tuning all layers")

    def get_param_groups(
        self,
        head_lr: float = 1e-3,
        backbone_lr: float = 1e-5,
    ) -> list[dict]:
        """Return separate parameter groups with different learning rates."""
        return [
            {"params": self.head.parameters(),     "lr": head_lr},
            {"params": self.backbone.parameters(), "lr": backbone_lr},
        ]


# ─────────────────────────────────────────────
# MobileNetV2 alternative (lighter / faster)
# ─────────────────────────────────────────────

def build_mobilenet_classifier(
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = True,
) -> BSRClassifier:
    """
    MobileNetV2 pretrained on ImageNet, adapted for single-channel mel-specs.
    Good for quick experiments or CPU deployment.
    """
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    backbone = mobilenet_v2(weights=weights)

    # Adapt first conv for 1-channel input (mel-spec has 1 channel)
    original_conv = backbone.features[0][0]
    backbone.features[0][0] = nn.Conv2d(
        1, original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False,
    )
    # Average the 3-channel pretrained weights into 1 channel
    if pretrained:
        with torch.no_grad():
            backbone.features[0][0].weight.data = (
                original_conv.weight.data.mean(dim=1, keepdim=True)
            )

    # Strip classifier head, keep feature extractor
    embedding_dim = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()

    return BSRClassifier(
        backbone       = backbone,
        embedding_dim  = embedding_dim,
        dropout        = dropout,
        freeze_backbone = freeze_backbone,
    )


# ─────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────

def build_model(
    backbone_name: str = "CNN14",
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.3,
) -> BSRClassifier:
    """
    Factory function — build model by name.

    Args:
        backbone_name: "CNN14" or "MobileNetV2"
        pretrained:    Load pretrained weights
        freeze_backbone: Freeze backbone initially (phase 1 training)
        dropout:       Head dropout rate

    Returns:
        BSRClassifier ready for training
    """
    if backbone_name == "CNN14":
        if pretrained:
            backbone = load_pretrained_cnn14()
        else:
            backbone = CNN14Backbone()
        return BSRClassifier(
            backbone        = backbone,
            embedding_dim   = 2048,
            dropout         = dropout,
            freeze_backbone = freeze_backbone,
        )

    elif backbone_name == "MobileNetV2":
        return build_mobilenet_classifier(
            pretrained      = pretrained,
            dropout         = dropout,
            freeze_backbone = freeze_backbone,
        )

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}. Choose CNN14 or MobileNetV2")
