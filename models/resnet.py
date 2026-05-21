"""
models/resnet.py
───────────────────────────────────────────────
Implementación adaptada de ResNet-18 para el benchmark.
Permite configurar el número de canales de entrada (in_channels) y
el número de clases de salida (num_classes).
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18(nn.Module):
    """
    Clase ResNet-18 adaptada.
    Instancia el modelo estándar de torchvision y ajusta las capas conv1 y fc
    para soportar dimensiones de entrada y salida arbitrarias.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10, pretrained: bool = False, **kwargs):
        """
        Inicializa ResNet18.

        Args:
            in_channels: Número de canales de entrada (ej. 3 para imágenes color). Por defecto 3.
            num_classes: Número de clases de salida. Por defecto 10.
            pretrained: Si se deben cargar pesos pre-entrenados de ImageNet.
        """
        super(ResNet18, self).__init__()

        # Para compatibilidad con otras convenciones de nomenclatura
        in_channels = kwargs.get("input_dim", in_channels)
        num_classes = kwargs.get("output_dim", num_classes)

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)

        # Ajustar primer convolución si los canales de entrada cambian
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels,
                self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=self.model.conv1.bias is not None
            )

        # Ajustar clasificador final
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Hacer que los sub-bloques sean directamente accesibles a nivel de clase para CFK
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        self.fc = self.model.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante (Forward pass).
        """
        return self.model(x)
