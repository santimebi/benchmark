"""
models/base_nn.py
───────────────────────────────────────────────
Define la arquitectura del modelo base (BaseMLP) para el benchmark de
machine unlearning.

Arquitectura:
    input_dim → hidden_dim → ReLU → hidden_dim → ReLU → output_dim

El diseño separa intencionalmente el ``feature_extractor`` del
``classifier`` para facilitar técnicas de unlearning que necesiten
acceder a representaciones intermedias o reiniciar solo la última capa.
"""

import torch
import torch.nn as nn


class BaseMLP(nn.Module):
    """
    Modelo Base Multilayer Perceptron para el Benchmark de Unlearning.

    Arquitectura: input_dim -> hidden_dim -> ReLU -> hidden_dim -> ReLU -> output_dim
    Por defecto: 2 -> 16 -> 16 -> 2

    Attributes:
        feature_extractor (nn.Sequential): Capas de extracción de características.
        classifier (nn.Linear): Capa lineal de clasificación final.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, output_dim: int = 2):
        """
        Inicializa el modelo BaseMLP.

        Args:
            input_dim: Dimensión de la entrada (nº de features). Por defecto 2.
            hidden_dim: Nº de neuronas en las capas ocultas. Por defecto 16.
            output_dim: Nº de clases de salida. Por defecto 2.
        """
        super(BaseMLP, self).__init__()

        # Separamos el extractor de características del clasificador.
        # Esto es muy útil para métodos de unlearning que necesitan acceder
        # a representaciones intermedias o reiniciar solo la última capa.
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Capa lineal final
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Propagación hacia adelante (Forward pass).

        Args:
            x: Tensor de entrada con forma ``(batch_size, input_dim)``.
            return_features: Si es ``True``, devuelve tanto los logits finales
                             como las características intermedias.

        Returns:
            torch.Tensor: Logits de salida ``(batch_size, output_dim)``.
            Tuple[torch.Tensor, torch.Tensor]: Si ``return_features=True``,
                devuelve ``(logits, features)`` donde features tiene forma
                ``(batch_size, hidden_dim)``.
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features

        return logits
