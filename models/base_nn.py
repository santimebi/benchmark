import torch
import torch.nn as nn

class BaseMLP(nn.Module):
    """
    Modelo Base Multilayer Perceptron para el Benchmark de Unlearning.
    Arquitectura: input_dim -> hidden_dim -> ReLU -> hidden_dim -> ReLU -> output_dim
    Por defecto: 2 -> 16 -> 16 -> 2
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, output_dim: int = 2):
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
            x (torch.Tensor): Tensor de entrada.
            return_features (bool): Si es True, devuelve tanto los logits finales 
                                    como las características intermedias.
        Returns:
            torch.Tensor o Tuple[torch.Tensor, torch.Tensor]
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        
        return logits
