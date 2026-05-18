"""
Tests para models/base_nn.py — BaseMLP
Valida la arquitectura, el forward pass, la separación feature_extractor/classifier,
y la capacidad de serialización del modelo.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from models.base_nn import BaseMLP


# ─────────────────────────────────────────────
# Inicialización y arquitectura
# ─────────────────────────────────────────────
class TestBaseMLP_Architecture:

    def test_default_dimensions(self):
        """Los valores por defecto deben ser input=2, hidden=16, output=2."""
        model = BaseMLP()
        assert model.feature_extractor[0].in_features == 2
        assert model.feature_extractor[0].out_features == 16
        assert model.feature_extractor[2].in_features == 16
        assert model.feature_extractor[2].out_features == 16
        assert model.classifier.in_features == 16
        assert model.classifier.out_features == 2

    def test_custom_dimensions(self):
        """El constructor debe respetar dimensiones personalizadas."""
        model = BaseMLP(input_dim=4, hidden_dim=32, output_dim=5)
        assert model.feature_extractor[0].in_features == 4
        assert model.feature_extractor[0].out_features == 32
        assert model.classifier.in_features == 32
        assert model.classifier.out_features == 5

    def test_feature_extractor_has_two_linear_layers(self):
        """El feature_extractor debe tener 2 capas lineales con ReLU."""
        model = BaseMLP()
        layers = list(model.feature_extractor)
        assert len(layers) == 4  # Linear, ReLU, Linear, ReLU
        assert isinstance(layers[0], torch.nn.Linear)
        assert isinstance(layers[1], torch.nn.ReLU)
        assert isinstance(layers[2], torch.nn.Linear)
        assert isinstance(layers[3], torch.nn.ReLU)

    def test_classifier_is_single_linear(self):
        """El classifier debe ser una sola capa lineal."""
        model = BaseMLP()
        assert isinstance(model.classifier, torch.nn.Linear)

    def test_parameter_count(self):
        """Verifica el número total de parámetros para la config por defecto (2->16->16->2)."""
        model = BaseMLP(input_dim=2, hidden_dim=16, output_dim=2)
        # Capa 1: 2*16 + 16 = 48
        # Capa 2: 16*16 + 16 = 272
        # Classifier: 16*2 + 2 = 34
        # Total = 354
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == 354


# ─────────────────────────────────────────────
# Forward pass
# ─────────────────────────────────────────────
class TestBaseMLP_Forward:

    def test_forward_output_shape(self):
        """Forward sin return_features devuelve logits con la forma correcta."""
        model = BaseMLP(input_dim=2, hidden_dim=16, output_dim=3)
        x = torch.randn(10, 2)
        logits = model(x)
        assert logits.shape == (10, 3)

    def test_forward_single_sample(self):
        """Debe funcionar con un solo ejemplo (batch_size=1)."""
        model = BaseMLP(input_dim=2, hidden_dim=16, output_dim=3)
        x = torch.randn(1, 2)
        logits = model(x)
        assert logits.shape == (1, 3)

    def test_forward_return_features_shapes(self):
        """Con return_features=True se devuelven logits Y features."""
        model = BaseMLP(input_dim=2, hidden_dim=16, output_dim=3)
        x = torch.randn(8, 2)
        logits, features = model(x, return_features=True)
        assert logits.shape == (8, 3)
        assert features.shape == (8, 16)

    def test_forward_return_features_consistency(self):
        """Logits con y sin return_features deben ser iguales para la misma entrada."""
        model = BaseMLP(input_dim=2, hidden_dim=16, output_dim=3)
        model.eval()
        x = torch.randn(5, 2)
        logits_only = model(x, return_features=False)
        logits_with, _ = model(x, return_features=True)
        torch.testing.assert_close(logits_only, logits_with)

    def test_forward_output_is_differentiable(self):
        """La salida debe mantener el grafo de gradientes."""
        model = BaseMLP(input_dim=2, hidden_dim=16, output_dim=3)
        x = torch.randn(4, 2)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        # Verificar que los gradientes existen
        for param in model.parameters():
            assert param.grad is not None


# ─────────────────────────────────────────────
# Serialización (save / load)
# ─────────────────────────────────────────────
class TestBaseMLP_Serialization:

    def test_save_and_load_state_dict(self, tmp_path):
        """El modelo se puede guardar y recargar correctamente."""
        model = BaseMLP(input_dim=2, hidden_dim=16, output_dim=3)
        path = tmp_path / "model.pth"
        torch.save(model.state_dict(), path)

        loaded = BaseMLP(input_dim=2, hidden_dim=16, output_dim=3)
        loaded.load_state_dict(torch.load(path, weights_only=True))

        # Verificar que los pesos son idénticos
        x = torch.randn(5, 2)
        model.eval()
        loaded.eval()
        torch.testing.assert_close(model(x), loaded(x))
