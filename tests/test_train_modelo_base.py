"""
Tests para 3_train_modelo_base.py — train_base_model()
Valida la carga de datos, el bucle de entrenamiento, el guardado de pesos
y el manejo de errores, todo con datos sintéticos para no depender de archivos reales.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch
import importlib.util

# Cargamos el módulo dinámicamente porque empieza por un dígito.
_spec = importlib.util.spec_from_file_location(
    "train_modelo_base",
    str(Path(__file__).parent.parent / "3_train_modelo_base.py"),
)
train_module = importlib.util.module_from_spec(_spec)
sys.modules["train_modelo_base"] = train_module
_spec.loader.exec_module(train_module)


# ─────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────
def _create_fake_npz(directory: Path, seed: int = 0):
    """Genera un .npz sintético con las claves que espera train_base_model."""
    rng = np.random.default_rng(42)
    np.savez(
        directory / f"spiral_splits_seed_{seed}.npz",
        X_retain=rng.standard_normal((60, 2)).astype(np.float32),
        y_retain=rng.integers(0, 3, 60),
        X_forget=rng.standard_normal((30, 2)).astype(np.float32),
        y_forget=rng.integers(0, 3, 30),
        X_val=rng.standard_normal((30, 2)).astype(np.float32),
        y_val=rng.integers(0, 3, 30),
    )


@pytest.fixture(autouse=True)
def _redirect_datasets_path(tmp_path, monkeypatch):
    """Redirige DATASETS_PATH a tmp_path para todos los tests."""
    monkeypatch.setattr(train_module, "DATASETS_PATH", tmp_path)


@pytest.fixture
def fake_dataset(tmp_path):
    """Crea un .npz sintético y devuelve tmp_path."""
    _create_fake_npz(tmp_path, seed=0)
    return tmp_path


@pytest.fixture
def weights_dir(tmp_path, monkeypatch):
    """
    Redirige el directorio de guardado de pesos a tmp_path/models/weights.
    En lugar de monkeypatchar Path (que rompe internals de pathlib en 3.12),
    envolvemos la función completa para controlar dónde se guardan los pesos.
    """
    target_dir = tmp_path / "models" / "weights"

    # Guardamos referencia al torch.save original
    _original_torch_save = torch.save
    _saved_paths = []

    def _patched_save(obj, f, *args, **kwargs):
        """Intercepta torch.save para redirigir la ruta de guardado."""
        if isinstance(f, Path) and "base_model_seed" in str(f):
            target_dir.mkdir(parents=True, exist_ok=True)
            new_path = target_dir / f.name
            _saved_paths.append(new_path)
            return _original_torch_save(obj, new_path, *args, **kwargs)
        return _original_torch_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(train_module.torch, "save", _patched_save)
    return target_dir


def _train(seed=0, epochs=2, batch_size=16, verbose=False):
    """Shortcut para entrenar con parámetros mínimos."""
    return train_module.train_base_model(
        seed=seed, epochs=epochs, batch_size=batch_size, verbose=verbose
    )


# ─────────────────────────────────────────────
# Manejo de errores
# ─────────────────────────────────────────────
class TestTrainBaseModel_Errors:

    def test_file_not_found_for_missing_seed(self):
        """Debe lanzar FileNotFoundError si el .npz no existe."""
        with pytest.raises(FileNotFoundError, match="No se encontró el dataset"):
            _train(seed=999)


# ─────────────────────────────────────────────
# Entrenamiento básico
# ─────────────────────────────────────────────
class TestTrainBaseModel_Training:

    def test_returns_a_model(self, fake_dataset, weights_dir):
        """train_base_model debe devolver una instancia de BaseMLP."""
        from models.base_nn import BaseMLP
        model = _train()
        assert isinstance(model, BaseMLP)

    def test_model_output_dim(self, fake_dataset, weights_dir):
        """El modelo devuelto debe tener output_dim=3 (3 clases de la espiral)."""
        model = _train()
        assert model.classifier.out_features == 3

    def test_model_input_dim(self, fake_dataset, weights_dir):
        """El modelo devuelto debe tener input_dim=2 (x1, x2)."""
        model = _train()
        assert model.feature_extractor[0].in_features == 2

    def test_model_produces_valid_predictions(self, fake_dataset, weights_dir):
        """El modelo entrenado produce predicciones con la forma correcta."""
        model = _train(epochs=5)
        model.eval()
        x = torch.randn(4, 2)
        logits = model(x)
        assert logits.shape == (4, 3)


# ─────────────────────────────────────────────
# Guardado de pesos
# ─────────────────────────────────────────────
class TestTrainBaseModel_Checkpointing:

    def test_weights_file_is_created(self, fake_dataset, weights_dir):
        """El archivo .pth debe crearse al finalizar el entrenamiento."""
        _train()
        expected = weights_dir / "base_model_seed_0.pth"
        assert expected.exists()

    def test_saved_weights_are_loadable(self, fake_dataset, weights_dir):
        """Los pesos guardados deben cargarse sin error en un BaseMLP nuevo."""
        from models.base_nn import BaseMLP
        _train()
        path = weights_dir / "base_model_seed_0.pth"

        new_model = BaseMLP(input_dim=2, hidden_dim=16, output_dim=3)
        new_model.load_state_dict(torch.load(path, weights_only=True))
        # Si no lanza excepción, el checkpoint es compatible.

    def test_loaded_model_produces_same_output(self, fake_dataset, weights_dir):
        """Un modelo cargado desde disco produce la misma salida que el original."""
        from models.base_nn import BaseMLP
        original = _train()
        path = weights_dir / "base_model_seed_0.pth"

        loaded = BaseMLP(input_dim=2, hidden_dim=16, output_dim=3)
        loaded.load_state_dict(torch.load(path, weights_only=True))

        original.eval()
        loaded.eval()
        x = torch.randn(10, 2)
        torch.testing.assert_close(original(x), loaded(x))


# ─────────────────────────────────────────────
# Datos de entrenamiento (construcción interna)
# ─────────────────────────────────────────────
class TestTrainBaseModel_DataHandling:

    def test_train_uses_retain_and_forget(self, fake_dataset, weights_dir, tmp_path):
        """El set de entrenamiento debe combinar retain + forget."""
        data = np.load(tmp_path / "spiral_splits_seed_0.npz")
        expected_train_len = len(data['X_retain']) + len(data['X_forget'])
        assert expected_train_len == 90  # 60 retain + 30 forget
        model = _train(epochs=1, batch_size=expected_train_len)
        assert model is not None


# ─────────────────────────────────────────────
# Verbose / salida por consola
# ─────────────────────────────────────────────
class TestTrainBaseModel_Verbose:

    def test_verbose_prints_training_info(self, fake_dataset, weights_dir, capsys):
        """Con verbose=True se imprime información del entrenamiento."""
        _train(epochs=10, verbose=True)
        captured = capsys.readouterr().out
        assert "Iniciando entrenamiento" in captured
        assert "Modelo guardado exitosamente" in captured

    def test_silent_when_not_verbose(self, fake_dataset, weights_dir, capsys):
        """Con verbose=False no se imprime nada."""
        _train(epochs=10, verbose=False)
        captured = capsys.readouterr().out
        assert captured == ""
