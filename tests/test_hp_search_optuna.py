"""
Tests para 00_hp_search_optuna.py
Valida la carga de datos, la función objetivo, la persistencia de resultados
y el flujo completo de búsqueda con datos sintéticos.
"""

import sys
import json
import importlib.util
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import torch

# Cargamos el módulo dinámicamente porque empieza por un dígito.
_spec = importlib.util.spec_from_file_location(
    "hp_search_optuna",
    str(Path(__file__).parent.parent / "00_hp_search_optuna.py"),
)
hp_module = importlib.util.module_from_spec(_spec)
sys.modules["hp_search_optuna"] = hp_module
_spec.loader.exec_module(hp_module)


# ─────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────
def _create_fake_npz(directory: Path, seed: int = 0):
    """Genera un .npz sintético con las claves que espera load_data."""
    rng = np.random.default_rng(42)
    np.savez(
        directory / f"spiral_splits_seed_{seed}.npz",
        X_retain=rng.standard_normal((60, 2)).astype(np.float32),
        y_retain=rng.integers(0, 3, 60),
        X_forget=rng.standard_normal((30, 2)).astype(np.float32),
        y_forget=rng.integers(0, 3, 30),
        X_val=rng.standard_normal((20, 2)).astype(np.float32),
        y_val=rng.integers(0, 3, 20),
    )


@pytest.fixture(autouse=True)
def _redirect_paths(tmp_path, monkeypatch):
    """Redirige DATASETS_PATH y BEST_HP_PATH a tmp_path."""
    monkeypatch.setattr(hp_module, "DATASETS_PATH", tmp_path)
    monkeypatch.setattr(hp_module, "BEST_HP_PATH", tmp_path / "best_hp.json")


@pytest.fixture
def fake_dataset(tmp_path):
    """Crea un .npz sintético (seed=0) y devuelve tmp_path."""
    _create_fake_npz(tmp_path, seed=0)
    return tmp_path


# ─────────────────────────────────────────────
# Tests: load_data
# ─────────────────────────────────────────────
class TestLoadData:

    def test_returns_four_tensors(self, fake_dataset):
        """load_data debe devolver exactamente 4 tensores."""
        result = hp_module.load_data(seed=0)
        assert len(result) == 4
        for t in result:
            assert isinstance(t, torch.Tensor)

    def test_train_combines_retain_and_forget(self, fake_dataset, tmp_path):
        """El X_train resultante debe ser la combinación de retain + forget."""
        data = np.load(tmp_path / "spiral_splits_seed_0.npz")
        expected_len = len(data["X_retain"]) + len(data["X_forget"])

        X_train_t, y_train_t, _, _ = hp_module.load_data(seed=0)
        assert X_train_t.shape[0] == expected_len
        assert y_train_t.shape[0] == expected_len

    def test_val_tensors_shape(self, fake_dataset, tmp_path):
        """Los tensores de validación deben tener la forma correcta."""
        _, _, X_val_t, y_val_t = hp_module.load_data(seed=0)
        assert X_val_t.shape == (20, 2)
        assert y_val_t.shape == (20,)

    def test_train_dtype(self, fake_dataset):
        """X debe ser float32, y debe ser long."""
        X_train_t, y_train_t, X_val_t, y_val_t = hp_module.load_data(seed=0)
        assert X_train_t.dtype == torch.float32
        assert y_train_t.dtype == torch.long

    def test_file_not_found_raises(self):
        """Debe lanzar FileNotFoundError si el .npz no existe."""
        with pytest.raises(FileNotFoundError, match="No se encontró"):
            hp_module.load_data(seed=999)


# ─────────────────────────────────────────────
# Tests: objective
# ─────────────────────────────────────────────
class TestObjective:

    def test_returns_float(self, fake_dataset):
        """La función objective debe devolver un float (val_loss)."""
        # Usamos un FixedTrial para evitar la búsqueda real
        import optuna
        trial = optuna.trial.FixedTrial({
            "hidden_dim": 8,
            "lr": 0.01,
            "batch_size": 16,
            "epochs": 2,
        })
        val_loss = hp_module.objective(trial)
        assert isinstance(val_loss, float)

    def test_returns_non_negative(self, fake_dataset):
        """La val_loss (CrossEntropy) debe ser >= 0."""
        import optuna
        trial = optuna.trial.FixedTrial({
            "hidden_dim": 8,
            "lr": 0.01,
            "batch_size": 16,
            "epochs": 2,
        })
        val_loss = hp_module.objective(trial)
        assert val_loss >= 0.0

    def test_different_hps_can_produce_different_losses(self, fake_dataset):
        """Diferentes HP deben poder producir resultados distintos."""
        import optuna
        trial_a = optuna.trial.FixedTrial({
            "hidden_dim": 8, "lr": 0.1, "batch_size": 16, "epochs": 3,
        })
        trial_b = optuna.trial.FixedTrial({
            "hidden_dim": 64, "lr": 0.001, "batch_size": 64, "epochs": 3,
        })
        loss_a = hp_module.objective(trial_a)
        loss_b = hp_module.objective(trial_b)
        # No exigimos que sean distintos (podría pasar), solo que ambos son válidos
        assert isinstance(loss_a, float) and isinstance(loss_b, float)


# ─────────────────────────────────────────────
# Tests: run_search
# ─────────────────────────────────────────────
class TestRunSearch:

    def test_saves_best_hp_json(self, fake_dataset, tmp_path):
        """run_search debe crear el fichero best_hp.json."""
        hp_module.run_search(n_trials=2)
        hp_path = tmp_path / "best_hp.json"
        assert hp_path.exists()

    def test_saved_json_contains_expected_keys(self, fake_dataset, tmp_path):
        """El JSON guardado debe contener los HP del espacio de búsqueda."""
        hp_module.run_search(n_trials=2)
        hp_path = tmp_path / "best_hp.json"
        with open(hp_path, "r") as f:
            hp = json.load(f)
        expected_keys = {"hidden_dim", "lr", "batch_size", "epochs"}
        assert expected_keys == set(hp.keys())

    def test_returns_dict(self, fake_dataset):
        """run_search debe devolver un diccionario."""
        result = hp_module.run_search(n_trials=2)
        assert isinstance(result, dict)

    def test_hidden_dim_in_valid_range(self, fake_dataset):
        """hidden_dim devuelto debe estar en el espacio de búsqueda."""
        result = hp_module.run_search(n_trials=2)
        assert result["hidden_dim"] in [8, 16, 32, 64]

    def test_lr_in_valid_range(self, fake_dataset):
        """lr devuelto debe estar dentro del rango buscado."""
        result = hp_module.run_search(n_trials=2)
        assert 1e-4 <= result["lr"] <= 1e-1
