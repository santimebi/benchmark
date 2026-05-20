"""
tests/test_hp_search.py
───────────────────────────────────────────────
Valida la lógica de búsqueda de hiperparámetros generalizada (00_hp_search.py).
"""

import sys
import json
from pathlib import Path
import pytest
import importlib.util
import optuna
import torch

# Importar dinámicamente el módulo
_spec = importlib.util.spec_from_file_location(
    "hp_search",
    str(Path(__file__).parent.parent / "00_hp_search.py"),
)
hp_search_module = importlib.util.module_from_spec(_spec)
sys.modules["hp_search"] = hp_search_module
_spec.loader.exec_module(hp_search_module)

from utils.hp_spaces import HP_SPACES


@pytest.fixture(autouse=True)
def _patch_config(tmp_path, monkeypatch):
    """Redirige DATASETS_PATH y salidas JSON a directorios temporales."""
    monkeypatch.setattr(hp_search_module, "DATASETS_PATH", tmp_path)
    
    # Redirigir output_path en HP_SPACES a directorios temporales
    monkeypatch.setitem(HP_SPACES["standard"], "output_path", tmp_path / "best_hp.json")
    monkeypatch.setitem(HP_SPACES["cfk"], "output_path", tmp_path / "best_cfk_hp.json")


def _create_fake_data(directory: Path, seed: int = 0):
    """Crea splits sintéticos para el buscador."""
    import numpy as np
    rng = np.random.default_rng(42)
    np.savez(
        directory / f"spiral_splits_seed_{seed}.npz",
        X_retain=rng.standard_normal((10, 2)).astype(np.float32),
        y_retain=rng.integers(0, 3, 10),
        X_forget=rng.standard_normal((5, 2)).astype(np.float32),
        y_forget=rng.integers(0, 3, 5),
        X_val=rng.standard_normal((5, 2)).astype(np.float32),
        y_val=rng.integers(0, 3, 5),
        X_test=rng.standard_normal((5, 2)).astype(np.float32),
        y_test=rng.integers(0, 3, 5),
    )


def test_hp_spaces_configuration():
    assert "standard" in HP_SPACES
    assert "cfk" in HP_SPACES
    assert HP_SPACES["standard"]["objective_type"] == "val_loss"
    assert HP_SPACES["cfk"]["objective_type"] == "unlearning_loss"


def test_hp_search_standard_objective(tmp_path):
    _create_fake_data(tmp_path, seed=0)
    
    study = optuna.create_study(direction="minimize")
    
    def objective_wrapper(trial):
        return hp_search_module.objective(trial, protocol="standard", seed=0, model_arch="models.base_nn.BaseMLP")
        
    study.optimize(objective_wrapper, n_trials=2)
    
    assert len(study.trials) == 2
    assert study.best_value is not None


def test_hp_search_cfk_objective(tmp_path, monkeypatch):
    _create_fake_data(tmp_path, seed=0)
    
    # Crear pesos base y naive dummy
    weights_dir = tmp_path / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    from models.base_nn import BaseMLP
    model = BaseMLP(input_dim=2, hidden_dim=8, output_dim=3)
    torch.save(model.state_dict(), weights_dir / "base_model_seed_0.pth")
    torch.save(model.state_dict(), weights_dir / "naive_model_seed_0.pth")
    
    # Crear models/best_hp.json dummy en tmp_path
    best_hp_path = tmp_path / "best_hp.json"
    with open(best_hp_path, "w", encoding="utf-8") as f:
        json.dump({"hidden_dim": 8, "lr": 0.01, "batch_size": 16, "epochs": 50}, f)
        
    # Redirigir load_best_hp para leer de nuestro best_hp_path temporal
    monkeypatch.setattr(hp_search_module, "load_best_hp", lambda f: {"hidden_dim": 8, "lr": 0.01, "batch_size": 16, "epochs": 50})
    
    # Redirigir torch.load para leer del directorio temporal de pesos
    _original_torch_load = torch.load
    def patched_torch_load(f, *args, **kwargs):
        filename = Path(f).name
        return _original_torch_load(weights_dir / filename, *args, **kwargs)
    monkeypatch.setattr(hp_search_module.torch, "load", patched_torch_load)
    
    study = optuna.create_study(direction="minimize")
    
    def objective_wrapper(trial):
        return hp_search_module.objective(trial, protocol="cfk", seed=0, model_arch="models.base_nn.BaseMLP")
        
    study.optimize(objective_wrapper, n_trials=2)
    
    assert len(study.trials) == 2
    assert study.best_value is not None
