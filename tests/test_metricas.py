"""
tests/test_metricas.py
───────────────────────────────────────────────
Valida la carga de clases, evaluación de precisión, el cálculo de métricas relativas (RA, FA),
la tolerancia a la división por cero y la exportación de resultados JSON.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn
import importlib.util
import json

# Cargar el módulo de métricas dinámicamente
_spec = importlib.util.spec_from_file_location(
    "metricas",
    str(Path(__file__).parent.parent / "4_metricas.py"),
)
metricas_module = importlib.util.module_from_spec(_spec)
sys.modules["metricas"] = metricas_module
_spec.loader.exec_module(metricas_module)


# ─────────────────────────────────────────────
# Fixtures / Helpers
# ─────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _redirect_paths(tmp_path, monkeypatch):
    """Redirige DATASETS_PATH y el guardado/carga de modelos."""
    # Redirigir DATASETS_PATH en el módulo
    monkeypatch.setattr(metricas_module, "DATASETS_PATH", tmp_path)
    
    # Redirigir la carga de pesos interceptando torch.load
    _original_torch_load = torch.load
    
    def _patched_load(f, *args, **kwargs):
        path_str = str(f)
        if "models/weights" in path_str or "models\\weights" in path_str:
            filename = Path(f).name
            new_path = tmp_path / "models" / "weights" / filename
            return _original_torch_load(new_path, *args, **kwargs)
        return _original_torch_load(f, *args, **kwargs)
        
    monkeypatch.setattr(metricas_module.torch, "load", _patched_load)


def _create_fake_npz(directory: Path, seed: int = 0, constant_labels: int = None):
    """Genera un .npz sintético para evaluación."""
    rng = np.random.default_rng(42)
    X_retain = rng.standard_normal((40, 2)).astype(np.float32)
    X_forget = rng.standard_normal((20, 2)).astype(np.float32)
    X_test = rng.standard_normal((20, 2)).astype(np.float32)
    
    if constant_labels is not None:
        y_retain = np.full(40, constant_labels)
        y_forget = np.full(20, constant_labels)
        y_test = np.full(20, constant_labels)
    else:
        y_retain = rng.integers(0, 3, 40)
        y_forget = rng.integers(0, 3, 20)
        y_test = rng.integers(0, 3, 20)
        
    np.savez(
        directory / f"spiral_splits_seed_{seed}.npz",
        X_retain=X_retain,
        y_retain=y_retain,
        X_forget=X_forget,
        y_forget=y_forget,
        X_val=rng.standard_normal((20, 2)).astype(np.float32),
        y_val=rng.integers(0, 3, 20),
        X_test=X_test,
        y_test=y_test,
    )


# ─────────────────────────────────────────────
# Test Carga Dinámica (load_class)
# ─────────────────────────────────────────────
def test_load_class_valid():
    cls = metricas_module.load_class("models.base_nn.BaseMLP")
    assert cls is not None
    assert cls.__name__ == "BaseMLP"


def test_load_class_invalid():
    with pytest.raises(ImportError):
        metricas_module.load_class("models.base_nn.NonExistentClass")


# ─────────────────────────────────────────────
# Test Evaluación de Modelos (evaluate_model)
# ─────────────────────────────────────────────
def test_evaluate_model_perfect_classifier():
    """Un clasificador que predice siempre la clase 1 tiene 100% acc sobre datos de clase 1."""
    class DummyModel(nn.Module):
        def forward(self, x):
            # Devolver logits altos en la clase 1
            out = torch.zeros(x.size(0), 3)
            out[:, 1] = 10.0
            return out
            
    model = DummyModel()
    X = np.random.randn(10, 2).astype(np.float32)
    y = np.ones(10, dtype=np.int64)
    
    acc = metricas_module.evaluate_model(model, X, y, torch.device("cpu"))
    assert acc == 1.0


def test_evaluate_model_zero_accuracy():
    class DummyModel(nn.Module):
        def forward(self, x):
            # Devolver logits altos en la clase 0
            out = torch.zeros(x.size(0), 3)
            out[:, 0] = 10.0
            return out
            
    model = DummyModel()
    X = np.random.randn(10, 2).astype(np.float32)
    y = np.ones(10, dtype=np.int64)  # Espera clase 1, pero predice clase 0
    
    acc = metricas_module.evaluate_model(model, X, y, torch.device("cpu"))
    assert acc == 0.0


# ─────────────────────────────────────────────
# Test Pipeline de Métricas (calculate_metrics)
# ─────────────────────────────────────────────
def test_calculate_metrics_integration(tmp_path):
    # 1. Crear datasets para semillas 0 y 1
    _create_fake_npz(tmp_path, seed=0)
    _create_fake_npz(tmp_path, seed=1)
    
    # 2. Crear pesos dummy para base, naive, y unlearned para ambas semillas
    weights_dir = tmp_path / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    from models.base_nn import BaseMLP
    for seed in [0, 1]:
        for prefix in ["base", "naive", "cfk"]:
            model = BaseMLP(input_dim=2, hidden_dim=8, output_dim=3)
            torch.save(model.state_dict(), weights_dir / f"{prefix}_model_seed_{seed}.pth")
            
    # 3. Correr cálculo de métricas
    output_dir = tmp_path / "results"
    results = metricas_module.calculate_metrics(
        unlearned_name="cfk",
        base_name="base",
        naive_name="naive",
        model_arch="models.base_nn.BaseMLP",
        seeds=[0, 1],
        hp={"hidden_dim": 8},
        output_dir=str(output_dir)
    )
    
    # 4. Aseverar estructura
    assert results["unlearned_name"] == "cfk"
    assert results["seeds"] == [0, 1]
    assert "0" in results["per_seed"]
    assert "1" in results["per_seed"]
    
    # Verificar claves de salida en per_seed
    per_seed_0 = results["per_seed"]["0"]
    assert "base" in per_seed_0
    assert "naive" in per_seed_0
    assert "unlearned" in per_seed_0
    assert "RA" in per_seed_0
    assert "FA" in per_seed_0
    
    # Verificar agregaciones
    assert "RA" in results["aggregated"]
    assert "mean" in results["aggregated"]["RA"]
    assert "std" in results["aggregated"]["RA"]
    
    # Verificar que el JSON se guardó correctamente
    json_file = output_dir / "cfk_metrics.json"
    assert json_file.exists()
    
    with open(json_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["unlearned_name"] == "cfk"


class FlexibleModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.pred_class = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
    def forward(self, x):
        cls = int(torch.round(self.pred_class).item())
        # Asegurar que está en rango
        cls = max(0, min(2, cls))
        out = torch.zeros(x.size(0), 3)
        out[:, cls] = 10.0
        return out


def test_calculate_metrics_zero_division_protection(tmp_path):
    """
    Verifica que si la precisión del modelo Naive es 0.0,
    el cálculo de RA y FA no cause un error de ejecución.
    """
    # 1. Crear datasets para semilla 0
    _create_fake_npz(tmp_path, seed=0, constant_labels=1) # Todas las etiquetas son clase 1
    
    weights_dir = tmp_path / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear modelo base/unlearned (predice 1 -> 100% acc) y modelo naive (predice 0 -> 0% acc)
    model_good = FlexibleModel()
    model_good.pred_class.copy_(torch.tensor(1.0))
    
    model_bad = FlexibleModel()
    model_bad.pred_class.copy_(torch.tensor(0.0))
    
    # Guardamos los modelos
    torch.save(model_good.state_dict(), weights_dir / "base_model_seed_0.pth")
    torch.save(model_bad.state_dict(), weights_dir / "naive_model_seed_0.pth")
    torch.save(model_good.state_dict(), weights_dir / "cfk_model_seed_0.pth")
    
    # 3. Correr métricas
    output_dir = tmp_path / "results"
    results = metricas_module.calculate_metrics(
        unlearned_name="cfk",
        base_name="base",
        naive_name="naive",
        model_arch="tests.test_metricas.FlexibleModel",
        seeds=[0],
        hp={"hidden_dim": 8},
        output_dir=str(output_dir)
    )
    
    # Naive retain acc es 0, cfk es 1.0. RA debe manejarse con seguridad.
    assert results["per_seed"]["0"]["naive"]["retain"] == 0.0
    assert results["per_seed"]["0"]["unlearned"]["retain"] == 1.0
    assert results["per_seed"]["0"]["RA"] == float("inf")
