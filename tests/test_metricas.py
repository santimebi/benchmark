"""
tests/test_metricas.py
───────────────────────────────────────────────
Valida la carga de clases, evaluación de precisión, el cálculo de métricas relativas (RR, RF, RT),
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
    
    # 2. Crear pesos dummy y metadatos para base, naive, y unlearned para ambas semillas
    weights_dir = tmp_path / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    from models.base_nn import BaseMLP
    for seed in [0, 1]:
        for prefix in ["base", "naive", "cfk"]:
            model = BaseMLP(input_dim=2, hidden_dim=8, output_dim=3)
            torch.save(model.state_dict(), weights_dir / f"{prefix}_model_seed_{seed}.pth")
            
            # Guardar metadatos dummy
            meta = {"epochs": 10, "time_elapsed": 1.5}
            with open(weights_dir / f"{prefix}_model_seed_{seed}_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)
            
    # 3. Correr cálculo de métricas
    output_dir = tmp_path / "results"
    results = metricas_module.calculate_metrics(
        unlearned_name="cfk",
        base_name="base",
        naive_name="naive",
        model_arch="models.base_nn.BaseMLP",
        seeds=[0, 1],
        hp={"hidden_dim": 8},
        output_dir=str(output_dir),
        weights_dir=str(weights_dir)
    )
    
    # 4. Aseverar estructura
    assert results["unlearned_name"] == "cfk"
    assert results["seeds"] == [0, 1]
    assert "0" in results["per_seed"]
    assert "1" in results["per_seed"]
    
    # Verificar claves de salida en per_seed
    per_seed_0 = results["per_seed"]["0"]
    for key in ["base", "naive", "unlearned"]:
        assert key in per_seed_0
        assert "retain" in per_seed_0[key]
        assert "forget" in per_seed_0[key]
        assert "test" in per_seed_0[key]
        assert "RR" in per_seed_0[key]
        assert "RF" in per_seed_0[key]
        assert "RT" in per_seed_0[key]
        assert "epochs" in per_seed_0[key]
        assert "time" in per_seed_0[key]
        assert "TR" in per_seed_0[key]
        
    # El naive debe dar exactamente 1.0
    assert per_seed_0["naive"]["RR"] == 1.0
    assert per_seed_0["naive"]["RF"] == 1.0
    assert per_seed_0["naive"]["RT"] == 1.0
    assert per_seed_0["naive"]["TR"] == 1.0
    
    # Verificar valores cargados
    assert per_seed_0["unlearned"]["epochs"] == 10
    assert per_seed_0["unlearned"]["time"] == 1.5
    assert per_seed_0["unlearned"]["TR"] == 1.0
    
    # Verificar agregaciones
    assert "unlearned_RR" in results["aggregated"]
    assert "mean" in results["aggregated"]["unlearned_RR"]
    assert "std" in results["aggregated"]["unlearned_RR"]
    assert "unlearned_TR" in results["aggregated"]
    
    # Verificar que el JSON se guardó correctamente
    json_file = output_dir / "cfk_metrics.json"
    assert json_file.exists()
    
    with open(json_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["unlearned_name"] == "cfk"


def test_calculate_metrics_integration_euk(tmp_path):
    # 1. Crear datasets para semillas 0 y 1
    _create_fake_npz(tmp_path, seed=0)
    _create_fake_npz(tmp_path, seed=1)
    
    # 2. Crear pesos dummy y metadatos para base, naive, y unlearned para ambas semillas
    weights_dir = tmp_path / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    from models.base_nn import BaseMLP
    for seed in [0, 1]:
        for prefix in ["base", "naive", "euk"]:
            model = BaseMLP(input_dim=2, hidden_dim=8, output_dim=3)
            torch.save(model.state_dict(), weights_dir / f"{prefix}_model_seed_{seed}.pth")
            
            # Guardar metadatos dummy
            meta = {"epochs": 10, "time_elapsed": 1.5}
            with open(weights_dir / f"{prefix}_model_seed_{seed}_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)
            
    # 3. Correr cálculo de métricas
    output_dir = tmp_path / "results"
    results = metricas_module.calculate_metrics(
        unlearned_name="euk",
        base_name="base",
        naive_name="naive",
        model_arch="models.base_nn.BaseMLP",
        seeds=[0, 1],
        hp={"hidden_dim": 8},
        output_dir=str(output_dir),
        weights_dir=str(weights_dir)
    )
    
    # 4. Aseverar estructura
    assert results["unlearned_name"] == "euk"
    assert results["seeds"] == [0, 1]
    assert "0" in results["per_seed"]
    assert "1" in results["per_seed"]
    
    # Verificar que el JSON se guardó correctamente
    json_file = output_dir / "euk_metrics.json"
    assert json_file.exists()
    
    with open(json_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["unlearned_name"] == "euk"


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
    Verifica que si la precisión del modelo Naive o su tiempo es 0.0,
    el cálculo de ratios no cause un error de ejecución y maneje floats correctamente.
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
    
    # Guardar metadatos con naive time = 0.0
    with open(weights_dir / "base_model_seed_0_meta.json", "w", encoding="utf-8") as f:
        json.dump({"epochs": 20, "time_elapsed": 2.0}, f)
    with open(weights_dir / "naive_model_seed_0_meta.json", "w", encoding="utf-8") as f:
        json.dump({"epochs": 20, "time_elapsed": 0.0}, f)
    with open(weights_dir / "cfk_model_seed_0_meta.json", "w", encoding="utf-8") as f:
        json.dump({"epochs": 20, "time_elapsed": 3.0}, f)
    
    # 3. Correr métricas
    output_dir = tmp_path / "results"
    results = metricas_module.calculate_metrics(
        unlearned_name="cfk",
        base_name="base",
        naive_name="naive",
        model_arch="tests.test_metricas.FlexibleModel",
        seeds=[0],
        hp={"hidden_dim": 8},
        output_dir=str(output_dir),
        weights_dir=str(weights_dir)
    )
    
    # Naive retain acc es 0, cfk es 1.0. RR debe ser float("inf").
    assert results["per_seed"]["0"]["naive"]["retain"] == 0.0
    assert results["per_seed"]["0"]["unlearned"]["retain"] == 1.0
    assert results["per_seed"]["0"]["unlearned"]["RR"] == float("inf")
    
    # Naive time es 0.0, cfk time es 3.0. TR debe ser float("inf").
    assert results["per_seed"]["0"]["unlearned"]["TR"] == float("inf")


def test_calculate_metrics_resnet_custom_dataset(tmp_path):
    """
    Test metric calculations for ResNet18 model on a custom dataset name.
    """
    import numpy as np
    rng = np.random.default_rng(42)
    y_retain = rng.integers(0, 3, 20)
    y_retain[:3] = np.arange(3)
    y_forget = rng.integers(0, 3, 10)
    y_forget[:3] = np.arange(3)
    y_val = rng.integers(0, 3, 10)
    y_val[:3] = np.arange(3)
    y_test = rng.integers(0, 3, 10)
    y_test[:3] = np.arange(3)

    np.savez(
        tmp_path / "cifar10_fake_splits_seed_0.npz",
        X_retain=rng.standard_normal((20, 3, 8, 8)).astype(np.float32),
        y_retain=y_retain,
        X_forget=rng.standard_normal((10, 3, 8, 8)).astype(np.float32),
        y_forget=y_forget,
        X_val=rng.standard_normal((10, 3, 8, 8)).astype(np.float32),
        y_val=y_val,
        X_test=rng.standard_normal((10, 3, 8, 8)).astype(np.float32),
        y_test=y_test,
    )

    weights_dir = tmp_path / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    from models.resnet import ResNet18
    for prefix in ["base", "naive", "cfk"]:
        model = ResNet18(in_channels=3, num_classes=3)
        torch.save(model.state_dict(), weights_dir / f"{prefix}_model_seed_0.pth")

        # Save metadata
        meta = {"epochs": 5, "time_elapsed": 1.0}
        with open(weights_dir / f"{prefix}_model_seed_0_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    output_dir = tmp_path / "results"
    results = metricas_module.calculate_metrics(
        unlearned_name="cfk",
        base_name="base",
        naive_name="naive",
        model_arch="models.resnet.ResNet18",
        seeds=[0],
        hp={},
        output_dir=str(output_dir),
        weights_dir=str(weights_dir),
        dataset="cifar10_fake"
    )

    assert results["unlearned_name"] == "cfk"
    assert "0" in results["per_seed"]
    assert results["per_seed"]["0"]["unlearned"]["epochs"] == 5
    assert results["per_seed"]["0"]["unlearned"]["time"] == 1.0

