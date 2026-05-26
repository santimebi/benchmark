"""
Tests para 1_create_cifar10.py
Valida la generación del dataset CIFAR-10 completo mockeado:
dimensiones de los splits, contenido de las clases, lógica de forget/retain,
y reproducibilidad con semillas.
"""

import sys
import importlib.util
from pathlib import Path
import pytest
import numpy as np
import torch
from unittest.mock import patch

# Cargamos el módulo dinámicamente
_spec = importlib.util.spec_from_file_location(
    "create_cifar10",
    str(Path(__file__).parent.parent / "1_create_cifar10.py"),
)
create_cifar10_mod = importlib.util.module_from_spec(_spec)
sys.modules["create_cifar10"] = create_cifar10_mod
_spec.loader.exec_module(create_cifar10_mod)

create_cifar10 = create_cifar10_mod.create_cifar10


def get_mock_cifar10(train=True, download=True, transform=None):
    """
    Crea un dataset mock que simula ser CIFAR-10.
    100 muestras por clase para train, 20 para test.
    """
    samples_per_class = 100 if train else 20
    num_classes = 10
    
    data = []
    # Usar una semilla fija para el mock
    rng = np.random.default_rng(42 if train else 24)
    
    for c in range(num_classes):
        for _ in range(samples_per_class):
            img = torch.tensor(rng.random((3, 32, 32)), dtype=torch.float32)
            data.append((img, c))
            
    return data


@patch("torchvision.datasets.CIFAR10")
def test_create_cifar10_shapes_and_splits(mock_cifar10, tmp_path):
    """
    Verifica que se generen correctamente los archivos npz con los tamaños y claves esperados.
    Train: 1000 muestras. Val_ratio: 0.1 -> Val: 100 (10/clase), Train_split: 900 (90/clase).
    Clase 7 forget_ratio: 0.4 -> Forget: 36, Retain (clase 7): 54.
    Otras clases (9 clases) -> Retain: 9 * 90 = 810.
    Total Retain: 810 + 54 = 864.
    Test: 200 muestras (fijo).
    """
    mock_cifar10.side_effect = lambda root, train, download, transform: get_mock_cifar10(train, download, transform)
    
    create_cifar10(output_dir=tmp_path, seeds=[0], download=False, forget_class=7, forget_ratio=0.4, val_ratio=0.1)
    
    npz_path = tmp_path / "cifar10_splits_seed_0.npz"
    assert npz_path.exists()
    
    data = np.load(npz_path)
    
    assert "X_retain" in data and "y_retain" in data
    assert "X_forget" in data and "y_forget" in data
    assert "X_val" in data and "y_val" in data
    assert "X_test" in data and "y_test" in data
    
    # Comprobar shapes
    assert data["X_retain"].shape == (864, 3, 32, 32)
    assert data["y_retain"].shape == (864,)
    
    assert data["X_forget"].shape == (36, 3, 32, 32)
    assert data["y_forget"].shape == (36,)
    
    assert data["X_val"].shape == (100, 3, 32, 32)
    assert data["y_val"].shape == (100,)
    
    assert data["X_test"].shape == (200, 3, 32, 32)
    assert data["y_test"].shape == (200,)


@patch("torchvision.datasets.CIFAR10")
def test_create_cifar10_class_distribution(mock_cifar10, tmp_path):
    """
    Valida la distribución de clases en los distintos splits generados.
    """
    mock_cifar10.side_effect = lambda root, train, download, transform: get_mock_cifar10(train, download, transform)
    
    create_cifar10(output_dir=tmp_path, seeds=[0], download=False, forget_class=7, forget_ratio=0.4, val_ratio=0.1)
    data = np.load(tmp_path / "cifar10_splits_seed_0.npz")
    
    y_retain = data["y_retain"]
    y_forget = data["y_forget"]
    y_val = data["y_val"]
    y_test = data["y_test"]
    
    # Forget debe contener únicamente la clase 7
    assert np.all(y_forget == 7)
    assert len(y_forget) == 36
    
    # Retain debe contener 54 muestras de la clase 7 y 90 de las clases restantes
    classes, counts = np.unique(y_retain, return_counts=True)
    count_dict = dict(zip(classes, counts))
    assert count_dict[7] == 54
    for c in range(10):
        if c != 7:
            assert count_dict[c] == 90
            
    # Val debe contener exactamente 10 muestras por clase
    classes_val, counts_val = np.unique(y_val, return_counts=True)
    assert len(classes_val) == 10
    assert np.all(counts_val == 10)
    
    # Test debe contener exactamente 20 muestras por clase
    classes_test, counts_test = np.unique(y_test, return_counts=True)
    assert len(classes_test) == 10
    assert np.all(counts_test == 20)


@patch("torchvision.datasets.CIFAR10")
def test_create_cifar10_reproducibility(mock_cifar10, tmp_path):
    """
    Verifica reproducibilidad con semillas.
    """
    mock_cifar10.side_effect = lambda root, train, download, transform: get_mock_cifar10(train, download, transform)
    
    create_cifar10(output_dir=tmp_path, seeds=[42, 42, 100], download=False, forget_class=7, forget_ratio=0.4, val_ratio=0.1)
    
    data_42_a = np.load(tmp_path / "cifar10_splits_seed_42.npz")
    
    tmp_path_b = tmp_path / "b"
    create_cifar10(output_dir=tmp_path_b, seeds=[42], download=False, forget_class=7, forget_ratio=0.4, val_ratio=0.1)
    data_42_b = np.load(tmp_path_b / "cifar10_splits_seed_42.npz")
    
    data_100 = np.load(tmp_path / "cifar10_splits_seed_100.npz")
    
    # Misma semilla -> Mismo resultado
    np.testing.assert_array_equal(data_42_a["X_retain"], data_42_b["X_retain"])
    np.testing.assert_array_equal(data_42_a["y_retain"], data_42_b["y_retain"])
    np.testing.assert_array_equal(data_42_a["X_forget"], data_42_b["X_forget"])
    
    # Semilla diferente -> Diferente resultado
    assert not np.array_equal(data_42_a["X_retain"], data_100["X_retain"])
