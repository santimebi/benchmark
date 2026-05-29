"""
Tests para 1_create_cifar_nano.py
Valida la generación del dataset CIFAR-nano: dimensiones de los splits,
contenido de las clases, lógica de forget/retain, y reproducibilidad con semillas.
"""

import sys
import importlib.util
from pathlib import Path
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# Cargamos el módulo dinámicamente
_spec = importlib.util.spec_from_file_location(
    "create_cifar_nano",
    str(Path(__file__).parent.parent / "1_create_cifar_nano.py"),
)
create_cifar_nano_mod = importlib.util.module_from_spec(_spec)
sys.modules["create_cifar_nano"] = create_cifar_nano_mod
_spec.loader.exec_module(create_cifar_nano_mod)

create_cifar_nano = create_cifar_nano_mod.create_cifar_nano


def get_mock_cifar10(train=True, download=True, transform=None):
    """
    Crea un dataset mock que simula ser CIFAR-10 con muestras suficientes para cada clase.
    """
    # Generamos 15 muestras por clase para train (150 total) y 5 por clase para test (50 total)
    samples_per_class = 15 if train else 5
    num_classes = 10
    
    data = []
    # Usar una semilla fija para el mock para que sea predecible
    rng = np.random.default_rng(42 if train else 24)
    
    for c in range(num_classes):
        for _ in range(samples_per_class):
            # Imagen mock (3, 32, 32)
            img = torch.tensor(rng.random((3, 32, 32)), dtype=torch.float32)
            data.append((img, c))
            
    return data


@patch("torchvision.datasets.CIFAR10")
def test_create_cifar_nano_shapes_and_splits(mock_cifar10, tmp_path):
    """
    Verifica que se generen correctamente los archivos npz con los tamaños y claves esperados.
    """
    # Configurar mock de CIFAR-10
    mock_cifar10.side_effect = lambda root, train, download, transform: get_mock_cifar10(train, download, transform)
    
    # Ejecutar la creación
    create_cifar_nano(output_dir=tmp_path, seeds=[0], download=False)
    
    # Comprobar que el archivo se ha creado
    npz_path = tmp_path / "cifar_nano_c0_n2_splits_seed_0.npz"
    assert npz_path.exists()
    
    # Cargar y verificar el contenido
    data = np.load(npz_path)
    
    assert "X_retain" in data and "y_retain" in data
    assert "X_forget" in data and "y_forget" in data
    assert "X_val" in data and "y_val" in data
    assert "X_test" in data and "y_test" in data
    
    # Comprobar shapes
    # Retain: 5 de la clase 0 + 7 * 9 de las clases 1-9 = 68 muestras
    assert data["X_retain"].shape == (68, 3, 32, 32)
    assert data["y_retain"].shape == (68,)
    
    # Forget: 2 de la clase 0 = 2 muestras
    assert data["X_forget"].shape == (2, 3, 32, 32)
    assert data["y_forget"].shape == (2,)
    
    # Val: 1 por clase = 10 muestras
    assert data["X_val"].shape == (10, 3, 32, 32)
    assert data["y_val"].shape == (10,)
    
    # Test: 2 por clase = 20 muestras
    assert data["X_test"].shape == (20, 3, 32, 32)
    assert data["y_test"].shape == (20,)


@patch("torchvision.datasets.CIFAR10")
def test_create_cifar_nano_class_distribution(mock_cifar10, tmp_path):
    """
    Valida la distribución de clases en los distintos splits generados.
    """
    mock_cifar10.side_effect = lambda root, train, download, transform: get_mock_cifar10(train, download, transform)
    
    create_cifar_nano(output_dir=tmp_path, seeds=[0], download=False)
    data = np.load(tmp_path / "cifar_nano_c0_n2_splits_seed_0.npz")
    
    y_retain = data["y_retain"]
    y_forget = data["y_forget"]
    y_val = data["y_val"]
    y_test = data["y_test"]
    
    # Forget debe contener únicamente la clase 0
    assert np.all(y_forget == 0)
    assert len(y_forget) == 2
    
    # Retain debe contener 5 muestras de la clase 0 y 7 de las clases 1-9
    classes, counts = np.unique(y_retain, return_counts=True)
    count_dict = dict(zip(classes, counts))
    assert count_dict[0] == 5
    for c in range(1, 10):
        assert count_dict[c] == 7
        
    # Val debe contener exactamente 1 muestra por clase
    classes_val, counts_val = np.unique(y_val, return_counts=True)
    assert len(classes_val) == 10
    assert np.all(counts_val == 1)
    
    # Test debe contener exactamente 2 muestras por clase
    classes_test, counts_test = np.unique(y_test, return_counts=True)
    assert len(classes_test) == 10
    assert np.all(counts_test == 2)


@patch("torchvision.datasets.CIFAR10")
def test_create_cifar_nano_reproducibility(mock_cifar10, tmp_path):
    """
    Verifica que la misma semilla produce exactamente los mismos splits y diferentes semillas producen diferentes.
    """
    mock_cifar10.side_effect = lambda root, train, download, transform: get_mock_cifar10(train, download, transform)
    
    create_cifar_nano(output_dir=tmp_path, seeds=[42, 42, 100], download=False)
    
    data_42_a = np.load(tmp_path / "cifar_nano_c0_n2_splits_seed_42.npz")
    # Para el segundo 42, se sobreescribe el mismo fichero, por lo que creamos otra carpeta
    tmp_path_b = tmp_path / "b"
    create_cifar_nano(output_dir=tmp_path_b, seeds=[42], download=False)
    data_42_b = np.load(tmp_path_b / "cifar_nano_c0_n2_splits_seed_42.npz")
    
    data_100 = np.load(tmp_path / "cifar_nano_c0_n2_splits_seed_100.npz")
    
    # Misma semilla -> Mismo resultado
    np.testing.assert_array_equal(data_42_a["X_retain"], data_42_b["X_retain"])
    np.testing.assert_array_equal(data_42_a["y_retain"], data_42_b["y_retain"])
    np.testing.assert_array_equal(data_42_a["X_forget"], data_42_b["X_forget"])
    
    # Semilla diferente -> Diferente resultado
    assert not np.array_equal(data_42_a["X_retain"], data_100["X_retain"])
