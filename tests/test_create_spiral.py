import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch
from create_spiral import make_spiral_dataset

def test_make_spiral_dataset_shapes(tmp_path):
    """Comprueba que las dimensiones de X e y sean correctas."""
    n_points_per_class = 100
    n_classes = 3
    file_name = tmp_path / "spiral.csv"
    
    X, y = make_spiral_dataset(
        file_name=file_name,
        n_points_per_class=n_points_per_class,
        n_classes=n_classes,
        verbose=False,
        plot_dataset=False
    )
    
    assert X.shape == (n_points_per_class * n_classes, 2)
    assert y.shape == (n_points_per_class * n_classes,)
    assert set(np.unique(y)) == set(range(n_classes))

def test_make_spiral_dataset_reproducibility(tmp_path):
    """Verifica que dado un random_state constante, el resultado es el mismo."""
    file_name1 = tmp_path / "spiral1.csv"
    file_name2 = tmp_path / "spiral2.csv"
    
    X1, y1 = make_spiral_dataset(file_name=file_name1, random_state=42, verbose=False, plot_dataset=False)
    X2, y2 = make_spiral_dataset(file_name=file_name2, random_state=42, verbose=False, plot_dataset=False)
    
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)

def test_make_spiral_dataset_saving_new_file(tmp_path):
    """Comprueba que se genera y guarda el CSV correctamente cuando no existe."""
    file_name = tmp_path / "test_spiral.csv"
    
    assert not file_name.exists()
    
    make_spiral_dataset(file_name=file_name, verbose=False, plot_dataset=False)
    
    assert file_name.exists()
    
    # Comprobar el contenido (ignorando la cabecera)
    data = np.loadtxt(file_name, delimiter=",", skiprows=1)
    assert data.shape[1] == 3 # x1, x2, label

def test_make_spiral_dataset_no_overwrite(tmp_path, capsys):
    """Comprueba que NO sobreescribe el archivo si overwrite_file=False."""
    file_name = tmp_path / "test_spiral.csv"
    
    # Primera ejecución: crea el archivo
    make_spiral_dataset(file_name=file_name, overwrite_file=False, verbose=True, plot_dataset=False)
    assert file_name.exists()
    
    capsys.readouterr() # Limpiamos la salida capturada
    
    # Segunda ejecución: no debe guardarlo
    make_spiral_dataset(file_name=file_name, overwrite_file=False, verbose=True, plot_dataset=False)
    
    captured = capsys.readouterr()
    assert "ya existe. Saltando guardado" in captured.out

def test_make_spiral_dataset_overwrite(tmp_path, capsys):
    """Comprueba que SÍ sobreescribe el archivo si overwrite_file=True."""
    file_name = tmp_path / "test_spiral.csv"
    
    make_spiral_dataset(file_name=file_name, overwrite_file=False, verbose=True, plot_dataset=False)
    assert file_name.exists()
    
    capsys.readouterr()
    
    make_spiral_dataset(file_name=file_name, overwrite_file=True, verbose=True, plot_dataset=False)
    
    captured = capsys.readouterr()
    assert "Dataset guardado en" in captured.out

@patch('create_spiral.plt.show')
@patch('create_spiral.plt.waitforbuttonpress')
@patch('create_spiral.plt.scatter')
@patch('create_spiral.plt.close')
def test_make_spiral_dataset_plot(mock_close, mock_scatter, mock_wait, mock_show, tmp_path):
    """Verifica que la función de plotteo se llame adecuadamente sin levantar ventanas."""
    file_name = tmp_path / "test_spiral.csv"
    
    make_spiral_dataset(file_name=file_name, verbose=False, plot_dataset=True)
    
    mock_scatter.assert_called_once()
    mock_show.assert_called_once()
    mock_wait.assert_called_once()
    mock_close.assert_called_once()
