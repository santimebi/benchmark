import pytest
import numpy as np
from unittest.mock import patch
from 2_split_dataset import split_dataset, save_all_splits

@pytest.fixture
def dummy_spiral_csv(tmp_path):
    file_path = tmp_path / "dummy_spiral.csv"
    data = []
    # Generamos un dataset ficticio de 1200 puntos (400 por clase)
    for class_id in range(3):
        for i in range(400):
            # Para simular la distancia, simplemente usaremos i como x1
            data.append([float(i), float(class_id), class_id])
    
    data = np.array(data)
    np.savetxt(file_path, data, delimiter=",", header="x1,x2,label", comments="")
    return file_path

def test_split_dataset_sizes(dummy_spiral_csv):
    """Comprueba que los splits tengan el tamaño adecuado."""
    X_retain, X_forget, X_val, X_test, y_retain, y_forget, y_val, y_test = split_dataset(
        file_name=dummy_spiral_csv, verbose=False, plot_dataset=False
    )
    
    assert len(X_test) == 240
    assert len(y_test) == 240
    assert len(X_val) == 96
    assert len(y_val) == 96
    assert len(X_forget) == 172
    assert len(y_forget) == 172
    assert len(X_retain) == 692
    assert len(y_retain) == 692

def test_forget_set_exclusivity(dummy_spiral_csv):
    """Asegura que el bloque 'forget' SOLO tenga muestras de la clase 0."""
    *_, y_forget, _, _ = split_dataset(file_name=dummy_spiral_csv, verbose=False, plot_dataset=False)
    
    assert len(y_forget) > 0
    assert np.all(y_forget == 0)

def test_split_reproducibility(dummy_spiral_csv):
    """Comprueba que una misma seed produce el mismo split."""
    res1 = split_dataset(file_name=dummy_spiral_csv, random_state=42, verbose=False, plot_dataset=False)
    res2 = split_dataset(file_name=dummy_spiral_csv, random_state=42, verbose=False, plot_dataset=False)
    
    for arr1, arr2 in zip(res1, res2):
        np.testing.assert_array_equal(arr1, arr2)

def test_split_different_seeds(dummy_spiral_csv):
    """Comprueba que diferentes seeds produzcan splits distintos."""
    X_ret1, *_ = split_dataset(file_name=dummy_spiral_csv, random_state=0, verbose=False, plot_dataset=False)
    X_ret2, *_ = split_dataset(file_name=dummy_spiral_csv, random_state=1, verbose=False, plot_dataset=False)
    
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(X_ret1, X_ret2)

@patch('split_dataset.plt.show')
@patch('split_dataset.plt.scatter')
@patch('split_dataset.plt.figure')
@patch('split_dataset.plt.legend')
@patch('split_dataset.plt.close')
def test_split_dataset_plot(mock_close, mock_legend, mock_figure, mock_scatter, mock_show, dummy_spiral_csv):
    """Verifica que la función grafique sin problemas si plot_dataset=True."""
    split_dataset(file_name=dummy_spiral_csv, verbose=False, plot_dataset=True)
    
    # Comprobamos que al menos se han llamado las funciones de matplotlib
    assert mock_figure.called
    assert mock_scatter.call_count >= 2  # Una vez para Retain, otra para Forget
    mock_legend.assert_called_once()
    mock_show.assert_called_once()
    mock_close.assert_called_once()

def test_save_all_splits(dummy_spiral_csv, tmp_path):
    """Verifica que la función de guardado empaquete correctamente en .npz"""
    output_dir = tmp_path / "datasets"
    seeds = [0, 1]
    
    save_all_splits(file_name=dummy_spiral_csv, output_dir=output_dir, seeds=seeds)
    
    for seed in seeds:
        file_path = output_dir / f"spiral_splits_seed_{seed}.npz"
        assert file_path.exists()
        
        data = np.load(file_path)
        expected_keys = ['X_retain', 'y_retain', 'X_forget', 'y_forget', 'X_val', 'y_val', 'X_test', 'y_test']
        for key in expected_keys:
            assert key in data
        
        assert len(data['X_test']) == 240
