"""
Tests para 2_split_dataset.py — split_dataset() y save_all_splits()
Valida los tamaños de los splits, la exclusividad del forget set,
la reproducibilidad, la visualización y el guardado en .npz.
"""

import sys
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import numpy as np

# Cargamos el módulo dinámicamente porque el nombre empieza por un dígito.
_spec = importlib.util.spec_from_file_location(
    "split_dataset",
    str(Path(__file__).parent.parent / "2_split_dataset.py"),
)
split_module = importlib.util.module_from_spec(_spec)
sys.modules["split_dataset"] = split_module
_spec.loader.exec_module(split_module)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────
@pytest.fixture
def dummy_spiral_csv(tmp_path):
    """Genera un CSV ficticio con 1200 puntos (400 por clase)."""
    file_path = tmp_path / "dummy_spiral.csv"
    data = []
    for class_id in range(3):
        for i in range(400):
            data.append([float(i), float(class_id), class_id])

    data = np.array(data)
    np.savetxt(file_path, data, delimiter=",", header="x1,x2,label", comments="")
    return file_path


# ─────────────────────────────────────────────
# Tamaños de los splits
# ─────────────────────────────────────────────
class TestSplitDataset_Sizes:

    def test_total_samples_preserved(self, dummy_spiral_csv):
        """La suma de todos los splits debe ser igual al total del dataset."""
        X_ret, X_fgt, X_val, X_tst, *_ = split_module.split_dataset(
            file_name=dummy_spiral_csv, verbose=False, plot_dataset=False
        )
        total = len(X_ret) + len(X_fgt) + len(X_val) + len(X_tst)
        assert total == 1200

    def test_test_size(self, dummy_spiral_csv):
        """Test set debe ser ~20% del total."""
        *_, X_test, _, _, _, _ = split_module.split_dataset(
            file_name=dummy_spiral_csv, verbose=False, plot_dataset=False
        )
        assert len(X_test) == 240

    def test_no_empty_splits(self, dummy_spiral_csv):
        """Ningún split debe quedar vacío."""
        result = split_module.split_dataset(
            file_name=dummy_spiral_csv, verbose=False, plot_dataset=False
        )
        for arr in result:
            assert len(arr) > 0


# ─────────────────────────────────────────────
# Forget set exclusividad
# ─────────────────────────────────────────────
class TestSplitDataset_ForgetExclusivity:

    def test_forget_only_class_0(self, dummy_spiral_csv):
        """El forget set debe contener SOLO muestras de la clase 0."""
        *_, y_forget, _, _ = split_module.split_dataset(
            file_name=dummy_spiral_csv, verbose=False, plot_dataset=False
        )
        assert len(y_forget) > 0
        assert np.all(y_forget == 0)

    def test_forget_sorted_by_distance(self, dummy_spiral_csv):
        """Las muestras del forget deben estar ordenadas por cercanía al origen."""
        X_ret, X_fgt, *_ = split_module.split_dataset(
            file_name=dummy_spiral_csv, verbose=False, plot_dataset=False
        )
        distances = np.linalg.norm(X_fgt, axis=1)
        # Verificar que están ordenadas de menor a mayor distancia
        assert np.all(distances[:-1] <= distances[1:])


# ─────────────────────────────────────────────
# Reproducibilidad
# ─────────────────────────────────────────────
class TestSplitDataset_Reproducibility:

    def test_same_seed_same_split(self, dummy_spiral_csv):
        """Misma seed produce el mismo split."""
        res1 = split_module.split_dataset(
            file_name=dummy_spiral_csv, random_state=42,
            verbose=False, plot_dataset=False
        )
        res2 = split_module.split_dataset(
            file_name=dummy_spiral_csv, random_state=42,
            verbose=False, plot_dataset=False
        )
        for a1, a2 in zip(res1, res2):
            np.testing.assert_array_equal(a1, a2)

    def test_different_seeds_different_splits(self, dummy_spiral_csv):
        """Seeds distintas producen splits distintos."""
        X_ret1, *_ = split_module.split_dataset(
            file_name=dummy_spiral_csv, random_state=0,
            verbose=False, plot_dataset=False
        )
        X_ret2, *_ = split_module.split_dataset(
            file_name=dummy_spiral_csv, random_state=1,
            verbose=False, plot_dataset=False
        )
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(X_ret1, X_ret2)


# ─────────────────────────────────────────────
# Visualización
# ─────────────────────────────────────────────
class TestSplitDataset_Plot:

    @patch.object(split_module.plt, 'close')
    @patch.object(split_module.plt, 'legend')
    @patch.object(split_module.plt, 'show')
    @patch.object(split_module.plt, 'scatter')
    @patch.object(split_module.plt, 'figure')
    def test_plot_calls_matplotlib(self, mock_fig, mock_scatter, mock_show,
                                    mock_legend, mock_close, dummy_spiral_csv):
        """Con plot_dataset=True se invocan las funciones de matplotlib."""
        split_module.split_dataset(
            file_name=dummy_spiral_csv, verbose=False, plot_dataset=True
        )
        assert mock_fig.called
        assert mock_scatter.call_count >= 2
        mock_legend.assert_called_once()
        mock_show.assert_called_once()
        mock_close.assert_called_once()


# ─────────────────────────────────────────────
# save_all_splits
# ─────────────────────────────────────────────
class TestSaveAllSplits:

    def test_creates_npz_files(self, dummy_spiral_csv, tmp_path):
        """Debe crear un .npz por cada seed."""
        output_dir = tmp_path / "datasets"
        seeds = [0, 1]
        split_module.save_all_splits(
            file_name=dummy_spiral_csv, output_dir=output_dir,
            seeds=seeds, verbose=False, plot_dataset=False,
        )
        for seed in seeds:
            assert (output_dir / f"spiral_splits_seed_{seed}.npz").exists()

    def test_npz_contains_all_keys(self, dummy_spiral_csv, tmp_path):
        """El .npz debe contener las 8 arrays esperadas."""
        output_dir = tmp_path / "datasets"
        split_module.save_all_splits(
            file_name=dummy_spiral_csv, output_dir=output_dir,
            seeds=[0], verbose=False, plot_dataset=False,
        )
        data = np.load(output_dir / "spiral_splits_seed_0.npz")
        expected = {"X_retain", "y_retain", "X_forget", "y_forget",
                    "X_val", "y_val", "X_test", "y_test"}
        assert expected == set(data.files)

    def test_no_overwrite_by_default(self, dummy_spiral_csv, tmp_path, capsys):
        """No regenera si el .npz ya existe y overwrite_file=False."""
        output_dir = tmp_path / "datasets"
        split_module.save_all_splits(
            file_name=dummy_spiral_csv, output_dir=output_dir,
            seeds=[0], overwrite_file=True, verbose=True, plot_dataset=False,
        )
        capsys.readouterr()
        split_module.save_all_splits(
            file_name=dummy_spiral_csv, output_dir=output_dir,
            seeds=[0], overwrite_file=False, verbose=True, plot_dataset=False,
        )
        captured = capsys.readouterr()
        assert "ya existe" in captured.out
