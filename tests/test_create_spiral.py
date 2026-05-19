"""
Tests para 1_create_spiral.py — make_spiral_dataset()
Valida la generación del dataset espiral: dimensiones, reproducibilidad,
guardado/sobreescritura y visualización.
"""

import sys
import importlib.util
from pathlib import Path

import pytest
import numpy as np
from unittest.mock import patch

# Cargamos el módulo dinámicamente para consistencia con el resto de tests.
_spec = importlib.util.spec_from_file_location(
    "create_spiral",
    str(Path(__file__).parent.parent / "1_create_spiral.py"),
)
create_spiral = importlib.util.module_from_spec(_spec)
sys.modules["create_spiral"] = create_spiral
_spec.loader.exec_module(create_spiral)

make_spiral_dataset = create_spiral.make_spiral_dataset


# ─────────────────────────────────────────────
# Forma y contenido del dataset
# ─────────────────────────────────────────────
class TestMakeSpiralDataset_Shape:

    def test_shapes(self, tmp_path):
        """X debe ser (N*C, 2) e y debe ser (N*C,)."""
        n_points = 100
        n_classes = 3
        X, y = make_spiral_dataset(
            file_name=tmp_path / "spiral.csv",
            n_points_per_class=n_points,
            n_classes=n_classes,
            verbose=False, plot_dataset=False,
        )
        assert X.shape == (n_points * n_classes, 2)
        assert y.shape == (n_points * n_classes,)

    def test_all_classes_present(self, tmp_path):
        """Todas las clases deben estar presentes en y."""
        n_classes = 3
        _, y = make_spiral_dataset(
            file_name=tmp_path / "spiral.csv",
            n_classes=n_classes,
            verbose=False, plot_dataset=False,
        )
        assert set(np.unique(y)) == set(range(n_classes))

    def test_custom_class_count(self, tmp_path):
        """Debe funcionar con un número distinto de clases."""
        _, y = make_spiral_dataset(
            file_name=tmp_path / "spiral.csv",
            n_classes=5, n_points_per_class=50,
            verbose=False, plot_dataset=False,
        )
        assert set(np.unique(y)) == {0, 1, 2, 3, 4}


# ─────────────────────────────────────────────
# Reproducibilidad
# ─────────────────────────────────────────────
class TestMakeSpiralDataset_Reproducibility:

    def test_same_seed_same_result(self, tmp_path):
        """Misma seed produce el mismo dataset."""
        f1 = tmp_path / "s1.csv"
        f2 = tmp_path / "s2.csv"
        X1, y1 = make_spiral_dataset(file_name=f1, random_state=42, verbose=False, plot_dataset=False)
        X2, y2 = make_spiral_dataset(file_name=f2, random_state=42, verbose=False, plot_dataset=False)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seed_different_result(self, tmp_path):
        """Seeds distintas producen datasets distintos."""
        f1 = tmp_path / "s1.csv"
        f2 = tmp_path / "s2.csv"
        X1, _ = make_spiral_dataset(file_name=f1, random_state=0, verbose=False, plot_dataset=False)
        X2, _ = make_spiral_dataset(file_name=f2, random_state=1, verbose=False, plot_dataset=False)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(X1, X2)


# ─────────────────────────────────────────────
# Guardado de ficheros
# ─────────────────────────────────────────────
class TestMakeSpiralDataset_FileIO:

    def test_creates_csv(self, tmp_path):
        """Debe crear el CSV cuando no existe."""
        f = tmp_path / "spiral.csv"
        assert not f.exists()
        make_spiral_dataset(file_name=f, verbose=False, plot_dataset=False)
        assert f.exists()

    def test_csv_has_correct_columns(self, tmp_path):
        """El CSV generado debe tener 3 columnas (x1, x2, label)."""
        f = tmp_path / "spiral.csv"
        make_spiral_dataset(file_name=f, verbose=False, plot_dataset=False)
        data = np.loadtxt(f, delimiter=",", skiprows=1)
        assert data.shape[1] == 3

    def test_no_overwrite_by_default(self, tmp_path, capsys):
        """No sobreescribe si overwrite_file=False."""
        f = tmp_path / "spiral.csv"
        make_spiral_dataset(file_name=f, overwrite_file=False, verbose=True, plot_dataset=False)
        capsys.readouterr()
        make_spiral_dataset(file_name=f, overwrite_file=False, verbose=True, plot_dataset=False)
        captured = capsys.readouterr()
        assert "ya existe. Saltando guardado" in captured.out

    def test_overwrite_when_requested(self, tmp_path, capsys):
        """Sí sobreescribe con overwrite_file=True."""
        f = tmp_path / "spiral.csv"
        make_spiral_dataset(file_name=f, overwrite_file=False, verbose=True, plot_dataset=False)
        capsys.readouterr()
        make_spiral_dataset(file_name=f, overwrite_file=True, verbose=True, plot_dataset=False)
        captured = capsys.readouterr()
        assert "Dataset guardado en" in captured.out


# ─────────────────────────────────────────────
# Visualización
# ─────────────────────────────────────────────
class TestMakeSpiralDataset_Plot:

    @patch.object(create_spiral.plt, 'close')
    @patch.object(create_spiral.plt, 'waitforbuttonpress')
    @patch.object(create_spiral.plt, 'show')
    @patch.object(create_spiral.plt, 'scatter')
    def test_plot_calls_matplotlib(self, mock_scatter, mock_show, mock_wait, mock_close, tmp_path):
        """Con plot_dataset=True se invocan las funciones de matplotlib."""
        make_spiral_dataset(file_name=tmp_path / "s.csv", verbose=False, plot_dataset=True)
        mock_scatter.assert_called_once()
        mock_show.assert_called_once()
        mock_wait.assert_called_once()
        mock_close.assert_called_once()
