"""
utils/config.py
───────────────────────────────────────────────
Configuración centralizada del proyecto benchmark.

Define las rutas y constantes globales que consumen los scripts
del pipeline (generación de datos, splits, búsqueda de HP y entrenamiento).
"""

from pathlib import Path

# Raíz del proyecto (benchmark/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ruta donde se almacenan los datasets generados (spiral.csv, splits .npz)
DATASETS_PATH = PROJECT_ROOT / "datasets"