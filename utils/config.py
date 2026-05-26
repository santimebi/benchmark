"""
utils/config.py
───────────────────────────────────────────────
Configuración centralizada del proyecto benchmark.

Define las rutas y constantes globales que consumen los scripts
del pipeline (generación de datos, splits, búsqueda de HP y entrenamiento).
"""

import os
from pathlib import Path

# Raíz del proyecto (benchmark/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directorio raíz para datos, modelos y resultados.
# Si está definida la variable de entorno BENCHMARK_DATA_DIR, se usa esa ruta.
# De lo contrario, se usa la raíz del proyecto (comportamiento local por defecto).
BASE_DATA_DIR = Path(os.getenv("BENCHMARK_DATA_DIR", PROJECT_ROOT))

# Rutas donde se almacenan los datasets, modelos, resultados y descargas
DATASETS_PATH = BASE_DATA_DIR / "datasets"
MODELS_PATH = BASE_DATA_DIR / "models"
RESULTS_PATH = BASE_DATA_DIR / "results"
DATA_PATH = BASE_DATA_DIR / "data"

# Asegurar la creación de los directorios necesarios
DATASETS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)
(MODELS_PATH / "weights").mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
DATA_PATH.mkdir(parents=True, exist_ok=True)