"""
utils/hp_spaces.py
───────────────────────────────────────────────
Definición de los espacios de búsqueda y objetivos para la optimización de
hiperparámetros (Optuna).
"""

from pathlib import Path
from typing import Dict, Any
import optuna

HP_SPACES: Dict[str, Dict[str, Any]] = {
    "standard": {
        "suggest_fn": lambda trial: {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [8, 16, 32, 64]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "epochs": trial.suggest_int("epochs", 50, 300, step=50),
        },
        "objective_type": "val_loss",
        "output_path": Path("models/best_hp.json"),
    },
    "cfk": {
        "suggest_fn": lambda trial: {
            "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            "k": trial.suggest_int("k", 1, 3),
            "epochs": 20,  # Fijo a 20 épocas para el desaprendizaje
        },
        "objective_type": "unlearning_loss",
        "output_path": Path("models/best_cfk_hp.json"),
    }
}
